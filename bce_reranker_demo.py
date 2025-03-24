import os
import re
from BCEmbedding.tools.llama_index import BCERerank
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader, Document
from llama_index.node_parser import SimpleNodeParser, TokenTextSplitter
# from llama_index.llms import OpenAI
from llama_index.retrievers import VectorIndexRetriever


# LLM设置
llm_model = "Qwen/Qwen2.5-7B-Instruct"
api_key = "sk-qgtkfrjfctkkfsdojemwxrpugqjkjxvafekaizzpbwwrqucy"
api_base="https://api.siliconflow.cn/v1"

if llm_model=='' or api_key=='' or api_base=='':
    print("模型调用参数未设置。")
    exit(0)
else:
    print("模型调用参数以已设置。")

# embedding/reranker模型参数设置
embedding_model_path = r'G:\Huggingface\bce-embedding-base_v1'
reranker_model_path = r'G:\Huggingface\bce-reranker-base_v1'

# llamaindex加载知识库
if any(os.path.isfile(os.path.join('./data', f)) for f in os.listdir('./data')):
    docs = SimpleDirectoryReader('./data').load_data()
else:
    print("目录下没有文件")
    exit(0)

from llama_index.llms.openai_like import OpenAILike
llm = OpenAILike(
    model=llm_model,
    api_base=api_base,
    is_chat_model=True,
    api_key=api_key
)

class CustomTitleSplitter(TokenTextSplitter):
    def split_text(self, text):
        # 使用正则表达式按标题分割，每段以标题开头，换行符分隔
        sections = re.split(r'\n\n\n\n', text.strip())
        chunks = []
        current_chunk = ""

        for section in sections:
            if section.strip():  # 确保不处理空行
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = section
        if current_chunk:  # 添加最后一段
            chunks.append(current_chunk.strip())
        return chunks
splitter = CustomTitleSplitter()

# 加载本地embedding模型与reranker模型

embed_args = {'model_name': embedding_model_path, 'max_length': 512, 'embed_batch_size': 32, 'device': 'cuda'}
embed_model = HuggingFaceEmbedding(**embed_args)
reranker_args = {'model': reranker_model_path, 'top_n': 1, 'device': 'cuda'}
reranker_model = BCERerank(**reranker_args)

# 配置
service_context = ServiceContext.from_defaults(embed_model=embed_model)


sections = splitter.split_text(docs[0].text)
documents = [Document(text=t) for t in sections]
index = VectorStoreIndex.from_documents(documents,service_context=service_context)


# 测试召回
# query = "在党政机关聚集中出现的群体性事件"
# query = "在政府机关聚集，怎么处理"
query = "在政府门口有人聚集维权怎么处置"
# query = "报警人称发现一具尸体，如何处置"
# query = "五六十人聚集在县委大门口，不知道干啥呢，应该如何处置"
# results = retriever.retrieve(query)

# 创建检索引擎
# embedding召回相关片段
retriever = index.as_retriever(similarity_top_k=3)
results = retriever.retrieve(query)

# 通过reranker重排序
retrieval_by_reranker = reranker_model.postprocess_nodes(results, query_str=query)



#查看最终召回的知识库文本
chunks_context = ''
for result in results:
    # chunks_context += result.node.text + '\n\n'
    chunks_context += result.node.text + '\n\n'

# 编写提示词
prompt = f"""
<请按照以下步骤完成RAG流程>
<step 1>:用户的提问是<{query}>;
<step 2>:从知识库检索召回的处置规范如下：
{chunks_context};
<step 3>:如果上述处置规范不适用问题<{query}>，则返回“没有找到的处置规范！”；
    反之，选择与问题最合适的警情处置规范（包含标题和具体处置规范），回答用户的提问，不要修改知识库内容；
"""
print(prompt)

# 提示词给LLM并流式输出
completions = llm.stream_complete(prompt, formatted=True)
for completion in completions:
    print(completion.delta, end="")






