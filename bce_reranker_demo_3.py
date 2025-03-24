import os
import re
from BCEmbedding.tools.llama_index import BCERerank
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader, Document
from llama_index.node_parser import SimpleNodeParser, TokenTextSplitter
# from llama_index.llms import OpenAI
from llama_index.retrievers import VectorIndexRetriever

from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="Qwen/Qwen2.5-7B-Instruct",
    api_base="https://api.siliconflow.cn/v1",
    is_chat_model=True,
    api_key="sk-qgtkfrjfctkkfsdojemwxrpugqjkjxvafekaizzpbwwrqucy"
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


embedding_model_path = r'G:\Huggingface\bce-embedding-base_v1'
reranker_model_path = r'G:\Huggingface\bce-reranker-base_v1'
embed_args = {'model_name': embedding_model_path, 'max_length': 512, 'embed_batch_size': 32, 'device': 'cuda'}
embed_model = HuggingFaceEmbedding(**embed_args)

reranker_args = {'model': reranker_model_path, 'top_n': 1, 'device': 'cuda'}
reranker_model = BCERerank(**reranker_args)

service_context = ServiceContext.from_defaults(embed_model=embed_model)

# documents = SimpleDirectoryReader(input_files=['./data/data_1.docx']).load_data()
# documents = SimpleDirectoryReader('./data - 副本').load_data()
# docs = SimpleDirectoryReader('./data - 副本').load_data()
docs = SimpleDirectoryReader('./data').load_data()
sections = splitter.split_text(docs[0].text)
documents = [Document(text=t) for t in sections]
index = VectorStoreIndex.from_documents(documents,service_context=service_context)

# node_parser = SimpleNodeParser.from_defaults(chunk_size=400, chunk_overlap=80)
# nodes = node_parser.get_nodes_from_documents(documents[0:36])
# index = VectorStoreIndex(nodes, service_context=service_context)

# 创建检索引擎
query_str = "五六十人聚集在县委大门口，不知道干啥呢，应该如何处置"
retriever = index.as_retriever(similarity_top_k=3)
results = retriever.retrieve(query_str)
retrieval_by_reranker = reranker_model.postprocess_nodes(results, query_str=query_str)

# vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10, service_context=service_context)
# retrieval_by_embedding = vector_retriever.retrieve("在政府门口有人聚集维权怎么处置")
# retrieval_by_reranker = reranker_model.postprocess_nodes(retrieval_by_embedding, query_str="报警人称发现一具尸体，如何处置")



# 测试召回
# query = "在党政机关聚集中出现的群体性事件"
# query = "在政府机关聚集，怎么处理"
# query = "在政府门口有人聚集维权怎么处置"
query = "报警人称发现一具尸体，如何处置"
# query = "五六十人聚集在县委大门口，不知道干啥呢，应该如何处置"
# results = retriever.retrieve(query)


chunks_context = ''
for result in results:
    # chunks_context += result.node.text + '\n\n'
    chunks_context += result.node.text + '\n\n'

# print(chunks_context)
# # 输出结果
# for result in results:
#     print(f"Score: {result.score}, Content: {result.node.text}")
#
#
prompt = f"""
<请按照以下步骤完成RAG流程>
<step 1>:用户的提问是<{query}>;
<step 2>:从知识库检索召回的处置规范如下：
{chunks_context};
<step 3>:如果上述处置规范不适用问题<{query}>，则返回“没有找到的处置规范！”；
    反之，选择与问题最合适的警情处置规范（包含标题和具体处置规范），回答用户的提问，不要修改知识库内容；
"""
print(prompt)

# response = llm.complete(prompt)
# print(str(response))
completions = llm.stream_complete(prompt, formatted=True)
for completion in completions:
    print(completion.delta, end="")






