# 使用llama-index实现RAG

结合llama-index和BCE的embedding与rerank模型实现

## llama-index安装
```
pip install llama-index==0.9.42.post2
```

## BCE相关库安装
BCEembedding仓库：https://github.com/netease-youdao/BCEmbedding/blob/master/README_zh.md

```
git clone git@github.com:netease-youdao/BCEmbedding.git
cd BCEmbedding
pip install -v -e .
```




## 代码

加载Embedding和rerank本地模型，需要从Huggingface/modelscope下载
### Embedding模型
https://modelscope.cn/models/maidalun/bce-embedding-base_v1

### reranker模型
https://modelscope.cn/models/maidalun/bce-reranker-base_v1

### LLM模型
硅基流动7B免费模型（替换模型/api_key）
```
from llama_index.llms.openai_like import OpenAILike
llm = OpenAILike(
    model="Qwen/Qwen2.5-7B-Instruct",
    api_base="https://api.siliconflow.cn/v1",
    is_chat_model=True,
    api_key=""
)
```

### 自定义切分策略(word按标题章节编写)
```
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

```


### 完整代码见bce_reranker_demo.py