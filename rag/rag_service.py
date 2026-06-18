"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()
        self.rewrite_template = PromptTemplate.from_template(
            "将用户问题改写为适合知识库检索的简洁关键词，只输出检索词，不要解释。\n用户问题：{query}"
        )
        self.rewrite_chain = self.rewrite_template | self.model | StrOutputParser()

    def _init_chain(self):
        return self.prompt_template | self.model | StrOutputParser()

    def rewrite_query(self, query: str) -> str:
        if not chroma_conf.get("enable_query_rewrite", True):
            return query
        try:
            rewritten = self.rewrite_chain.invoke({"query": query}).strip()
            logger.debug(f"[rag] query rewrite: {query} -> {rewritten}")
            return rewritten or query
        except Exception as e:
            logger.warning(f"[rag] query rewrite failed: {e}")
            return query

    def retriever_docs(self, query: str) -> list[Document]:
        search_query = self.rewrite_query(query)
        docs = self.retriever.invoke(search_query)
        final_k = chroma_conf["k"]
        return docs[:final_k]

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)

        context = ""
        for counter, doc in enumerate(context_docs, start=1):
            context += (
                f"【参考资料{counter}】: 参考资料：{doc.page_content} "
                f"| 参考元数据：{doc.metadata}\n"
            )

        prompt_value = self.prompt_template.invoke({"input": query, "context": context})
        logger.debug(f"[rag] prompt length={len(prompt_value.to_string())}")

        return self.chain.invoke({"input": query, "context": context})


if __name__ == "__main__":
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
