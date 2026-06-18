from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, pdf_loader, txt_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
import os


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )
        self._bm25_retriever = None

    def get_retriever(self):
        if chroma_conf.get("enable_hybrid_search", False):
            return self.get_hybrid_retriever()
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def _load_bm25_documents(self) -> list[Document]:
        data = self.vector_store._collection.get()
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or [{}] * len(documents)
        return [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(documents, metadatas)
        ]

    def get_hybrid_retriever(self):
        vector_k = chroma_conf.get("hybrid_k", 10)
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": vector_k})

        docs = self._load_bm25_documents()
        if not docs:
            logger.warning("[hybrid] 向量库为空，降级为纯向量检索")
            return vector_retriever

        try:
            if self._bm25_retriever is None:
                self._bm25_retriever = BM25Retriever.from_documents(docs)
            self._bm25_retriever.k = vector_k

            weights = chroma_conf.get("hybrid_weights", [0.5, 0.5])
            return EnsembleRetriever(
                retrievers=[vector_retriever, self._bm25_retriever],
                weights=weights,
            )
        except ImportError as e:
            logger.warning(f"[hybrid] BM25 不可用，降级为纯向量检索: {e}")
            return vector_retriever

    def load_document(self) -> tuple[int, int]:
        """增量加载知识库，返回 (skipped, indexed) 计数。"""
        collection_count = len(self.vector_store._collection.get().get("documents") or [])
        md5_path = get_abs_path(chroma_conf["md5_hex_store"])
        if collection_count == 0 and os.path.exists(md5_path) and os.path.getsize(md5_path) > 0:
            logger.warning("[加载知识库] 向量库为空但存在 MD5 记录，将重新索引")
            open(md5_path, "w", encoding="utf-8").close()

        def check_md5_hex(md5_for_check: str) -> bool:
            md5_path = get_abs_path(chroma_conf["md5_hex_store"])
            if not os.path.exists(md5_path):
                open(md5_path, "w", encoding="utf-8").close()
                return False

            with open(md5_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5_for_check:
                        return True
            return False

        def save_md5_hex(md5_for_check: str) -> None:
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str) -> list[Document]:
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        skipped = 0
        indexed = 0
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                skipped += 1
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                self._bm25_retriever = None
                indexed += 1
                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)

        return skipped, indexed


def ensure_knowledge_indexed() -> tuple[int, int]:
    service = VectorStoreService()
    return service.load_document()


if __name__ == "__main__":
    skipped, indexed = ensure_knowledge_indexed()
    print(f"索引完成：跳过 {skipped} 个，新增 {indexed} 个")

    vs = VectorStoreService()
    retriever = vs.get_retriever()
    res = retriever.invoke("小户型适合哪些扫地机器人")
    for r in res:
        print(r.page_content)
        print("-" * 20)
