import jieba
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredHTMLLoader,
)  # 从 langchain_community.document_loaders 模块中导入各种类型文档加载器类
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)  # 通用递归字符分块器，按分隔符层级递归切分，对中英文混排文本鲁棒
from pymilvus import MilvusClient, DataType
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from logger import logger
from typing import List, Tuple
import os
import torch

import agent_full

logger = logger()

"""
一、索引流程
支持 PDF / Word / PPT / Excel / CSV / Markdown / XML / HTML / TXT 等多种文档格式，
通过 langchain_community 的文档加载器解析为文本，再用 RecursiveCharacterTextSplitter
分割为每块 512 字符、重叠 128 字符的文本块，并由 bge-small-zh-v1.5 嵌入模型转化为
归一化嵌入向量，最后写入 Milvus Lite 本地库（IP/FLAT 索引，配合归一化向量即余弦相似度）。
"""

# Milvus Lite 本地库文件（相对 main.py 所在目录）；bge-small-zh-v1.5 输出 512 维向量
MILVUS_URI = os.path.join(os.path.dirname(__file__), "main_milvus.db")
COLLECTION_NAME = "rag_chunks"  # 向量数据库 collection 名
VECTOR_DIM = 512  # 向量维度

# 重排序模型：优先本地目录，缺失则从 HuggingFace 拉取到 HF 缓存
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "bge-reranker-v2-m3")

# 文档扩展名 → (加载器类, 加载参数) 映射，提为模块级常量避免每次调用重建
DOCUMENT_LOADER_MAPPING = {
    ".pdf": (PDFPlumberLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".csv": (CSVLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".xml": (UnstructuredXMLLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
}

# 嵌入模型单例：懒加载，索引与检索共用同一实例，避免反复从磁盘加载权重
_embedding_model: SentenceTransformer = None


def _get_embedding_model(model_name: str = "bge-small-zh-v1.5") -> SentenceTransformer:
    """懒加载嵌入模型单例，首次调用时加载权重，后续复用同一实例。"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            os.path.join(os.path.dirname(__file__), model_name)
        )
    return _embedding_model


# 重排序模型单例：懒加载，避免每个查询都重载几百 MB 权重
_reranker_model = None
_reranker_tokenizer = None


def _get_reranker():
    """懒加载 bge-reranker-v2-m3 cross-encoder 单例。
    优先从本地目录加载，缺失则回退到 HuggingFace 仓库名（首次联网下载到 HF 缓存）。
    :return: (model, tokenizer)
    """
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is None:
        # 本地目录存在则用本地，否则用 HF 仓库名（transformers 自动联网下载到缓存）
        model_path = (
            RERANKER_LOCAL_DIR
            if os.path.isdir(RERANKER_LOCAL_DIR)
            else RERANKER_MODEL_NAME
        )
        logger.debug(f"加载重排序模型: {model_path}")
        _reranker_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _reranker_model.eval()
    return _reranker_model, _reranker_tokenizer


# =========== 1.文档加载和解析 ===========
def load_document(file_path: str) -> str:
    """
    按扩展名选择对应的 langchain 加载器解析文档，返回拼接后的纯文本内容。
    :param file_path: 文档文件路径
    :return: 文档内容字符串；解析失败或不支持的格式返回空字符串
    """
    ext = os.path.splitext(file_path)[1]  # 获取文件扩展名，确定文档类型
    loader_tuple = DOCUMENT_LOADER_MAPPING.get(ext)  # 取对应加载器类与参数

    if not loader_tuple:  # 不支持的格式
        logger.warning(f"不支持的文档类型: '{ext}'，跳过 {file_path}")
        return ""

    loader_class, loader_args = loader_tuple
    try:
        loader = loader_class(file_path, **loader_args)  # 创建加载器实例
        documents = loader.load()  # 加载文档
        content = "\n".join([doc.page_content for doc in documents])  # 多页内容拼合
        logger.debug(f"文档 {file_path} 解析完成，共 {len(content)} 字符，预览: {content[:100]}...")
        return content
    except Exception as e:  # 单个文件解析失败不应中断整个批次
        logger.error(f"文档 {file_path} 解析失败: {e}")
        return ""


# ============= 2.文档分块 ===========
def split_text(text: str, chunk_size: int = 512, chunk_overlap: int = 128) -> List[str]:
    """
    使用 LangChain 的 RecursiveCharacterTextSplitter 对文本进行分块。
    :param text: 源文本
    :param chunk_size: Chunk 大小（字符数，非 token）
    :param chunk_overlap: Chunk 重叠部分大小
    :return: chunks List[str]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: List[str] = splitter.split_text(text)
    logger.debug(f"分割的文本块数量: {len(chunks)}")
    return chunks


# ============ 3. 向量化与索引 =====================
def embed_by_sentence_transformer(
    chunks: List[str], model_name: str = "bge-small-zh-v1.5"
) -> List:
    """
    将文本块列表批量转化为归一化嵌入向量。
    normalize_embeddings=True 对向量归一化，配合 IP 索引即等价余弦相似度。
    :param chunks: 分割后的文本块列表
    :param model_name: 本地嵌入模型目录名
    :return: 嵌入向量列表（顺序与 chunks 一致）
    """
    model = _get_embedding_model(model_name)
    # 批量 encode：一次传入全部 chunk，比逐条 encode 效率高得多
    embeddings = model.encode(chunks, normalize_embeddings=True, batch_size=32)
    return embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)


def embed_and_index(
    chunks: List[str], model_name: str = "bge-small-zh-v1.5"
) -> Tuple[MilvusClient, str]:
    """
    将文本块嵌入向量并写入 Milvus Lite 本地库。
    :param chunks: 文档文本分割成的文本块 Chunk 列表
    :param model_name: 本地嵌入模型目录名
    :return: (MilvusClient, collection_name) —— client 已 load_collection，可直接 search
    """
    logger.debug("嵌入过程开始.")
    embeddings = embed_by_sentence_transformer(chunks, model_name)

    # 建 Milvus Lite client（本地文件存储，无需起服务）
    client = MilvusClient(uri=MILVUS_URI)
    # 每次重新嵌入：若 collection 已存在则先 drop，避免重复数据
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    # 显式 schema：id 自增主键 + 向量 + 原文（类表结构）
    # pymilvus 2.6 已废弃 create_collection(fields=[...]) 旧写法，改用 schema + IndexParams
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)

    # 索引：IP（内积）+ FLAT，配合归一化向量即余弦相似度，等价原 faiss.IndexFlatIP
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",
        metric_type="IP",
    )

    # 建表 + 建索引 + 自动 load（schema 模式一步到位）
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    # 插入：向量 + 原文（id 自增不传）
    data = [{"vector": emb, "text": chunk} for emb, chunk in zip(embeddings, chunks)]
    client.insert(collection_name=COLLECTION_NAME, data=data)
    logger.debug(f"已插入 {len(data)} 条向量。")
    logger.debug("索引过程完成.")
    return client, COLLECTION_NAME


# ===== all ============
# 处理文件夹一批文件，串联 1.文件解析 + 2.文档分块 + 3.向量化 + 4.索引
def indexing_process(
    folder_path: str, embedding_model: str = "bge-small-zh-v1.5"
) -> Tuple[MilvusClient, str, List[str]]:
    """
    索引流程：遍历文件夹下所有文档，逐个解析分块，汇总后批量嵌入并写入 Milvus 库。
    :param folder_path: 文档文件夹路径
    :param embedding_model: 本地嵌入模型目录名
    :return: (MilvusClient, collection_name, all_chunks) —— all_chunks 供 BM25 检索复用；
             文件夹无可用文档时返回 (None, None, [])
    """
    all_chunks: List[str] = []
    chunk_size = 512
    chunk_overlap = 128

    # 遍历文件夹中的所有文档文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):  # 跳过子目录
            continue
        # 1.解析文档文件，获得文档字符串内容
        document_text = load_document(file_path)
        if not document_text:  # 跳过解析失败或不支持的格式
            continue
        # 2. 切割
        chunks = split_text(document_text, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    if not all_chunks:  # 文件夹下无可用文档，避免向 Milvus 写空
        logger.warning(f"文件夹 {folder_path} 下没有可解析的文档。")
        return None, None, []

    # 3.批量嵌入 + 4.写入 Milvus
    client, collection = embed_and_index(all_chunks, embedding_model)
    return client, collection, all_chunks


"""
二、检索（使用混合检索）
用户的查询（Query）被预加载的 bge-small-zh-v1.5 嵌入模型转化为归一化的嵌入向量。
然后，在 Milvus 库中检索与查询向量内积相似度最高的前 top_k 个文本块（向量已归一化，内积即余弦相似度）。
检索结果携带原文字段直接返回，供后续生成过程使用。
"""

def retrieval_process(chunks: List[str],
    query: str,
    client: MilvusClient,
    collection: str,
    top_k: int = 5,
    model_name: str = "bge-small-zh-v1.5",
) -> List[str]:
    """
    混合检索流程：向量检索（语义）+ BM25 检索（关键词）双路召回，
    用 RRF（倒数排名融合）合并两路结果并去重，取最终 top_k 个文本块。
    :param chunks: 全量文档分块列表（BM25 建索引用）
    :param query: 查询语句
    :param client: 已建立的 MilvusClient
    :param collection: collection 名称
    :param top_k: 返回最相似的前 K 个结果
    :param model_name: 本地嵌入模型目录名
    :return: 融合去重后的文本块列表
    """
    logger.debug("检索过程开始.")
    logger.debug(f"查询语句: {query}")

    # 每路召回的候选数应略大于最终 top_k，给融合留余量，避免两路重叠后不足 top_k
    candidate_k = top_k * 3

    # ---- 1. 向量检索（语义召回）----
    query_embeddings = embed_by_sentence_transformer([query], model_name)
    query_vector = query_embeddings[0]  # todo：后续可以多个查询
    res = client.search(
        collection_name=collection,
        data=[query_vector],
        limit=candidate_k,
        output_fields=["text"],
    )
    # 向量召回结果（按 IP 相似度降序）
    vector_chunks = [hit["entity"]["text"] for hit in res[0]]
    for i, hit in enumerate(res[0]):
        c = hit["entity"]["text"].replace("\n", "")
        logger.debug(f"[向量] 文本块 {i}:\n{c}；相似度得分: {hit['distance']}\n")

    # ---- 2. BM25 检索（关键词召回）----
    # 对所有文档分块中文分词，构建 BM25 语料索引
    tokenized_corpus = [list(jieba.cut(doc)) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(tokenized_query)
    # 按相关性降序取前 candidate_k 个文档的索引
    bm25_top_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:candidate_k]
    bm25_chunks = [chunks[i] for i in bm25_top_indices]
    for rank, i in enumerate(bm25_top_indices):
        c = chunks[i].replace("\n", "")
        logger.debug(f"[BM25] 文本块 {rank}:\n{c}；BM25 得分: {bm25_scores[i]}\n")

    # ---- 3. RRF 融合（倒数排名融合）+ 去重 ----
    # RRF 分数 = Σ 1/(k + rank)，按文本块去重累加；用文本块内容本身做 key
    # k=60 是常见经验值，抑制排名靠前项的过度主导，使两路都有话语权
    rrf_k = 60
    rrf_scores: dict[str, float] = {}
    for rank, chunk in enumerate(vector_chunks):
        rrf_scores[chunk] = rrf_scores.get(chunk, 0.0) + 1.0 / (rrf_k + rank)
    for rank, chunk in enumerate(bm25_chunks):
        rrf_scores[chunk] = rrf_scores.get(chunk, 0.0) + 1.0 / (rrf_k + rank)

    # 按 RRF 分数降序取最终 top_k
    combined_results = [
        chunk for chunk, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    ][:top_k]
    for i, chunk in enumerate(combined_results):
        c = chunk.replace("\n", "")
        logger.debug(f"[融合] 文本块 {i}:\n{c}；RRF 得分: {rrf_scores[chunk]}\n")

    logger.debug("检索过程完成.")
    return combined_results



#### 增加重排序：cross-encoder 对检索结果二次精排
def reranking(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """
    用 bge-reranker-v2-m3 cross-encoder 对检索结果做二次精排（语义相关性打分）。
    :param query: 用户查询语句
    :param chunks: 检索阶段召回的文本块列表
    :param top_k: 重排序后返回前 K 个
    :return: 重排序后的文本块列表
    """
    if not chunks:  # 空输入直接返回，避免下游索引报错
        return []

    model, tokenizer = _get_reranker()

    # 构造 [query, chunk] 对，cross-encoder 同时编码两段并输出相关性分数
    input_pairs = [[query, chunk] for chunk in chunks]
    with torch.no_grad():
        inputs = tokenizer(
            input_pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        logits = model(**inputs).logits.view(-1).float()

    # sigmoid 归一化到 0~1，与 FlagReranker 的 normalize=True 语义一致
    scores = torch.sigmoid(logits).tolist()
    # 归一成 list：FlagReranker.compute_score 单条返回 float、多条返回 list，
    # 直接索引会踩坑，这里统一成 list 避免类型分叉
    if isinstance(scores, (int, float)):
        scores = [float(scores)]

    logger.debug(f"重排序得分: {scores}")
    # 按相关性降序取前 top_k
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    reranking_chunks = [chunks[i] for i in sorted_indices]
    for rank, idx in enumerate(sorted_indices):
        c = chunks[idx].replace("\n", "")
        logger.debug(f"[重排] 文本块 {rank}:\n{c}；相关性得分: {scores[idx]}\n")
    return reranking_chunks

"""
三、生成
结合用户查询与检索到的文本块内容组织成大模型提示词（Prompt）。
随后，代码通过调用 Qwen 大模型云端 API，将构建好的 Prompt 发送给大模型，并利用流式输出的方式逐步获取模型生成的响应内容，实时输出并汇总为最终的生成结果。
"""
# 使用 agent_full

if __name__ == "__main__":
    query: str = "四味寒假指什么?帮我解释下rag流程?"
    # 批量索引文件夹下所有文档
    client, collection, all_chunks = indexing_process(
        os.path.join(os.path.dirname(__file__), "..", "resource")
    )
    # 4. 检索
    retrieval_res = retrieval_process(all_chunks, query, client, collection, 3, "bge-small-zh-v1.5")
    logger.info(f"retrieval_process结果： {retrieval_res}")
    reranking_res = reranking(query, retrieval_res, 2)
    logger.info(f"reranking结果： {reranking_res}")

    # 5. 生成：agent 引入 rag（粗暴prompt生成）
    agent_full.run_single(query, reranking_res)
