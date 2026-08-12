# main — RAG 检索增强生成流水线

单文件版 Agent（七篇合一）与 RAG 流水线的教学实现。

`main.py` 是入口：多格式文档加载 → 分块 → 嵌入 → Milvus 索引 → 检索 → 喂给 `agent_full.run_single()` 由 Agent 结合检索文档作答。

## 流水线

```
resource/ 下文档（pdf/docx/pptx/xlsx/csv/md/xml/html/txt）
   │  langchain_community 加载器解析为文本
   ▼
RecursiveCharacterTextSplitter  (chunk_size=512, overlap=128)
   │
   ├──→ bge-small-zh-v1.5 嵌入 (normalize_embeddings=True, 模型单例 + 批量 encode)
   │        │
   │        ▼
   │     Milvus Lite (IP/FLAT, 本地 main_milvus.db)  ← 向量+原文入库
   │        │
   │        ▼  向量检索 top_k×3（语义召回）
   └──→ jieba 分词 → BM25Okapi 语料索引
            │
            ▼  BM25 检索 top_k×3（关键词召回）
            │
            ▼
   RRF 倒数排名融合 + 去重 → top_k 个候选 chunk
            │
            ▼  bge-reranker-v2-m3 cross-encoder 二次精排
            │  （模型单例，本地优先，sigmoid 归一化 0~1）
            ▼
   重排后 top_k 个 chunk
            │
            ▼
agent_full.run_single(task, rag_chunks)
   │  检索结果作为"参考文档"拼进 system prompt
   ▼
Agent 工具循环 → 输出
```

## 支持的文档格式

由 `load_document` 按扩展名分发到对应 langchain 加载器：

| 扩展名 | 加载器 |
|---|---|
| `.pdf` | `PDFPlumberLoader` |
| `.txt` | `TextLoader`（utf-8） |
| `.doc`/`.docx` | `UnstructuredWordDocumentLoader` |
| `.ppt`/`.pptx` | `UnstructuredPowerPointLoader` |
| `.xlsx` | `UnstructuredExcelLoader` |
| `.csv` | `CSVLoader` |
| `.md` | `UnstructuredMarkdownLoader` |
| `.xml` | `UnstructuredXMLLoader` |
| `.html` | `UnstructuredHTMLLoader` |

## 文件说明

| 文件 | 职责 |
|---|---|
| `main.py` | RAG 流水线入口，`load_document`（多格式）→ `split_text` → `embed_and_index` → `retrieval_process`（向量+BM25 RRF 融合）→ `reranking`（cross-encoder 精排）→ `agent_full.run_single`。`indexing_process` 批量索引文件夹下所有文档；嵌入/重排模型均懒加载单例；`embed_and_index` 用 pymilvus 2.6 的 `create_schema` + `IndexParams`（IP/FLAT）显式建表 |
| `agent_full.py` | 741 行单文件 Agent（七篇合一）。RAG 文档通过 `build_system_prompt(rag_chunks)` 注入 system prompt |
| `logger.py` | logging 配置（console + `demo.log` 文件） |
| `download_model.py` | 从 HuggingFace 下载 `bge-small-zh-v1.5`（嵌入）和 `bge-reranker-v2-m3`（重排）到本地 |
| `bge-small-zh-v1.5/` | 嵌入模型权重目录（本地，不入 git） |
| `bge-reranker-v2-m3/` | 重排序 cross-encoder 权重目录（本地，不入 git，首次用 `download_model.py` 下载） |
| `main_milvus.db` | Milvus Lite 本地向量库文件（运行产物，不入 git） |
| `agent_memory.md` | Agent 记忆持久化文件 |
| `resource/` | 默认被索引的文档目录（支持多格式） |

## 必需的两个环境变量

`main.py`（经 `agent_full.py`）在模块加载时直接读这两个变量初始化 OpenAI client：

```python
# agent_full.py 顶部
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
    http_client=httpx.Client(verify=False),
)
```

- **`OPENAI_API_KEY`** — LLM 服务 API Key。
- **`OPENAI_BASE_URL`** — OpenAI 兼容 API 的 base_url。当前默认模型 `MODEL = "GLM-5.2"`，指向的是 GLM 系列兼容端点；可按需改 `agent_full.py` 顶部的 `MODEL` 常量适配实际模型。

注意两点：
1. 这两个变量是在 **import `agent_full` 时** 就求值的（模块顶层），所以必须在运行 `main.py` **之前**设置好，进程启动后无法再改。
2. `main.py` 第 10 行 `import agent_full`，意味着 import 即触发 client 初始化——变量缺失会直接抛连接错误。

设置方式（任选其一）：

```bash
# 临时（当前 shell）
export OPENAI_API_KEY=sk-your-key
export OPENAI_BASE_URL=https://your-endpoint/v1

# 或 Windows
set OPENAI_API_KEY=sk-your-key
set OPENAI_BASE_URL=https://your-endpoint/v1

# 或写入 .env 后用 dotenv 加载（main.py 当前未引入 python-dotenv，需自行加）
```

## 运行

`main/` 下所有 Python 执行使用项目根目录的 `.venv` 虚拟环境（非 conda）：

```bash
cd main

# 首次需安装依赖（见下一节）
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt         # Unix

# 首次需下载 bge 嵌入模型（如 bge-small-zh-v1.5/ 不存在）
.venv/Scripts/python.exe download_model.py                    # Windows
# .venv/bin/python download_model.py                          # Unix

# 设置 LLM 环境变量（见上一节）后
.venv/Scripts/python.exe main.py                              # Windows
# .venv/bin/python main.py                                    # Unix
```

> `main.py` 的 `__main__` 里 query 和 PDF 路径是硬编码的（`query = "四味寒假指什么"`、`resource/1.pdf`、`top_k=2`），改这些参数直接编辑文件。

## 依赖

所有依赖与固定版本见 `requirements.txt`，版本取自 `.venv` 实际安装版本，直接安装：

```bash
cd main
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt         # Unix
```

核心依赖（用途详见 `requirements.txt` 注释）：

| 依赖 | 用途 |
|---|---|
| `langchain-community` | 多格式文档加载器（`PDFPlumberLoader` / `TextLoader` / `Unstructured*` / `CSVLoader` 等） |
| `langchain-text-splitters` | 文本分块（`RecursiveCharacterTextSplitter`） |
| `pdfplumber` | `PDFPlumberLoader` 底层 PDF 解析引擎 |
| `unstructured` | `Unstructured*` 系列加载器底层（Word/PPT/Excel/Markdown/XML/HTML） |
| `sentence-transformers` | 加载并 encode `bge-small-zh-v1.5` 嵌入模型 |
| `transformers` | 加载 `bge-reranker-v2-m3` cross-encoder 重排序模型（`AutoModel` / `AutoTokenizer`） |
| `pymilvus` | Milvus 向量库 client（`MilvusClient`，IP/FLAT 索引，余弦相似度） |
| `milvus-lite` | `pymilvus` 可选依赖：Milvus Lite 本地文件存储后端 |
| `jieba` | 中文分词（BM25 检索对文档与查询分词） |
| `rank-bm25` | BM25 关键词检索（与向量检索 RRF 融合召回） |
| `huggingface_hub` | `download_model.py` 拉取嵌入模型权重 |
| `openai` | OpenAI 兼容 LLM client（读 `OPENAI_API_KEY` / `OPENAI_BASE_URL`） |
| `httpx` | 自定义 transport（`verify=False`，跳过证书校验） |
| `torch` | `sentence-transformers` 底层依赖，显式固定避免版本错配 |
