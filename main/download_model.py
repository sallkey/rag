from huggingface_hub import snapshot_download

# 下载嵌入模型到本地（bge-small-zh-v1.5，输出 512 维归一化向量）
snapshot_download(repo_id="BAAI/bge-small-zh-v1.5", local_dir="./bge-small-zh-v1.5")

# 下载重排序模型到本地（bge-reranker-v2-m3，cross-encoder，用于检索结果二次精排）
snapshot_download(repo_id="BAAI/bge-reranker-v2-m3", local_dir="./bge-reranker-v2-m3")
