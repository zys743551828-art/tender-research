# 本地 BGE 与 pgvector 检索对照

冻结 RULE-001 开发参考集：12 份公开资料、531 个物理页，经产品 pdfium-1.3 生成 4,127 个原生文字块。参考状态为“原文核对；无独立工程复核或盲测”。查询器只读取查询、项目及文件过滤条件，检索结束后才读参考引用计分；多份所需依据须全部命中。完整摘录要求正确文件及页上的前 k 条摘录合起来包含全部参考锚点。文字层缺口和块边界也会影响结果，不把指标解释为工程业务准确率。

| 通道 | 查询数 | 页码@1 | 页码@3 | 页码@5 | 完整摘录@1 | 完整摘录@3 | 完整摘录@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 32 | 24 | 31 | 31 | 17 | 21 | 21 |
| BM25＋编号数字 RRF | 32 | 24 | 31 | 31 | 17 | 21 | 21 |
| BGE 语义 | 32 | 23 | 25 | 28 | 15 | 19 | 21 |
| 三路 RRF | 32 | 25 | 30 | 30 | 17 | 21 | 23 |

各路过滤后取 50，RRF offset=60。只有两条带英数查询（RULE-011 的 A4／50、RULE-031 的 108），不足以代表数字编号的真实质量。完整边界、小数精度、全角字符、百分号／千分号由产品单元测试覆盖，仍需独立真实引用查询。

融合改善 RULE-007、RULE-015 的完整摘录，补回 RULE-017 页码，但丢失 RULE-019、RULE-032 的前五条页码命中。因此保留 keyword 默认，pgvector 为可选验证通道，尚不删除 Milvus 部署定义。没有按答案调参或添加任务特判。

复用 D 盘 CPU Torch 环境与 safetensors，只补缺少的 ONNX 导出依赖。运行模块只用 ONNX Runtime 与 tokenizers，无需 Torch。模型摘要与转换对照见 [model-manifest.json](model-manifest.json)；指纹包含前缀、长度、维数、池化及运行文件摘要，运行时不下载。4,127 块编码约 75.01 秒，14 块超过 512 词元并记录截断；这是本机开发耗时，不是生产吞吐量测量。

[runtime-check.json](runtime-check.json) 与 [runtime-check-final.json](runtime-check-final.json) 核对正文／查询、动态批量和截断：产品分词输入与本地 Transformers 逐项一致，ONNX 输出与同输入对照一致、单位范数通过。导出另与 CPU 原模型对照，最大绝对误差约 1.23e-7。

[linux-container.json](linux-container.json) 记录只读 Linux 容器、只读模型挂载和一次性数据库的真实验证：PostgreSQL 16.15／pgvector 0.8.7，编码两条合成证据、回填并查询，工期问题优先返回 180 天依据。它证明运行链路；正式业务库没有切换数据库镜像或默认通道。最终关闭 ONNX 遥测的容器复验另存 linux-container-final.json。

逐查询结果见 [results.json](results.json)。来源：[pgvector 官方说明](https://github.com/pgvector/pgvector)、[BAAI 模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)。权重来自 ModelScope 的 AI-ModelScope/bge-small-zh-v1.5 镜像，不向这些服务上传项目资料。

复现（复用已有环境；输出使用新实验目录）：

```powershell
.venv-parse-docling/Scripts/python.exe scripts/export_embedding_onnx.py --model .runtime/embeddings/bge-small-zh-v1.5
.venv-parse-docling/Scripts/python.exe scripts/check_embedding_runtime.py --model .runtime/embeddings/bge-small-zh-v1.5 --output results/<新实验>/runtime-check.json
.venv-parse-docling/Scripts/python.exe scripts/hybrid_retrieval_probe.py --model .runtime/embeddings/bge-small-zh-v1.5 --output results/<新实验>/results.json
```
