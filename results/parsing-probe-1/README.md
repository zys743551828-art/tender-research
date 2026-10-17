# 本地解析候选页级对照

这是四份真实资料的十个物理页探针。四份原件共 740 页，已核对原件 SHA-256 并完成文字层／图片对象盘点；候选模型只对选定十页推理。它是选型的探索性记录，不是全文、表格准确率或独立测试集验收。

样本身份、原件路径、哈希、物理页和文字锚点见 [参考协议](../../experiments/parsing-probe-1.json)。文字、HTML、原页渲染和失败堆栈保存在本机 `D:\workspace\tender\codex-diagnostics\parsing-comparison-20261017`，原件只在本机处理，模型权重和缓存位于研究仓库的 `.runtime/parsing/`。权重不进 Git，[模型文件清单](model-files.json)记录本地文件哈希，四份环境文件记录已安装版本。

## 方法与边界

- 基线使用 PDFium 原生文字层；MinerU 4.0.12 使用本地 ONNX basic 档、auto OCR，关闭图像分析；Docling 2.140.0 使用 RapidOCR 全页模式和表格结构模型；PaddleOCR 3.7.0／PaddlePaddle 3.3.1 使用 PP-StructureV3；RapidOCR 3.10.0 单独作为文字回收通道。
- Paddle 与独立 RapidOCR 输入为 PDFium 2 倍渲染的原页图，等效 144 DPI；MinerU 和 Docling 输入为原始 PDF 的指定物理页。两种输入方式有差异，结果不视为同配置算法排名。
- 保存逐页结果、错误、耗时、原始引擎产物和软件版本。锚点仅作 Unicode NFKC 与空白归一化；不纠正错字、数字或单位。HTML 内嵌图片从文字输出中移除，原始 JSON 仍保留图片引用。锚点命中不能替代行列归属、数字单位、图形语义的检查。
- [比较 JSON](comparison.json)分别保留推理协议 SHA 和本次参考协议 SHA。原页的“普工”最初录为“普通工”，经原图复核改正；Paddle、Docling 推理沿用其当时加载的协议，复核明确记 `reference_changed=true`。没有把修正后的数字包装成看结果前冻结的准确率。
- 缺少页面仍计入预期页数；引擎失败的锚点保持未评价，不作 0/0 的通过判断。两页只观察排版和普通文字，不设文字锚点。人工核对的局部事实见 [检查记录](manual-review.json)，其余单元格和图意保持未评价。
- 多个本地进程曾并行占用 CPU。耗时包含本轮初始化和逐页处理，缓存已预热；MinerU 逐页启动 CLI，其他引擎复用实例。下载、安装、人工核对和失败尝试不计入成功轮耗时。不能据此声称生产吞吐或性能排名。无按页云推理费用；未估算电力、硬件折旧或维护成本。

## 已发现的问题与选型结论

| 引擎 | 有解析产物的页 | 文字锚点命中 | 本轮记录耗时 |
| --- | --- | --- | --- |
| 原生 PDFium | 10/10 | 20/25 | 0.24 秒 |
| MinerU basic | 10/10 | 22/25 | 70.95 秒 |
| 独立 RapidOCR | 10/10 | 25/25 | 15.51 秒 |
| PP-StructureV3，关闭 oneDNN | 10/10 | 25/25 | 442.94 秒 |
| Docling | 0/10 | 未评价，预期 25 个 | 109.46 秒后各页均失败 |

有产物只表示引擎返回了结果。锚点总数、缺页与失败细节均由比较脚本生成；以下人工检查包含锚点以外的错误，因此不能把 25/25 称为识别完全正确。耗时的输入、缓存和 CPU 负载差异见前述边界。

文字层无法覆盖扫描表，含原生标题的图页也可能丢失几乎全部图内内容。仅按“有文字／无文字”划定覆盖不足：扫描收费页仍有四个页脚字符，总平面页也有大量文字但不包含地图空间关系。

MinerU 能恢复劳动力表和用地表的部分行列关系，但扫描收费页漏掉表下三个金额示例。独立 RapidOCR 找到了这些金额，仍有严重错字；Paddle 也找到了金额，但收费表相邻表头被合并、评分表分值格混入正文。三种候选都把收费单位“‰”读成“%”。网络图连线与甘特图工序条均不能凭文字和空表格判作已读懂。

因此保留“原生文字优先、OCR 补漏、结构解析独立留存”的候选接入方向；尚未批准任何候选直接替代系统解析器。OCR 和结构结果都需要原件版本／物理页／原图坐标绑定，表格及图页保留未识别范围，重要金额、单位、评分分值和跨页关系须独立核对。图形区域被提取或文字锚点命中不会把覆盖改成完整。

恢复过程：Docling 因官方 Hugging Face 模型源 TLS EOF 无法加载版面模型，模型源与下载镜像的连通检查均失败；本轮没有质量结论。Paddle 默认 oneDNN 推理出现 `ConvertPirAttribute2RuntimeAttribute` 错误，关闭 `enable_mkldnn` 后可以推理；失败轮和恢复轮分开保存。未关闭 TLS 验证，未用远程文档推理替代本地候选。

下一步补齐跨页表、旋转页与更高分辨率对照，建立行列及数字单位的参考标注，再接入系统的页级覆盖、超时、失败恢复和按需 OCR。图形语义需要另行验收。本次没有更改系统部署或评分结论。

## 复跑

先检查现有环境；本机已安装的三个 `.venv-parse-*` 可以直接复用，不需要重复安装。以下命令在 PowerShell 7、研究仓库下执行，选用对应引擎环境。独立 RapidOCR 复用 Docling 环境。模型下载只访问官方来源，原件不传给远程服务。

```powershell
$env:PYTHONUTF8 = '1'
.\.venv-parse-mineru\Scripts\python.exe scripts\parsing_probe.py --engine mineru --protocol experiments\parsing-probe-1.json --datasets ..\tender-datasets --output .runtime\parsing-rerun --cache .runtime\parsing
.\.venv\Scripts\python.exe scripts\parsing_probe_report.py --protocol experiments\parsing-probe-1.json --run .runtime\parsing-rerun\mineru --output .runtime\parsing-rerun\comparison.json
```

探针开始前核对原件哈希。MinerU 运行时要求本地模型完整；缺少时先按[官方模型下载说明](https://opendatalab.github.io/MinerU/usage/model_source/)预取 basic／ONNX 模型到 `.runtime/parsing/mineru`。其他候选通过各自环境调用同一脚本，`--engine` 可选 `pdfium`、`mineru`、`docling`、`paddle`、`rapidocr`，`--sample` 可按单页定位故障。使用新的输出目录，保留已有故障记录。

配置依据：[MinerU 本地解析](https://opendatalab.github.io/MinerU/quick_start/)、[Docling 全页 OCR](https://docling-project.github.io/docling/_generated/examples/full_page_ocr/)、[PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PP-StructureV3.en.md)。运行结果以本仓库记录为准。
