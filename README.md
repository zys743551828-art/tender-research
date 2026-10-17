# tender-research

施组投标诊断的研究与验证程序：规则证据检索、有限规则与疑点诊断、成绩导入、技术分评测与校准、版本闭环，以及大模型（DeepSeek）评测。实际结果、失败和选型判断统一见 [tender-docs 的验证报告](https://github.com/zys743551828-art/tender-docs/blob/main/04-验证与测试/验证报告.md)；本仓库维护运行方法、代码边界和实验输出。

## 1. 环境

Python 3.11 及以上。数据集默认取与本仓库同级的 `tender-datasets`，可用环境变量 `TENDER_DATASETS` 指定其他位置。

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

固定依赖见 `pyproject.toml`。OCR 与 AI 编排检查的依赖较重，各用独立环境（`requirements-ocr.txt`、`requirements-orchestration.txt`）。

原件不在 git 中：数据集仓库只保存资料清单、校验值与存储键，原件在对象存储（阿里云 OSS 等 S3 兼容服务）和本地缓存（`TENDER_DATASETS_CACHE`，默认 `~/.cache/tender-datasets`）。读取原件的程序先找数据集工作目录、再找本地缓存，都没有时提示取回命令。新环境先安装存储依赖并取回原件：

```bash
.venv/bin/pip install -e ".[store]"
python -m tender_research.store fetch --dest ../tender-datasets   # 按资料清单取回到数据集工作目录（已被 .gitignore 排除）
```

存储配置（`TENDER_STORE_*`）见 tender-datasets 的 `入库模板/README.md` 第 5 节。
签名默认按端点选择：阿里云 OSS 使用 `s3`，其余使用 `s3v4`；自定义域名可设置 `TENDER_STORE_SIGNATURE=s3`。

## 2. 入口

所有模块以 `python -m tender_research.<模块>` 运行，输出写入 `results/`。

| 命令 | 作用 |
| --- | --- |
| `run_all` | 重新生成全部本地结果并运行测试：检索对照、规则与疑点检查、成绩导入、契约、测试记录；不调用外部模型 |
| `retrieval_eval` | 9 种解析×窗口候选在固定规则参考集上的检索对照 |
| `rule_checks` | 真实资料上的有限句式、单案例疑点与评分就绪核验 |
| `rule_evidence` | 对原件检索候选证据，不生成规则答案或成绩（`--query`、`--top-k`、`--extractor`） |
| `intake` | 样本登记表＋收件目录批量入库：校验、定位原件、SHA-256／页数／文字层检查；原件按 DATA-002 §2.4 归档到本地缓存，数据集仓库写资料清单 1.3、`评分配对集.json` 与 `入库报告.md`（`--sheet`、`--inbox`、`--datasets`、`--cache`、`--items`、`--upload`、`--dry-run`）；不覆盖已有案例，重复运行跳过已入库样本 |
| `store` | 原件对象存储（内容寻址 `originals/sha256/…`）：`push` 上传、`fetch` 取回并核对、`verify` 核对桶中对象、`status` 计数；需 `.[store]` 与 `TENDER_STORE_*` 配置 |
| `score_eval` | 评分配对集（格式 A）与系统批量预测（格式 B）在冻结协议下的技术分评测、校准与模型导出（见第 5 节） |
| `ocr_probe` | 三页本地 OCR 探针（需 OCR 环境） |
| `deepseek_client` | 检查 API 连接与可用模型 |
| `llm_rules`、`llm_baseline`、`llm_fulltext`、`llm_visual` | 大模型评测，**会发起付费请求**；已有实验的协议、输入与响应带校验值并已冻结，改动代码或配置后须使用新的实验名称输出，不覆盖既有结果 |

```bash
python -m tender_research.rule_evidence \
  --manifest ../tender-datasets/公开规则资料/资料清单.json \
  --project 2026BFFGZ51067 --query '技术入围后的复核不通过是否增补？' --top-k 5
python -m tender_research.llm_rules --model deepseek-v4-pro
python -m tender_research.llm_fulltext --visual-facts
```

`rule_evidence` 的 `--extractor` 支持 pypdf、pdfplumber-crop、pdfplumber-render-guard（默认第三种）。证据含文件身份与 SHA-256、物理页、抽取字符范围和原文；正文印刷页码可能与物理页不同。

AI 编排检查需要 PostgreSQL 16：

```bash
python -m venv .venv-orchestration && .venv-orchestration/bin/pip install -r requirements-orchestration.txt
PG_DSN="host=localhost port=5432 user=postgres" .venv-orchestration/bin/python scripts/orchestration_check.py
```

### 大模型凭据

读取环境变量 `DEEPSEEK_API_KEY`，POSIX 系统也可使用 `.secrets/deepseek_api_key`（目录权限 700、文件 600，已被版本管理忽略）。Windows 的 ACL 不能用 POSIX 权限位核实，因此只接受环境变量，不读取凭据文件；`run_all` 遇到该文件仍判为权限检查未通过。凭据不进入提示、结果或日志；`run_all` 会检查仓库内是否出现该密钥。企业全文只在公司明确授权的范围内发送，授权范围记录在各实验的 `protocol.json`。

## 3. 代码

| 模块 | 职责 | 边界 |
| --- | --- | --- |
| `rule_evidence`、`retrieval_eval` | 原件核验、按页抽取、渲染空白保护、BM25、固定参考集回归 | 没有通用语义规则或向量重排；页面保护只处理已发现的两类问题 |
| `contracts` | Evidence、OfficialLabel、ReportedResult、PairSample（格式 A）、SystemPrediction（格式 B）、EvalProtocol、Finding、Prediction 的严格边界 | 用户报送不能自动升为官方训练标签；不替代来源核实和生产授权 |
| `diagnostics`、`rule_checks` | 显式句式、补疑原值匹配、限定管材疑点、修订前置门槛 | 疑点须工程师确认；没有自动编制可提交稿 |
| `intake` | 登记表校验、原件定位与归档、正文 MinHash 模板分组、同场成绩与分项得分、交接格式 A | 只读 CSV；分项只随新样本写入，已入库资料清单不改写；无文字层正文不做 OCR |
| `store`、`paths` | 内容寻址存储键、上传前查询与覆盖保护（OSS 用 x-oss-forbid-overwrite，其他用 If-None-Match）、上传后与下载后核对；原件在工作目录与缓存之间的定位 | 只用 head/put/get 三个接口；桶名与密钥是部署配置，不写入清单 |
| `scoring`、`score_model`、`score_text`、`score_eval` | 标签准入、多来源合并、连通分组与时间隔离、pf-1 特征、基线／校准／岭回归／锚定／正文近邻方法、误差、区间、排名、入围概率、学习曲线、scoring-model/1 导出；档案评分就绪核验 | 只用 NumPy 与标准库；结论限定于样本覆盖范围，样本不足返回不可评价 |
| `model_service`、`deepseek_client` | 厂商无关接口、适配器注册与路由；DeepSeek 适配 | 当前只注册 DeepSeek |
| `llm_common`、`llm_rules`、`llm_baseline`、`llm_fulltext`、`llm_visual` | 冻结输入与响应、引用与范围门槛、公开规则、全文基线、页 ID 绑定与思考模式、图页探针 | 引用匹配不证明语义成立；单例结果不是独立精度 |
| `lifecycle` | 内容寻址、不可变记录、快照、整包路由、幂等租约、结果更正、配对恢复 | SQLite 本地夹具；只允许在 fixture 环境发布 |
| `outcomes` | 明确表头的 13 列公示成绩、缺分与候选阶段 | 只支持已验证的版式；没有正文配对就不进入监督数据 |
| `ocr_probe` | 本地 OCR 候选对照 | 三页探针；OCR 数字与表格结构仍需复核 |
| `tests/` | 故障、泄漏、恢复、供应商与口径边界、真实原件完整性 | 测试数据为合成夹具，只证明程序行为 |

## 4. 使用条件

官方总分、人工诊断、候选预测分别存储；综合分与未知分不转换为技术标签，未提交修订稿不继承原稿成绩。评分特征限定为 pf-1 并由协议白名单收窄，事后信息一律拒绝。

均值、中位数、评分办法分组均值是任何系统方法都必须显著超过的基线；预处理、alpha、保序校准与保形残差都只在训练折内拟合，误差同时保留原量尺与满分百分制。

`lifecycle.Store` 维护文件、快照、实验、模型组合、任务及结果；不可变记录由内容计算 ID。已有任务锁定整包，失去租约的 worker 不得回写；更正和回退只追加记录。

大模型实验的输入、协议和响应在调用前后冻结并记录校验值，模型别名不固定服务权重，同一输入的重复调用也可能不同。单例不同方案有明显分差和拒绝估分，不是置信区间、独立准确率或生产评分模型。

## 5. 技术分评测与校准

技术分预测分三步：① `intake` 入库，得到评分配对集（交接格式 A）；② tender-app 的 `backend/tools/batch/predict_dataset.py` 用产品流水线逐份跑预测，得到系统批量预测（交接格式 B，先 `--estimate-only` 估算费用，再 `--limit 10` 试点、再全量）；③ 本节的 `score_eval` 在冻结协议下比较方法、给出结论并导出可部署的评分模型，经 `bidwb-admin register-scoring-model` 登记后产品即按它校准。产品里登记了技术分的提交可用 `bidwb-admin export-score-pairs` 导出同样的格式 A／B，与历史数据合并后重训，数据越多模型越新。目标量尺为满分百分制 y = 100·官方技术分/技术满分。

```bash
python -m tender_research.score_eval \
  --pairs ../tender-datasets/评分配对集.json [更多配对集…] \
  --predictions <格式B目录> [更多目录…] \
  --protocol 评测协议.json --name <实验名> [--mode cv|holdout] \
  [--datasets-root <目录>] [--repeat-predictions <目录>…] [--results results] [--force]
```

输出 `results/score-eval-<实验名>/`：`report.json`（全部数字、每样本折外预测、协议／数据／代码 SHA）、`report.md`（中文，可直接贴进验证报告）、`calibration_bundle.json`（scoring-model/1）。同名输出不覆盖，须换名或加 `--force`。`--mode holdout` 是最终测试：同一协议 SHA 只运行一次，记录在 `results/score-eval-holdout-registry.json`，`--force` 重跑会登记为强制重跑。正文文字按 SHA-256 缓存在 `results/score-eval-text-cache/`。

多个来源（批量回放与产品导出的同格式文件）合并评测：相同正文 SHA 只保留一份（优先标签可用的一份，另一份记为 `duplicate_source`），同一样本编号指向不同正文时拒绝运行。

**准入**：只用数值已公布、不超满分的 `technical_total` 标签，且来源类型在协议 `label_sources` 内（此时待核验标签也可进入）；综合总分、未公布分、配对集已排除的样本不进入监督集。报告列出标签来源构成，并在全部准入标签之外单独复算“已核验官方原件”子集（样本不足时不可评价）；结论以全部准入标签为准，导出模型记录 `label_sources` 与来源构成。系统预测缺失或输入指纹（招标与正文 SHA）与配对集不符的样本排除；运行失败与拒绝估分保留，作为系统表现计入（特征缺失按训练折中位数插补）。

**分组与划分**：项目、模板组、相同正文 SHA 用并查集连成连通组，组不跨训练／测试；每个划分都再次复核。`cv` 为重复分组 K 折（每次重复换种子重新成折）；`holdout` 取提交日期最晚的一部分连通组作测试，训练侧剔除跨越测试起点的组与测试起点时尚未取得的标签（`known_at`），方法选择只在训练侧的重复分组交叉验证内完成。

**方法**（同一组划分上比较）：

| 方法 | 内容 | 导出 |
| --- | --- | --- |
| `mean`、`median`、`scheme_mean` | 训练均值、中位数、按评分办法分组均值（基线） | 否 |
| `raw` | 系统原始估计百分制 raw_pct | 是 |
| `linear` | a + b·raw_pct 最小二乘 | 是 |
| `isotonic` | raw_pct 的保序回归（PAV，分段线性插值） | 是 |
| `anchor` | 锚定对比估计 anchor_pct 直接作预测 | 是 |
| `ridge` | pf-1 特征（不含锚定特征）标准化后闭式 L2，alpha 由训练折内嵌套分组交叉验证选定 | 是 |
| `ridge_anchor` | 同上，含 anchor_pct／anchor_agree／anchor_n | 是 |
| `ridge_isotonic` | ridge 输出再经保序校准（在训练折内折外输出上拟合） | 是 |
| `knn` | 正文字符 2–3 元哈希 TF-IDF 余弦，训练折内（排除同组）前 k 个近邻的相似度加权目标 | 否 |
| `ridge_knn` | ridge 加近邻目标特征 | 否 |

选定方法取重复分组交叉验证 MAE 最低的可导出方法（或协议指定）；正文近邻类依赖历史正文库，只作对照。

**特征（pf-1）**：以格式 B 中产品后端写入的 `artifact_extra.features` 为准；早期文件无此字段时，从顶层字段派生同名特征（raw/low/high/width_pct、分项覆盖率与档位、max_ratio、正文页数与字数对数、资格违反与无法判断数、一致性与范围问题数）。不属于 pf-1 的键不进入模型并在报告中列出；协议 `features.include/exclude` 只能在 pf-1 内收窄，`*official*`、`*label*`、`*cutoff*` 等事后信息一律拒绝。插补取训练折中位数，只有训练折内缺失与否有变化时才加 `<特征>__missing` 指示列。

**指标与结论**：MAE、RMSE、偏差、绝对误差 P50/P90；选定方法相对最佳基线的 MAE 增益与按连通组重采样的 bootstrap 区间；分层样本少于下限的层标“不可评价”；分组 split-conformal 区间（半宽取训练折内折外绝对残差的有限样本分位数）的实际覆盖率与宽度，以及系统原生 low–high 区间覆盖率；同场排名（项目内 ≥2 份有标签样本：Spearman 均值、两两排序正确率、最高分命中率）；配对集提供同场成绩时另报项目相对目标（减同场中位数）、同场百分位误差和入围概率（保形预测分布下 P(得分 ≥ 入围线)，Brier、对数损失、AUC 与基率对照、校准分箱）；配对集提供官方分项时报分项匹配率与分项误差；学习曲线（训练连通组取 25%/50%/75%/100%，测试折不变）。结论只有三种：通过（MAE ≤ 目标且增益区间下限 > 0，范围限定为样本覆盖的地区／工程类型／评分办法）、未通过、不可评价（有标签样本或独立项目低于下限时照常输出报告，不报精度）。

**导出模型 scoring-model/1**：评测结束后用全部准入样本重拟合选定方法；预测 = clip(isotonic(intercept + Σ coef·z))，z = (x − mean)/scale（scale 为 0 取 1，null 取 impute，`__missing` 列原特征为 null 取 1），区间半宽为该方法在全部样本上分组折外残差的 80%／90% 保形分位数。`score_model.apply_bundle` 是参考实现；`protocols/scoring-model-1.golden.json` 给出产品后端须逐位复现的参考向量。未通过或不可评价时照常导出，`evaluation.verdict` 标明，产品只应加载通过的模型。

### 评测协议字段

示例见 `protocols/评测协议.example.json`。复制后填写，在查看任何测试标签前冻结；报告记录协议文件 SHA-256。

| 字段 | 含义 |
| --- | --- |
| `schema` | 固定 `score-eval-protocol/1` |
| `frozen_at` | 冻结日期 |
| `description` | 本次评测目的与数据范围说明 |
| `target_mae` | 目标误差：选定方法 MAE（满分百分制）不得高于此值 |
| `selection` | `cv_mae`（重复分组交叉验证 MAE 最低的可导出方法）或指定某个可导出方法 |
| `methods` | 参与比较的方法，须同时含至少一个基线与一个可导出方法 |
| `ridge_alphas` | 岭回归 alpha 候选，训练折内嵌套分组交叉验证选定 |
| `seed` | 成折、alpha 选择、bootstrap、学习曲线的固定种子 |
| `cv.folds`、`cv.repeats` | 分组 K 折的折数与重复次数（也用于训练折内的嵌套划分与保形残差） |
| `holdout.test_fraction` | 时间留出中作测试的最晚连通组比例 |
| `holdout.min_test_samples` | 留出测试样本下限，不足则不可评价 |
| `bootstrap_resamples`、`confidence` | 组级 bootstrap 次数与区间置信水平 |
| `interval_levels` | 报告的保形区间名义覆盖水平（导出模型固定给 0.8 与 0.9） |
| `features.include`、`features.exclude` | 多特征模型使用的 pf-1 特征（通配符），只能在 pf-1 内收窄 |
| `knn.k`、`knn.ngrams`、`knn.hash_bits` | 正文近邻的近邻数、字符 n-gram 长度、哈希维数（2^bits） |
| `slices` | 分层维度：`scoring_scheme`、`engineering_type`、`own_company`、`region`、`system_status` |
| `item_match_threshold` | 系统分项与官方分项匹配的字符二元组 Jaccard 下限 |
| `minimums.labeled_samples`、`minimums.independent_projects` | 低于任一项时整体不可评价（默认 30 份、10 个项目） |
| `minimums.slice_samples` | 分层样本下限（默认 10） |
| `minimums.ranking_projects` | 同场排名可比项目下限（默认 5） |
| `minimums.shortlist_samples` | 入围概率评测样本下限（默认 20） |
| `minimums.item_matches` | 分项误差评测的匹配分项下限（默认 10） |
| `learning_curve_fractions` | 学习曲线的训练连通组比例 |
| `label_sources` | 准入的标签来源类型：`official_document`、`company_ledger`、`user_statement`（示例为前两种） |
| `allow_repeat_runs` | 是否允许 `--repeat-predictions` 读取同一子集的其他运行目录，报告系统原始估计的跨运行波动 |
