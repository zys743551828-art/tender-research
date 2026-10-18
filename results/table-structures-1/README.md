# 扫描网格结构与可能续表候选

产品 `6078008`，解析器 `pdfium-1.4`，数据库仍为 0007。接口、产物和交互范围见产品仓库 `contracts/table-structures.md`。默认 keyword 保持，正常 Word 转换服务保留；独立回归测试容器已清理，Claude 已关闭的测试容器未启动。

## 实际执行

- 后端完整质量门与 **305 项测试通过**，mypy 检查 82 个源文件；前端格式／lint／类型／构建及 **322 项测试通过**；编排 23 项通过。
- 纯几何／归属测试覆盖空格、跨格行不猜位置、OCR 行引用越界，以及表头、列位置、页边与相邻条件缺失时不提示续表。正式接口验证行／表翻页、负参数、解析过期、产物损坏、旧解析未执行及核验不改原结构。
- `table_structure_runtime_check.py` 在无网络只读实际镜像执行：清晰三行两列网格归属通过；空白、单列、缺少横线及 9° 旋转合成反例均未产生候选。核对 `backend/constraints.txt` 的 76 个运行包，均与本次更新前正常部署版本一致。此检查不代表一般旋转／合并格质量验收。
- `scanned_table_live_check.py` 生成两页纯栅格 PDF，文字层字符数 **[0, 0]**。通过同源 API／OSS／正式解析 worker 的本地 OCR 生成两张三行两列 `raster_grid` 表候选；金额 15.12 在对应单元格出现，重复表头及列位置触发 pending 的可能续表提示。
- 接受一条金额摘录生成新解析，原 PDF／原证据保持，原始结构数组与跨页候选保持；旧解析请求返回 409。结构及未完成 OCR 核验的阻断缺口继续保留；项目关闭自动评审，只出现 parse 任务，没有业务模型调用。
- 两个既有合成项目的历史报告／快照前后完全一致；旧规则确认与旧输入重放仍拒绝，无新增业务任务。机器结果见 `live-before.json`、`live-after.json`。

合成扫描原件 [live/synthetic-scanned-tables.pdf](live/synthetic-scanned-tables.pdf)，SHA-256 `25b5f0f7ee9c055d9be338beb3ade2d6756d69cee87e2c7f5b1cc2d4c066fb8a`。原始结构预览见 `live/first-table.json`、`live/second-table.json`；核验与身份结果见 `live/receipt.json`。仅有两页合成表格，不能推断真实标书、无线表、合并格、斜表、复杂跨页表或图形语义质量。

更新前本地数据库备份 `codex-diagnostics/pre-table-structures-20261018.dump`，253,748 字节，SHA-256 `1df608728b4398333ee13f7456a354ce0a4785654ba01226d32ac9893b4f073e`，不提交 dump。无新增迁移，未因本次功能重做完整灾难恢复演练。

实际后端镜像 `sha256:fa6210beb95f76ca0c12c08ccb3d7ab6bc4c2859033cfa3f2af24c0824b727af`，web `sha256:0e842dc211fea5cb4a77eb41ed43543cac67590afdef1b3590c2bd35d128693e`。11 个业务服务健康，见 `deployment-receipt.json`。

浏览器连接仍因插件 `26.1013.40934/scripts/browser-service.mjs` 缺失而失败，未产出截图或视觉验收；组件交互测试与真实 API／OCR 验证不替代浏览器整轮验收。自动接续保持关闭，main 未改动。
