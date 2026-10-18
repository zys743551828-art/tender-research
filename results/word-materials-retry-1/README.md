# 已部署转换重试与实际 PDF 采用保留

合成项目 `prj_7422077c3d2d46c6`，结果为 `live.json`，项目自动评审关闭，模型调用 0。

`scripts/word_materials_live_check.py` 通过实际登录、同源分片上传、已部署解析 worker、无网络转换器和原件下载，验证普通中文受控转换、动态域拒绝、单稿 PDF 补交、新稿与历史稿分离、复杂表格等待核对及模拟确认采用。额外发起真实 Word 转换重试，产生第二份转换候选，既有采用 PDF 保持不变。

API 只配置受控方法指纹，未挂载队列；解析 worker 挂载输入可写／输出只读卷。初次重试验证发现 API 的队列路径配置判断不适用，改为方法配置判断后重跑通过。

在工作区执行：`tender-app/backend/.venv/Scripts/python tender-research/scripts/word_materials_live_check.py --output <新的JSON路径>`。使用已有后端 httpx 环境，不要求给研究环境重复安装依赖；结果不得覆盖。脚本仅创建合成项目，自动评审关闭，采用确认属于接口模拟，不冒充人工真实标书验收。

未验证：浏览器真实交互与视觉（插件缺少运行文件）、真实外部标书排版准确率和多解析 worker 容量。原稿／实际 PDF 来源冻结与截止、并发、过期校验见产品 `tests/test_word_materials.py`；合成接口跑通不代表这些质量边界已验收。
