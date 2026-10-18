# 隔离 DOCX 转换专项实测

八种合成输入由 `scripts/word_converter_check.py` 生成，使用真实 LibreOffice 7.4.7、PDFium 5.13.0 与 Landlock ABI 7；不调用模型、不访问业务资料和数据库。结果见 `results.json`，逐任务隔离与启动拒绝见 `isolation.json`。

- 普通中文转换为 PDF，中文、180 日历天、15.12‰ 的文字与页面渲染核对通过；页面图为 `plain-chinese.png`。
- 表格生成 PDF 并保留复杂版式待核实；不存在的字体触发字体替换待核实，不把生成文件判为完全对应。
- PAGE 域、外部链接、无效 DOCX、原件摘要不符及转换方法指纹不符均失败，并且没有执行转换或生成 PDF。
- 子进程能读自己的任务文件；读取其他任务、共享队列、父进程应用文件以及向父进程发信号均被拒绝。未加容器权限限制的启动被主动拒绝。

方法指纹为 `7c6487b8fca6b1a5752bfb0c2950e06bf9d6dc8e167db6883d2b7610c1af3e21`。其标识程序、系统包、字体与转换选项；PDF 本身含生成时间等元数据，跨次 PDF 摘要不同不等于内容变化，每次都须按各自回执验证。

复现：在工作区执行 `tender-research/.venv/Scripts/python tender-research/scripts/word_converter_check.py --output <新的目录>`，镜像先按产品仓库 `deploy/word/README.md` 构建。脚本自动创建并清理自己命名的容器与卷；不覆盖已有结果。

边界：仅验证转换基础设施，尚未接入业务上传、PDF 采用与逐文件补交；未验证真实标书、复杂排版准确率、多并发队列容量及生产主机可用性。文字对应检查不能证明完整视觉一致，也不能证明 Word 原件合格或用户补交 PDF 与原件一致。
