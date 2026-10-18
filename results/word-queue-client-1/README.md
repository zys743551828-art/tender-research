# 产品 Word 队列客户端联调

使用 `scripts/word_converter_check.py --client-module ../tender-app/backend/src/bidwb/word_conversion.py --output <新的目录>` 重跑八种基础设施输入，并额外让产品客户端在真实 Linux 容器、业务 UID 10001、无网络／只读根／输出只读卷下接收普通中文成功回执与动态域失败回执。共十种输入通过，结果为 `results.json`；隔离及启动拒绝为 `isolation.json`。

客户端按预期核对原件、操作及方法身份，成功 PDF 大小及摘要一致，失败不交付 PDF；结束后自己的请求已清理。方法指纹与 `../word-converter-1/` 相同。前一目录保存已核对的普通中文页面图与转换边界，本次 PDF 不作全标书版式质量结论。

产品单元测试另外覆盖错配回执、隔离能力不足、虚假页数、状态与缺口不一致、摘要不符、心跳过期、租约取消和超时清理。这里只验证客户端与隔离基础设施联调，尚未接业务上传、采用决定、数据库回执和快照。
