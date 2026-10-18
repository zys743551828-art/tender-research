# 无关图注的五对重复核查

只使用合成施工进度条款；复用 `verified-context-1/contrasts/protocol.json` 的基准和无关图注正文，正文共同部分的证据编号统一为 `shared_body_0/1`。此前两组使用不同前缀，因此本实验并非旧请求的逐字重放。协议、完整提示词、固定五对交替顺序及失败处理在任何调用前写入 `protocol.json`，不按结果补抽或提前停止。温度 0，产品方法 complete-chapters-2、提示词 p4，请求模型别名 deepseek-chat；未验证服务端实际模型版本。

| 指标 | 基准 | 附无关图注 |
| --- | --- | --- |
| 五次分数（满分 8） | 8、8、8、8、8 | 6、6、6、6、6 |
| 范围／均值 | 8—8／8 | 6—6／6 |
| 有效性 | 五次 supported | 五次 weak |
| 一致性 | 五次 supported | 五次 supported |

五对图注减基准均为 -2，达到预先定义的“本夹具五对均降分”条件；这不是跨工程统计结论，也不能证明输出始终相同。核对理由可见：基准认可同一组班组／机械与延误增班措施，图注组则强调缺少进场计划、产能测算等资源论证，多次称图片“不加分”。存在输入敏感性，但本实验不能把两分差额全部归因为模型直接惩罚插图；同一未分档条款的主观尺度也可能改变。

10 次均有有效输出，无格式修复或失败；接口返回用量为输入 9,210、输出 3,756 token。适配器内的 HTTP 重试细节和未返回请求的用量不可见，不将返回调用数冒充全部网络尝试数。`*-responses.json` 保留每次交给适配器的请求和全部返回原文，`pair-*.json` 保存正式校验与 grounding 输出。`audit.json` 复核协议／源文件摘要、请求与冻结提示词一致、返回 JSON 与保存原文一致、引用 ID 均属于给定正文、grounding 降级为 0；这不替代引用含义的独立工程核验。

规范 JSON SHA-256（不含末尾换行）：`0a83d369a0c5cdf74e5337695f2b582ac00f4e435892081622c582143ae49296`。

复现须使用新目录，避免覆盖已有协议与结果；复用后端环境及本机配置，不安装依赖：

```powershell
$env:PYTHONUTF8='1'
& tender-app/backend/.venv/Scripts/python.exe tender-research/scripts/caption_repeat_check.py --source tender-research/results/verified-context-1/contrasts/protocol.json --output tender-research/results/caption-repeat-new
```

没有图像像素、真实最终正文／官方技术分配对或独立工程标注，不能宣称真实评分准确率、图片理解或质量验收通过。未修改业务库、正式提示词或部署；单夹具不足以调整正式方法。方法调整需先建立多条款、无关／有关附文、数字矛盾与资源信息增减的独立对照，分别检查误扣、漏判和引用，不把本例当唯一目标。完整浏览器链路是可独立推进的下一项验收。
