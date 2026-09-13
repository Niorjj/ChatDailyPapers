# 多模态与模型加速论文日报

每天自动检索 arXiv，以 SenseNova 6.8 Flash Lite 对多模态及模型加速优化论文评分并生成中文摘要，发送至 `yangxue7410@gmail.com`，同时保存 Markdown 并创建 GitHub Issue。

## 部署配置

在仓库 **Settings → Secrets and variables → Actions** 添加：

| Secret | 内容 |
|---|---|
| `SENSENOVA_API_KEY` | SenseNova API Key |
| `EMAIL_SENDER` | `yangxue7410@gmail.com` |
| `EMAIL_APP_PASSWORD` | Gmail 16 位应用专用密码（去除空格） |

然后在 **Settings → Actions → General → Workflow permissions** 选择 **Read and write permissions**。

进入 **Actions → Daily paper digest → Run workflow** 可立即测试。定时任务每天 UTC 23:30 运行，通常对应北京时间 07:30。GitHub 可能有少量排队延迟。

## 工作流程

1. 在多个 AI、视觉、语言和系统 arXiv 分类中回溯 72 小时。
2. 用 `config.yaml` 中的关键词高召回初筛。
3. 调用 `sensenova-6.8-flash-lite` 对最多 30 篇候选论文评分和总结。
4. 发送 Top 8 的 HTML、纯文本及 Markdown 附件。
5. 用 `state/seen.json` 去重；即使当天无符合条件论文也发送状态邮件。

## 自定义

编辑 `config.yaml` 可调整关键词、分类、数量、评分阈值和回溯时间。接口基址固定为 `https://token.sensenova.cn/v1`，SDK 会调用 `/chat/completions`。

密钥只能保存在 GitHub Secrets 中，绝不能写入源码、配置或提交记录。
