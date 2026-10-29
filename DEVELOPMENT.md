# 开发分支与跨仓协作

后续提交使用 dev，跟踪 origin/dev。开始工作先 fetch、核对工作区与并行提交；正常合并，禁止 reset 覆盖他人历史或强推。保留原来源分支用于追溯，不继续向旧分支分散新开发。

项目关联仓库仅 tender-app、tender-docs、tender-research、tender-design、tender-datasets；各仓提交不同，不要求SHA相等。应用主源码在 workbench，backend/frontend供体的既有dev改动保留。正式文档在 tender-docs；行为变更同步正式主记录和验收边界，阶段记录不替代需求验收。

推送dev不代表部署许可、模型资格、付费许可或生产验收。应用Workbench CI仅运行检查，支持dev；其余仓当前没有Actions工作流。外部webhook配置无读取权限，未声称已检查全部外部集成。
