# H200 formal run 1 audit

采集前 GPU 7 仅有约 126 MiB 驱动占用，计算进程查询为空。采集结束后的复查发现 GPU 7 出现 PID 627811（约 78 GiB）和 PID 656677（约 9 GiB），无法证明整个采集窗口保持独占。

因此 `h200_formal_core_run1.json` 和 `h200_formal_hpc_run1.json` 的所有 measurement 均降级为 `clean_environment=false`。这些数据可用于功能验证和确定复测 shape，但不得进入正式 reference envelope。

正式采集必须增加租卡/调度器锁或持续进程监控，并在发现新的 compute PID 时使整轮失败。
