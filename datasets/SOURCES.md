# 题集来源审计

本项目只保存题目规格、测试 oracle、必要的适配代码和来源元数据，不直接把第三方仓库的完整实现复制为“标准答案”。每个参考实现都必须保留上游 commit、路径和许可证信息。

| 来源 | 当前基线 | 许可证 | 纳入内容 | 处理规则 |
|---|---|---|---|---|
| SGLang | 49c012d33 | Apache-2.0 | RMSNorm、融合激活、MoE routing、Merge State | 记录上游路径与适配差异 |
| Miles | 05c94552e | Apache-2.0 | 扩展 INT4、MoE backward | 首版不作为主结果 |
| FlashInfer | 已安装包，版本运行时记录 | Apache-2.0（以发布包/仓库为准） | sampling、cascade、paged KV | 只做 API 参考，oracle 独立实现 |
| TileLang | 已安装包，版本运行时记录 | Apache-2.0（以发布包/仓库为准） | TileLang GEMM、online softmax、attention | 模板不直接等同标准答案 |
| Tencent HPC-Ops | 1cd332980ed46bd0172091c1c35d55338fcae47a | MIT；CUTLASS 依赖 BSD-3-Clause | RoPE/KV、FP8 norm、Route GEMM、Grouped GEMM、decode attention | 保留 LICENSE.txt 和第三方 attribution |

## HPC-Ops 适配记录

- hpc_rope_norm_store_kv：保留 PyTorch reference 的分页、位置和 tail-clear 语义，题面隐藏实现细节。
- hpc_rmsnorm_scale：将 FP8 输出误差和 FP32 分支分别作为判定目标，避免只比较量化结果。
- hpc_route_gemm：要求模型解释 high/low BF16 decomposition 与误差来源，不能只检查 GEMM 数值。
- hpc_group_gemm_fp8：显式测试空 group、group skew 和 group permutation。
- hpc_attention_decode：作为扩展题，需先隔离 page table、workspace 和动态调度依赖。

正式发布前，运行 python -m kernelscope.tools.audit_sources 生成机器可读审计报告，包含本地路径存在性、commit、许可证摘要和题目状态。
