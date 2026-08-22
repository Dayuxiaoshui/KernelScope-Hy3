# H200 基线数据

## h200_shared_initial.json

首次管线采集，设备为 NVIDIA H200（SM90），PyTorch 2.11.0+cu130，CUDA 13.0。物理 GPU 4 在采集前连续 6 次利用率采样均为 0%，但存在约 62 GiB 常驻显存和其他进程，因此所有记录均标记为 clean_environment=false。

本文件只验证以下能力：

- 参考实现 correctness 先于性能测量；
- 多 shape、多 provider 数据可以落盘；
- median、p10、p90、CV 和环境元数据完整；
- 非独占或高噪声结果不会进入正式 reference envelope。

首次结果中，SGLang RMSNorm/Attention State Merge 的中位延迟约为 14--16 us，Torch oracle 约为 75--261 us。由于部分 CV 超过 5%，不得将这些数值用于正式排名、阈值或论文结论。

正式基线要求：独占 H200、锁定或记录时钟、采集前后 GPU 状态、至少三轮独立重复、每轮足够预热，并仅接纳 clean_environment=true 且 CV <= 0.05 的记录。

## h200_shared_v2.json

使用 50 次预热、100 个样本、每个 CUDA Event 样本内重复 20 次 launch，降低微秒级 kernel 的计时与同步开销。覆盖 RMSNorm 和 Attention State Merge；RMSNorm 同时记录 SGLang、FlashInfer 与 Torch oracle，Merge State 记录 SGLang 与 Torch oracle。15 条记录中 11 条满足 CV <= 0.05，但因共享卡全部保留 `clean_environment=false`。

SGLang 与 FlashInfer 的包装层可能调用相同或高度相关的底层 kernel，因此二者不能自动视为独立实现。正式 reference envelope 还需要增加来源独立的实现，例如 Triton/TileLang 或自行审核的 CUDA reference。

## h200_sampling_shared_v1.json

覆盖 top-k-only、top-p-only 和小词表 tail shape，共 3 个 case、2 个 provider、6 条记录。每个 provider 在计时前均与独立 PyTorch oracle 比对。该文件只评估概率过滤和重新归一化，记录中的 `scope` 固定为 `filtering_only`；它不代表随机抽样、随机数状态或联合 top-k/top-p/min-p 的完整性能。

本轮大词表 top-p（batch=99, vocab=128256）中位延迟为 SGLang 214.138 us、FlashInfer 260.410 us。共享环境中的 SGLang top-k 记录 CV=0.1689，超过接纳阈值，说明独占卡复测是必要步骤。联合过滤 case 已保留在题集，但在具备语义一致的 full-operator provider 之前不会进入本过滤基线。

## h200_softmax_shared_v1.json

使用项目内可审计的 Triton 稳定 Softmax 与 Torch FP32 oracle，覆盖 `cols=1024`、非对齐 `cols=1537` 和 BF16 `cols=8192`。三个 case 均在计时前通过数值校验。Triton 中位延迟为 14.034--14.298 us，Torch 组合实现为 21.541--22.034 us。共享卡导致部分 CV 超限，因此仍不进入正式 envelope。

## h200_scaled_rmsnorm_shared_v1.json

依据 Tencent HPC-Ops 原始测试契约，普通模式计算 RMSNorm 后除以静态 scale，并转换为 `float8_e4m3fn`。FlashInfer fused provider 与独立 E4M3 oracle 在计时前按 `rtol=0.0125, atol=0.15` 校验。两个普通模式 case 中，fused kernel 中位延迟为 11.533--11.802 us，Torch oracle 为 87.531--87.746 us，所有记录 CV 均低于 5%。

HPC-Ops 的 MoE 模式返回 FP32 归一化结果和两个不同 scale 的 FP8 输出，契约不同于普通 FlashInfer kernel。该 case 保留在题集，但在接入 HPC-Ops 原生实现或等价 provider 前不采集性能基线。

## h200_route_gemm_shared_v1.json

加载固定 commit `1cd332980ed46bd0172091c1c35d55338fcae47a` 编译的 HPC-Ops SM90a 扩展，使用原生 `gemm_bf16xfp32` 与 Torch FP32 oracle。两个 `N=256` case 通过 correctness，HPC-Ops 中位延迟为 15.856--15.974 us，Torch oracle 为 24.637--25.184 us。`N=257` tail case 被原生实现明确拒绝（要求 N divisible by 64），因此没有伪造性能记录；它仍是重要的边界失败样本。

该文件仍为共享 GPU 数据（`clean_environment=false`），正式排名需在独占 H200 上复测。

## 自动评分口径

`kernelscope.tools.score_baselines` 只接纳带 `correctness_gate=true` 的新格式记录，并要求 provider 的 `clean_environment=true` 且 CV <= 0.05 才能进入 reference envelope。早期基线没有 correctness 字段，评分器会将其排除并给出缺失告警；需要按当前采集器重新采集后再做正式排名。示例：

```bash
PYTHONPATH=src python -m kernelscope.tools.score_baselines \
  --task hpc_route_gemm --candidate hpc_ops_native \
  --baseline datasets/baselines/h200_route_gemm_shared_v1.json \
  --correctness-passed
```

## h200_scaled_rmsnorm_native_shared_v1.json

使用同一 HPC-Ops SM90a 原生扩展的 `fused_rmsnorm_with_scale`，普通模式两个 case 均通过 E4M3 FP8 oracle。原生 kernel 中位延迟为 10.869--10.899 us；Torch oracle 为 81.213--81.466 us。该结果显示 native HPC-Ops provider 比 FlashInfer fused 版本略快，但 hidden case 的共享卡 CV=0.0949，不能进入正式排名。

## h200_formal_hpc_run4.json

GPU 7 独占窗口下使用 `launches_per_sample=1000`、100 iterations 和 50 warmup。该聚合窗口将微秒级 Event 量化噪声显著降低：8 条记录全部 CV <= 0.05，且 correctness gate 全部通过。HPC-Ops native RMSNorm 相对 Torch oracle ratio 为 8.237--8.266；Route GEMM ratio 为 1.623--1.629。MoE RMSNorm 仍保留为缺失能力告警。Route GEMM 的 N=257 tail 使用 padding adapter：补齐权重和输出到 64 的倍数后调用原生 kernel，再切回真实 N，padding 和切片开销计入性能。

## h200_formal_route_padded_gpu5_v1.json

GPU 5 空闲窗口采集了 Route GEMM 的 public、small-M 和 N=257 tail 三个 case。padding adapter 将原生 N 维补齐到 64 的倍数，并在 kernel 外切片输出。三个 candidate 记录 correctness gate 和 clean environment 均通过，CV 分别为 0.0498、0.0305、0.0039；Torch reference 在 small-M case CV=0.0583，因此该 case 被评分器跳过。public 与 tail 的正式 ratio 为 1.311 和 1.303，padding 开销已计入。
