# KernelScope-Hy3 实现方案

## 1. 项目定位

KernelScope-Hy3 是面向推理引擎 CUDA/Triton 算子生成的过程评估系统。使用 SGLang 部署 Hy3 作为 Generator，要求模型输出结构化工程决策链、算子代码和可验证的优化声明；随后在 H200 上完成编译、正确性、隐藏测试和性能验证，并定位最早错误步骤。

核心问题：相比只检查最终测试或只使用 LLM Judge，结合结构化声明、源码分析和真实 H200 执行证据，能否更准确判断过程是否成立，并识别结果正确但过程错误的样本？

项目不进行训练、微调或后训练。H200 性能是验证证据，不是唯一正确性标准。

## 2. 首版题集

| 难度 | 算子 | 来源 | 主要能力 |
|---|---|---|---|
| Easy | SiLU-and-Mul | SGLang | 融合、连续访存、尾部 mask |
| Easy | RMSNorm | SGLang | FP32 累加、warp/block 规约 |
| Medium | Fused Add + RMSNorm | SGLang | 融合语义、原地更新、同步 |
| Medium | Attention State Merge | SGLang | log-sum-exp、Inf、非对齐维度 |
| Medium/Hard | MoE Top-k Softmax | SGLang | 数值稳定、top-k、tie、renorm |
| Hard | Fused Q-Norm + RoPE | SGLang | 多算子融合、位置索引、layout |

FlashInfer 和 TileLang 可以加入题集，但职责不同：FlashInfer 作为真实推理算子的参考实现/测试来源；TileLang 作为可生成的 DSL 后端和实现赛道。Miles 的 Blockwise Fake INT4 Quantization 和 MoE backward 仍作为扩展题，不纳入首版主结果：前者需要先编译 CUDA extension，后者依赖较重且现有测试容差较宽。

FlashInfer 当前环境已安装，可提供 sampling、RMSNorm、attention cascade/merge、paged KV、decode attention、MoE routing 等模块。TileLang 当前环境已安装，但安装包主要提供 compiler、tileop 和 flashattention/gemm 模板，不应把 pip 包内模板直接当作题目标准答案；应以题目 oracle 加多种合法 TileLang/CUDA 实现作为标准。

## 3. 算子来源与新增题型

### FlashInfer 候选

优先加入以下两个 FlashInfer 题型：

| 题型 | 参考来源 | 难度 | 过程评估重点 |
|---|---|---|---|
| Top-k/Top-p/Min-p Sampling | FlashInfer sampling API 与 SGLang test_sampling.py | Medium | 概率归一化、过滤顺序、随机性和边界 |
| Cascade Attention State Merge | FlashInfer cascade/merge 与 SGLang test_merge_state_v2.py | Medium | log-sum-exp、Inf、分支合并、数值稳定 |

第二阶段可加入 paged KV append/gather 或 decode attention，但必须先把 page table、slot mapping 和 cache layout 固化为独立 task spec，不能直接调用完整引擎 wrapper。

FlashInfer 题的标准答案应分成三层：PyTorch 数学 oracle、FlashInfer 参考路径、候选生成 kernel。随机 sampling 不比较单次 token 完全相等，而比较随机种子可复现性、采样结果是否落在合法集合内和多次采样分布性质。

### TileLang 候选

TileLang 适合设置为实现后端，而不是单独的“框架 API 题”。同一 task spec 可以要求模型分别生成 CUDA、Triton 或 TileLang 实现，再统一编译并比较：

| 题型 | 推荐后端 | 难度 | 过程评估重点 |
|---|---|---|---|
| Tiled GEMM + bias/activation | TileLang | Medium | tile shape、shared memory、layout、边界 |
| Online Softmax | TileLang | Medium | 数值稳定、分块规约、流水依赖 |
| FlashAttention-style forward（缩小规格） | TileLang | Hard | Q/K/V tile、online softmax、因果 mask |

TileLang 题必须保留 PyTorch oracle 和一个简单 CUDA/Triton baseline；TileLang 编译失败、版本不支持或 kernel cache 失败应归为环境/格式错误，不得混入算法错误。正式数据集记录 TileLang 版本、commit（若可得）、目标架构和生成后端。

### 题集调整建议

首版正式题目仍控制在六题，但可将原来的 Fused Q-Norm + RoPE 替换为更容易独立验证的 FlashInfer Sampling，形成：SGLang 3 题、FlashInfer 2 题、TileLang 1 题。FlashAttention-style TileLang 题放入 Hard 扩展集，避免首版被编译和调参复杂度拖慢。

## 4. 系统架构

任务规格 -> Hy3 Generator -> JSON Schema 校验 -> 结构化过程解析 -> 隔离编译 -> sanitizer 与公开测试 -> 隐藏/随机/变形测试 -> H200 性能测试 -> 声明/代码/硬件证据聚合 -> 过程判定、错误定位、错误分类。

组件包括 Generator、Judge、Static Analyzer、Sandbox Runner、H200 Verifier 和 Evidence Aggregator。Judge 负责语义审查，执行器负责客观判定。

## 5. Generator 输出协议

不要求输出不可审计的自由形式思维链，而要求工程决策与证据声明。核心字段包括 task_understanding、steps、complexity、final_kernel 和 launch_config。每个 step 必须有 id、type、claim、depends_on、code_symbols 和 evidence_expected。解析失败、缺少步骤 ID 或缺少代码字段，直接归类为 FORMAT 错误，不进入 GPU 执行。

## 6. 题集与标准答案

每题包含 spec.yaml、oracle.py、reference、public_tests.json、hidden_tests.json、metamorphic_tests.py、rubric.yaml 和 mutations。测试覆盖对齐/非对齐 shape、小 batch/生产 shape/大 shape、FP16/BF16、空输入、尾部元素、Inf/极值、重复 top-k 分数、随机数据和变形性质。

难度由语义复杂度、规约层级、同步原语、数值稳定性、layout、边界复杂度和融合算子数量评分，每项 0 到 2 分后映射 Easy/Medium/Hard。

参考实现记录仓库、commit、文件路径、许可证和上游来源。当前本地基线为 SGLang commit 49c012d33、Miles commit 05c94552e，正式生成数据集时重新确认。

## 7. 过程评估器

执行顺序固定为：schema -> compile -> compute-sanitizer -> public tests -> hidden tests -> random/metamorphic tests -> performance。性能测试只有 correctness 通过后执行。最终状态分别记录 compile、public_tests、hidden_tests、sanitizer 和 performance，不压缩成单一布尔值。

每个步骤输出四态证据：verified、contradicted、insufficient_evidence、not_applicable。关键词出现只能作为弱证据，不能仅凭 shared、mma 或 cp.async 等字符串判定优化有效。

Hy3 Judge 审查需求理解、假设、公式、算法、步骤依赖、优化选择、复杂度和边界；执行证据优先级高于 Judge。Generator 与 Judge 使用不同 system prompt，并用人工标注校准。

## 8. 错误定位与分类

最早错误步骤定义为：第一个被可靠证据反驳且能导致后续失败的步骤。无法映射时输出 error_detected_but_not_localized，不强行猜测。

错误类型：SPEC（题意、shape、dtype、layout、inplace 语义）；ALGO（公式、规约、算法）；NUM（溢出、精度、log-sum-exp、累加）；MEM（越界、尾部 mask、对齐、访存）；SYNC（barrier、race、atomic）；IMPL（声明与代码不一致）；PERF（occupancy、register spill、低并行度、假优化）；FORMAT（JSON、编译、接口）。允许多标签。

## 9. 结果正确但过程错误

重点构造全零输入下 race condition 巧合通过、只支持公开对齐 shape、声称 Tensor Core 但实际 SIMT、声称 shared memory/double buffering 但源码没有、FP16 累加仅小值通过、Merge State 向量正确但 LSE 错误、Top-k renormalization 未实现、正确代码配错误解释等样本。单独报告 correct_result_invalid_process_recall。

## 10. 评估器有效性验证

从正确样本注入单一错误，构造 6 题 x 6 类型 x 2 变体 = 72 个受控错误样本，由于注错位置已知，可计算精确定位率。每题生成约 10 个候选，收集约 60 个自然样本，由两名标注者独立标注，分歧由第三次复核解决。正确对照集包含参考实现、人工确认的 Hy3 输出、不同但合法的实现策略和正确 naive 实现。

报告最终答案正确率、过程正确率、错误检测召回率、Top-1/Top-2 定位准确率、错误类型 Macro-F1、正确结果错误过程识别召回率、误报率和 insufficient_evidence 比例。比较 Judge-only、Rules+Tests、Judge+Rules 和完整证据聚合器。

## 11. H200 与 SGLang

运行时保存 GPU UUID、compute capability、驱动、CUDA、时钟、功耗和显存状态。性能使用 CUDA Events，预热后重复运行，报告 median/p10/p90 和相对参考实现 latency ratio。不把固定 SM 数量作为通过条件，不使用 nvprof。

SGLang 负责 Hy3 Generator/Judge 部署。README 记录 structured output、RadixAttention 前缀缓存、batch、chunked prefill 和 TTFT/吞吐实验，但这些是系统工程指标，不替代过程评估主结果。

## 12. 仓库结构

建议目录：app（Generator、Judge、schema）、tasks（题目、oracle、reference、tests、rubric）、evaluator（静态、执行、证据、定位、分类）、sandbox（编译和运行限制）、datasets（任务、受控错误、自然样本、标注）、experiments（生成、评估、消融、报告）、configs（SGLang 和 H200 配置）、reports（表格、案例、人工抽检）。

## 13. 里程碑

MVP 先完成 RMSNorm、Attention State Merge、MoE Top-k Softmax 三题，打通 schema、生成、编译、隐藏测试、claim 检查、Judge、定位和 JSON 报告，准备至少 30 个受控错误样本和 15 个正确样本。

正式版扩展到六题，完成 72 个受控错误、约 60 个自然样本、至少 30 个正确过程样本，完成双人标注、消融、H200 性能基线和难度分析。

## 14. Demo

以 Attention State Merge 为例：输入需求 -> Hy3 输出结构化决策和 kernel -> 公开测试通过 -> 隐藏 Inf/大 LSE 测试失败 -> 定位数值稳定性步骤 -> 展示源码证据和失败输入 -> Hy3 修复 -> 隐藏测试通过并输出报告。提前完成模型服务和 JIT 预热，Demo 不现场等待长时间编译。

## 15. 风险与交付物

H200 被占用时使用独占 GPU、记录状态并标注 contended；Judge 自偏好用 Judge-only 基线、人工标注和执行证据校准；不同合法实现使用多实现正确集和 insufficient_evidence；测试覆盖用隐藏、随机、变形和 sanitizer；生成代码在无网络、临时目录、资源限制和超时环境执行。

最终交付：可运行的 SGLang/Hy3 Generator 与 Judge，六道分层题及标准答案/隐藏测试，沙盒执行器和证据聚合器，过程正确性/定位/分类脚本，完整结果与人工抽检记录，README、环境配置、运行说明和两分钟 Demo。
