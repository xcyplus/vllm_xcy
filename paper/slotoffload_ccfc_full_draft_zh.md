# SlotOffload：面向单机 Agent 服务的结构感知 KV Cache 卸载准入

> CCF-C/同等级系统方向中文论文完整初稿。  
> 作者、单位、基金、会议模板、页边距和参考文献格式待按目标会议统一调整。  
> 当前定位为单机单 GPU LLM serving 系统优化小论文，不按 SC/HPC 顶会叙事包装。
> 当前实验数据来自已完成的 0.5B/3B/7B core，以及 0.5B/3B/7B smoke；长输出、正确性和开销实验待补。

## 摘要

大语言模型在线服务广泛使用 KV Cache 和前缀缓存来降低预填充开销。随着 Agent 应用普及，请求提示词逐渐呈现出明显的结构化形态：角色指令、工具 schema、输出约束在大量请求间重复，而用户输入、工具参数和工具观测结果随请求变化。现有 KV Cache CPU 卸载机制通常按块保存可卸载 KV，不区分提示词内部片段的复用价值，容易将低复用的动态槽位写入 CPU 缓存，造成写入放大、带宽浪费和缓存污染。本文聚焦单机单 GPU 的私有化 Agent serving 场景，而非多节点 supercomputing 或集群级调度问题。

本文提出 SlotOffload，一种面向结构化 Agent 提示词的 KV Cache 卸载准入机制。SlotOffload 由应用在组装 prompt 时附加轻量级 token 区间元数据，将输入划分为 instruction、schema、slot 和 observation 等类型；在 vLLM V1 OffloadingConnector 的 STORE 路径上，SlotOffload 对每个 KV block 判断是否值得写入 CPU。我们实现两种轻量策略：延迟优先的 Binary 结构过滤策略，以及写入效率优先的 Value 价值准入策略。该机制只改变 CPU offload 的准入决策，不改变注意力计算、KV 内容和前缀哈希匹配语义；对未携带结构元数据的请求保持 vLLM 原生行为。

我们在 vLLM 中实现 SlotOffload，并构造 Agent/MCP 风格结构化服务 workload，覆盖金融风控、客服工单、合同合规、安全告警、物流异常、设备维护等 8 类 Agent 模板。实验在 Qwen2.5-0.5B、3B 和 7B 上进行。结果表明，在 Qwen2.5-7B 的核心实验中，Binary 相对 vLLM 原生 CPU offload 最高降低 48.34% 平均 TTFT，Value 最高降低 35.45% 平均 TTFT；两者均可减少 96.37%--99.09% 的 CPU KV 写入。实验也显示，Binary 通常具有更好的延迟，而 Value 通常具有更少的写入量。本文不声称 Value 全面优于 Binary，而是将二者定位为延迟优先与写入效率优先的不同取舍。

**关键词：** 大语言模型推理；KV Cache；Agent；MCP；前缀缓存；缓存卸载；vLLM

## 1 引言

KV Cache 是大语言模型在线推理系统的核心优化之一。在自回归 Transformer 中，输入 prompt 的预填充阶段会为每一层生成 Key 和 Value；当后续请求具有相同前缀时，系统可以直接复用此前的 KV Cache，避免重复预填充计算。vLLM 等推理系统通过 PagedAttention 和 prefix caching 支持高效的 KV 管理，并进一步提供 CPU KV offload：当 GPU 显存无法容纳更多 KV block 时，将已计算的 KV 写入 CPU 内存，在后续命中时再加载回 GPU。

CPU offload 扩大了逻辑缓存容量，但也带来新的系统问题。首先，GPU 到 CPU 的写入并非免费，会消耗 PCIe 带宽和 CPU 内存带宽。其次，CPU 缓存容量依然有限，写入低价值 block 会挤出高价值 block。最后，CPU 到 GPU 的加载也会引入额外延迟；若一个 block 未来不会被命中，或者重算成本低于传输成本，则写入本身就是无效开销。

Agent 服务使这一问题更加突出。一个典型 Agent prompt 包含角色说明、任务规则、工具列表、工具 inputSchema、输出 JSON schema、当前用户输入、工具调用参数和工具返回结果。前几类内容在同一 Agent 模板内高度稳定，而用户输入和工具观测高度动态。例如客服 Agent 的“退款前必须查询订单”“search_order(order_id) 工具 schema”会被大量请求复用；订单号、用户诉求和查询返回结果则几乎每次不同。若推理系统不区分这些结构，原生 offload 会把稳定 schema 和动态 slot 一起写入 CPU，造成严重写入放大。

本文关注的问题是：**在不改变模型计算和前缀正确性的前提下，推理系统能否利用 Agent prompt 的结构信息，只将更有复用价值的 KV block 写入 CPU？** 为此，我们提出 SlotOffload。SlotOffload 不尝试跨不同 prompt 融合 KV，也不放宽 vLLM 的精确前缀匹配条件，而是在 vLLM 准备执行 STORE 操作时增加一个 admission policy。它根据应用传入的 token range metadata 计算每个 block 中 instruction、schema、slot、observation 的比例，并结合历史访问、复用距离、计算与传输代价以及 CPU cache pressure，决定 STORE 或 SKIP。

需要强调的是，本文不是面向 supercomputing 场景的多节点 LLM serving 系统，也不讨论跨 GPU 负载均衡、集群级调度或资源隔离。本文的目标更加收敛：面向单机 GPU 上常见的私有化 Agent 服务，研究结构化 prompt 信息能否改善 vLLM CPU KV offload 的写入效率和首 token 延迟。这样的定位更符合本文实现和实验规模。

本文贡献如下：

1. **场景抽象。** 将 KV offload 问题扩展到 Agent/MCP 风格结构化提示词服务，指出 instruction/schema 与 slot/observation 在 KV 复用价值上的系统性差异。
2. **结构感知。** 设计 token range metadata 表示，将 prompt 内部结构传递给 vLLM，使 offload 策略能识别块内混合组成，而不是只按 token 序列或 block hash 决策。
3. **两种准入策略。** 实现延迟优先的 Binary 结构过滤和写入优先的 Value 价值准入，明确讨论二者在延迟、写入量和命中之间的取舍，而不是声称单一策略全面最优。
4. **vLLM 原型。** 在 vLLM V1 OffloadingConnector 上完成 STORE 前准入实现，保持对无 metadata 请求的原生行为，并提供后台实验、进度查询和结果汇总脚本。
5. **实验评估。** 构造 Agent/MCP structured workload，在不同模型规模、访问分布和复用距离下比较 GPU-only、native、threshold、binary 和 value 五种策略，展示 SlotOffload 对写入量、有效命中和 TTFT 的影响，同时说明单机合成 workload 的边界。

## 2 背景与动机

### 2.1 KV Cache、Prefix Cache 与 CPU Offload

Transformer 解码每个新 token 时需要访问前文 token 的 Key 和 Value。预填充阶段一次性处理 prompt 并生成 KV Cache，解码阶段持续追加新 token 的 KV。若多个请求共享完全相同的前缀，系统可跳过已命中前缀的预填充。vLLM 将 KV Cache 组织为固定大小 block，并通过 block hash 实现 prefix cache。

CPU offload 的基本思想是将 GPU 中暂时不活跃但可能未来复用的 KV block 写入 CPU 内存。后续请求若命中这些 block，可从 CPU 加载回 GPU，避免重算。设 block \(b\) 包含 \(B\) 个 token、KV 大小为 \(K_b\) 字节，GPU 到 CPU 和 CPU 到 GPU 有效带宽分别为 \(BW_{g2c}\) 和 \(BW_{c2g}\)，则存取开销近似为

\[
T_s(b)=\frac{K_b}{BW_{g2c}}, \quad
T_l(b)=\frac{K_b}{BW_{c2g}}.
\]

若该 block 未来被复用，可避免 \(B\) 个 token 的预填充重算；若不复用，\(T_s(b)\) 与 CPU 空间占用就是纯开销。因此，CPU offload 的关键不只是“能不能存”，而是“值不值得存”。

### 2.2 Agent/MCP Prompt 的结构特征

Agent prompt 与普通问答 prompt 的显著差异在于其结构稳定性。一个常见 Agent 请求可抽象为

\[
P = I \Vert S \Vert U \Vert O,
\]

其中 \(I\) 是 instruction，包括角色、目标和规则；\(S\) 是 schema，包括工具定义、inputSchema 和输出格式；\(U\) 是 slot，包括用户输入、参数和业务字段；\(O\) 是 observation，包括工具返回结果、历史摘要或环境观测。

在同一 Agent 模板内部，\(I\) 和 \(S\) 经常完全相同，\(U\) 和 \(O\) 则随请求变化。以客服 Agent 为例，退款规则和订单查询工具 schema 会在大量请求间重复，而订单号、商品、用户诉求和订单状态每次不同。若服务系统无差别 offload，动态 slot 和 observation 会大量进入 CPU cache，却很难产生后续命中。

### 2.3 为什么不是直接做真实 Agent 调用

本文不研究 Agent planner、工具选择正确率或 MCP server 协议执行本身。研究对象是 LLM serving 层的 KV Cache 管理。因此，实验使用 Agent/MCP 风格结构化 prompt 和伪工具观测结果来模拟真实 Agent 请求形态，但不执行外部工具调用。这样的设计有两个好处：一是可以隔离 KV offload 策略的系统效果；二是可以控制模板频率、复用距离和 slot 变化，从而进行可重复实验。

### 2.4 设计目标

SlotOffload 的设计目标包括：

- **减少写入放大。** 避免将一次性 slot 或低复用 observation 大量写入 CPU。
- **保持有效命中。** 在减少写入的同时尽量保留 instruction/schema 等高价值 block。
- **适应系统压力。** CPU cache 越紧张，准入门槛越高。
- **保持正确性。** 不近似 KV、不改变注意力、不改变前缀匹配语义。
- **易于集成。** 作为 vLLM offload STORE 前的 admission policy 实现。

## 3 方法设计

### 3.1 Prompt 结构元数据

SlotOffload 要求应用在组装 prompt 时生成 token range metadata。metadata 是随请求传入 vLLM 的结构标注，不改变 prompt 文本，也不改变模型输出。例如：

```json
{
  "slot_offload": {
    "instruction_ranges": [[0, 120]],
    "schema_ranges": [[120, 430]],
    "slot_ranges": [[430, 520]],
    "observation_ranges": [[520, 610]]
  }
}
```

range 是 token 下标范围，采用左闭右开形式。metadata 通过 OpenAI-compatible request 的 `kv_transfer_params` 传入：

```json
{
  "model": "...",
  "prompt": "...",
  "max_tokens": 1,
  "kv_transfer_params": {
    "slot_offload": {
      "instruction_ranges": [[0, 120]],
      "schema_ranges": [[120, 430]],
      "slot_ranges": [[430, 520]],
      "observation_ranges": [[520, 610]]
    }
  }
}
```

该设计不需要神经网络分类 token，因为模板系统、Agent 框架或业务网关在生成 prompt 时天然知道哪些片段是工具 schema、哪些片段是用户输入。

### 3.2 块内结构比例

vLLM 的 KV 管理以 block 为单位，而结构片段边界不一定与 block 边界对齐。因此，SlotOffload 计算每个候选 KV block 中不同类型 token 的比例。设 block \(b\) 包含 \(B\) 个 token，类型 \(k\) 的 token 数为 \(n_{b,k}\)，槽位比例为

\[
r_{slot}(b)=\frac{n_{b,slot}}{B}.
\]

结构分数定义为

\[
S(b)=clip\left(\sum_{k\ne slot} w_k\frac{n_{b,k}}{B}
-\lambda r_{slot}(b),0,1\right).
\]

默认权重设置为：instruction/system/template 为 1.0，schema 为 0.9，context 为 0.7，observation 为 0.4，slot 惩罚 \(\lambda=1.0\)。observation 的权重低于 schema，因为工具返回结果通常比工具定义更动态。

### 3.3 热度与复用距离

仅有结构信息不足以判断是否值得写入：一个纯 schema block 若只出现一次，仍然不应占用 CPU cache。SlotOffload 为每个 block hash 维护访问历史。第 \(t\) 次观察到 block \(b\) 时，其衰减访问计数更新为

\[
c_t(b)=c_{t-1}(b)e^{-\gamma \Delta_t}+1,
\]

其中 \(\Delta_t\) 是距离上次观察经过的候选 block 事件数，\(\gamma\) 是衰减系数。热度分数为

\[
H(b)=\min(1,\frac{c_t(b)}{C_{sat}}).
\]

因此，同样出现多次的 block，如果复用距离更短，会获得更高热度。

### 3.4 代价收益估计

设一个 block 的累计访问次数为 \(a_b\)，复用概率估计为

\[
P_r(b)=\frac{a_b}{a_b+\eta},
\]

其中 \(\eta\) 是平滑项。设每 token 预填充时间为 \(t_p\)，则重算代价为

\[
T_r(b)=B\cdot t_p.
\]

期望净收益定义为

\[
G(b)=P_r(b)T_r(b)-T_s(b)-P_r(b)T_l(b).
\]

若 \(G(b)<0\)，说明即使命中概率存在，存取传输也可能不划算，策略直接拒绝 STORE。否则将收益归一化为代价分数

\[
C(b)=clip(\frac{G(b)}{T_r(b)},0,1).
\]

### 3.5 压力自适应阈值

综合价值分数为

\[
V(b)=\alpha S(b)+\beta H(b)+\delta C(b).
\]

设 CPU cache 使用率为 \(q\)，准入阈值为

\[
\theta(q)=\min(1,\theta_0+\mu q).
\]

当 CPU cache 空闲时，系统可以较宽松地保存候选 block；当 CPU cache 接近满载时，只有高价值 block 能够写入。

最终决策流程为：

1. 若没有 metadata，则保持 vLLM 原生 STORE 行为。
2. 若 \(r_{slot}(b)>r_{max}\)，拒绝。
3. 若访问次数小于 `min_accesses`，拒绝。
4. 若 \(G(b)<0\)，拒绝。
5. 若 \(V(b)\ge\theta(q)\)，STORE；否则 SKIP。

### 3.6 Binary 与 Value 两种策略

本文评估两种 SlotOffload 策略。

**Binary SlotOffload** 是结构过滤基线。它只根据块类型决定是否保存：instruction、schema、context、system、unknown 被保存，slot 和 observation 被跳过。它验证“结构信息是否有用”。

**Value-aware SlotOffload** 是完整策略。它进一步考虑热度、复用距离、传输代价和 CPU cache pressure。它验证“结构信息之外，价值准入是否能进一步降低写入放大和缓存污染”。

二者的区别可以概括为：binary 问“这个 block 是不是稳定结构”，value 问“这个 block 未来收益是否值得当前写入成本”。

## 4 vLLM 实现

### 4.1 集成位置

SlotOffload 基于 vLLM V1 OffloadingConnector 实现。vLLM 正常完成 prefill 并生成 GPU KV block；当 OffloadingConnector 准备创建 STORE job 将 KV 写入 CPU 时，调度器调用 SlotOffloadAdmissionPolicy。若返回 STORE，后续流程完全复用原生 offload；若返回 SKIP，则该 block 不写入 CPU。

```text
请求 prompt + kv_transfer_params
        |
        v
vLLM prefill 生成 GPU KV block
        |
        v
OffloadingConnector 准备 STORE
        |
        v
SlotOffloadAdmissionPolicy
        |
   STORE 或 SKIP
```

### 4.2 代码修改

主要实现包含以下模块：

- `vllm/entrypoints/slot_offload_prompt.py`：将 typed prompt parts 转换为 prompt 文本和 token ranges。
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/slot_policy.py`：实现结构分析、热度表、代价模型和准入决策。
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`：在 STORE job 创建前调用准入策略，并在 reset external cache 时清空热度状态。
- `benchmarks/slot_offload_benchmark.py`：构造 Agent/MCP workload，生成 prompt 和 metadata，发送请求并采集指标。
- `examples/features/run_slot_offload_experiment_suite.sh` 与 `run_slot_offload_agent_experiments.sh`：自动启动 vLLM、运行策略矩阵、后台执行与进度查询。

### 4.3 Reset 与实验公平性

Value-aware 策略维护 block 热度。如果多个实验复用同一个 server 而不清空热度，后续策略会继承前一组 workload 的访问历史，导致不公平。我们在 connector 的 `reset_cache()` 中同步调用 SlotOffloadAdmissionPolicy 的 `reset()`，确保外部 prefix cache reset 时，CPU offload cache 和策略历史一起清空。

### 4.4 复杂度

每个候选 block 的结构统计最多扫描 \(B\) 个 token 标签，\(B\) 是固定 block size，可视作常数。热度表使用有序映射，平均更新开销为 \(O(1)\)，并通过 `slot_offload_max_tracker_size` 限制内存。该实现不增加 GPU kernel，不改变 KV 数据格式，主要开销位于 CPU 调度路径。

## 5 实验设计

### 5.1 实验问题

实验回答以下问题：

- **RQ1：** SlotOffload 能否减少 CPU KV 写入？
- **RQ2：** 减少写入后是否仍能保持有效命中？
- **RQ3：** 在 7B 模型上，结构感知策略是否能改善 TTFT，并与 GPU-only 和 threshold 基线相比如何？
- **RQ4：** 模型规模、模板频率和复用距离如何影响收益？
- **RQ5：** Binary 与 Value 的延迟/写入取舍是什么？

### 5.2 实验环境

实验在单机单 GPU 环境中进行，GPU 显存为 48 GiB。模型包括 Qwen2.5-0.5B-Instruct、Qwen2.5-3B-Instruct 和 Qwen2.5-7B-Instruct。为控制变量，不同模型使用校准后的 `--kv-cache-memory-bytes`，使 GPU KV 容量约为 1024 tokens；CPU KV cache 设置为 GPU KV 容量的 4 倍。主要配置如下：

| 模型 | GPU KV bytes | CPU KV bytes | MAX_MODEL_LEN |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B | 12,582,912 | 50,331,648 | 1024 |
| Qwen2.5-3B | 37,748,736 | 150,994,944 | 1024 |
| Qwen2.5-7B | 58,720,256 | 234,881,024 | 1024 |

Agent/MCP prompt 的 token 长度约为 697--798 tokens，因此 `MAX_MODEL_LEN=1024` 能覆盖全部请求。核心实验中每个请求 `max_tokens=1`，这是一个有意设计的 prefill-oriented microbenchmark，用于隔离 prompt KV 复用和 CPU offload 准入对 TTFT 的影响。该设置不代表完整 Agent 会话中的长输出、工具调用循环或 decode-heavy 服务。为避免过度外推，最终稿需要补充 `max_tokens=16/64` 的长输出实验，展示 decode 阶段占比增大时收益如何被稀释。

### 5.3 Workload 构造

我们构造 8 类 Agent/MCP 模板，包括个人贷款、交易反欺诈、客服工单、合同合规、作业评分、安全告警、物流异常和预测性维护。每个请求包含四段：

- instruction：Agent 角色、目标和规则；
- schema：MCP 工具定义、inputSchema 和输出 JSON Schema；
- slot：当前用户请求 JSON；
- observation：伪工具返回结果 JSON-RPC。

同一 Agent 模板内 instruction 和 schema 保持稳定，slot 和 observation 随请求变化。访问分布包括：

- `uniform`：8 个 Agent 模板近似等频；
- `zipf`：少数模板热门，大量模板长尾；
- `one_hit`：部分模板只出现一次，测试冷模板污染；
- `reuse_short`：同模板连续出现，复用距离约为 1；
- `reuse_long`：同模板间隔较久出现，复用距离约为 8。

核心实验每组包含 512 个请求，3 个随机种子，5 种策略，共 \(5\times3\times5=75\) 组；结果聚合为 25 行均值。

### 5.4 对比策略

- **GPU-only：** 不启用 CPU offload，只使用 GPU prefix cache。
- **Native：** vLLM 原生 CPU offload。
- **Threshold：** 使用 vLLM store_threshold 类阈值策略。
- **Binary：** SlotOffload 二元结构过滤策略。
- **Value：** SlotOffload 价值感知策略。

### 5.5 指标

主要指标包括平均 TTFT、P95/P99 TTFT、请求吞吐、CPU STORE MiB、CPU LOAD MiB、external prefix hit tokens、相对 native 的 TTFT 变化、STORE 减少比例和单位写入命中效率。

需要强调的是，单独减少 STORE 不足以证明策略有效，因为“完全不写入”也能减少 STORE。因此本文同时报告 hit tokens、TTFT 和单位写入命中效率：

\[
E_{store}=\frac{HitTokens}{StoreMiB}.
\]

原始结果表中还保留 hit retention，但本文将其视为辅助指标。由于 native、threshold 与结构策略的写入集合不同，hit retention 可能为 0 或超过 100%，直接用它解释性能容易产生歧义；更稳妥的解释应结合 STORE MiB、LOAD MiB、HitTokens 和 TTFT。

## 6 实验结果

### 6.1 7B 核心实验结果

表 1 给出 Qwen2.5-7B 上的核心实验。为避免选择性展示，表中包含 GPU-only、Native、Threshold、Binary 和 Value 五种策略。可以看到，Binary 和 Value 相比 Native 均显著减少 CPU 写入，并在多数分布下降低平均 TTFT。

**表 1：Qwen2.5-7B Agent/MCP core 实验结果**

| 分布 | 策略 | TTFT mean (ms) | TTFT vs native | STORE MiB | LOAD MiB | Hit tokens | Hit/Store |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| one_hit | gpu_only | 65.745 | -48.29% | 0.00 | 0.00 | 0 | - |
| one_hit | native | 44.337 | 0.00% | 6806.62 | 8490.12 | 155248 | 22.8 |
| one_hit | threshold | 53.294 | -20.20% | 114.62 | 5346.25 | 97760 | 852.9 |
| one_hit | binary | 41.405 | +6.61% | 214.38 | 8708.88 | 159248 | 742.8 |
| one_hit | value | 41.709 | +5.93% | 143.79 | 8737.46 | 159771 | 1111.1 |
| reuse_long | gpu_only | 81.153 | +2.33% | 0.00 | 0.00 | 0 | - |
| reuse_long | native | 83.087 | 0.00% | 20104.00 | 0.00 | 0 | 0.0 |
| reuse_long | threshold | 64.006 | +22.97% | 218.75 | 6942.25 | 126944 | 580.3 |
| reuse_long | binary | 42.922 | +48.34% | 214.38 | 13505.62 | 246960 | 1152.0 |
| reuse_long | value | 53.631 | +35.45% | 182.00 | 10342.50 | 189120 | 1039.1 |
| reuse_short | gpu_only | 37.611 | +1.96% | 0.00 | 0.00 | 0 | - |
| reuse_short | native | 38.363 | 0.00% | 6047.12 | 0.00 | 0 | 0.0 |
| reuse_short | threshold | 37.549 | +2.12% | 0.00 | 0.00 | 0 | - |
| reuse_short | binary | 38.284 | +0.21% | 214.38 | 0.00 | 0 | 0.0 |
| reuse_short | value | 38.266 | +0.25% | 219.62 | 0.00 | 0 | 0.0 |
| uniform | gpu_only | 79.804 | -17.07% | 0.00 | 0.00 | 0 | - |
| uniform | native | 68.167 | 0.00% | 14791.58 | 4869.38 | 89040 | 6.0 |
| uniform | threshold | 64.111 | +5.95% | 217.29 | 6506.21 | 118971 | 547.5 |
| uniform | binary | 43.058 | +36.83% | 214.38 | 13070.17 | 238997 | 1114.8 |
| uniform | value | 51.715 | +24.13% | 180.25 | 10536.75 | 192672 | 1069.0 |
| zipf | gpu_only | 68.876 | -33.51% | 0.00 | 0.00 | 0 | - |
| zipf | native | 51.589 | 0.00% | 9286.38 | 6763.46 | 123675 | 13.3 |
| zipf | threshold | 57.887 | -12.21% | 135.04 | 4339.71 | 79355 | 587.6 |
| zipf | binary | 41.561 | +19.44% | 214.38 | 9361.62 | 171184 | 798.5 |
| zipf | value | 44.589 | +13.57% | 187.83 | 8635.96 | 157915 | 840.7 |

**写入放大显著降低。** Native 在 uniform、zipf 和 reuse_long 中分别写入 14.8 GiB、9.3 GiB 和 20.1 GiB KV 数据。Binary 将写入降低到约 214 MiB，Value 进一步降低到 180--188 MiB。Value 在 7B core 中减少 96.37%--99.09% STORE，说明原生 offload 写入了大量低价值动态 KV。

**延迟收益在 7B 上显现。** 在 uniform、zipf 和 reuse_long 中，Binary 分别降低 36.83%、19.44% 和 48.34% 平均 TTFT；Value 分别降低 24.13%、13.57% 和 35.45%。这表明在较大模型中，过滤低价值写入和保留共享结构能够转化为端到端收益。

**Binary 延迟更优，Value 写入更少。** 在多数场景中，Binary 的 TTFT 最低，而 Value 的 STORE MiB 最低。原因是 Binary 更积极地保存所有稳定结构块，命中更多，延迟更好；Value 更保守，进一步减少写入和 cache 污染，但可能牺牲部分命中。二者体现了“延迟优先”和“写入效率优先”的不同取舍。

**GPU-only 是必要基线。** 在 `reuse_short` 中，GPU-only 与所有 CPU offload 策略接近甚至略快，说明当复用距离极短时，GPU prefix cache 已经足以覆盖复用，CPU offload 的额外收益有限。在 `one_hit`、`uniform` 和 `zipf` 中，Native 明显优于 GPU-only，说明 CPU offload 本身在该受限 GPU KV 容量下是有价值的；SlotOffload 的作用不是简单关闭 offload，而是避免 Native 的低价值写入。

### 6.2 复用距离影响

`reuse_short` 中所有策略 TTFT 接近：

| 策略 | TTFT mean (ms) |
| --- | ---: |
| native | 38.363 |
| binary | 38.284 |
| value | 38.266 |

这是因为同一模板连续出现，复用距离约为 1，GPU prefix cache 已能覆盖大部分复用，CPU offload 准入策略发挥空间有限。

相比之下，`reuse_long` 中同一模板间隔约 8 个请求再次出现，GPU cache 容易被其他模板挤出，CPU offload 的价值变高。此时 Native TTFT 为 83.087 ms，Binary 降至 42.922 ms，Value 降至 53.631 ms。该结果说明 SlotOffload 更适合 GPU cache 无法直接覆盖、但 CPU cache 仍能保留共享前缀的中长复用距离场景。

### 6.3 模型规模影响

表 2 汇总 zipf core 实验在 0.5B、3B 和 7B 上的结果。相比 smoke，core 实验包含 512 个请求和 3 个随机种子，因此更适合作为模型规模趋势的依据。随着模型规模增大，结构感知策略的 TTFT 收益更稳定。

**表 2：不同模型规模下 zipf core 结果**

| 模型 | 策略 | TTFT mean (ms) | TTFT vs native | STORE MiB | STORE vs native |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5B | native | 23.220 | 0.00% | 1989.94 | 0.00% |
| 0.5B | binary | 25.691 | -10.64% | 45.94 | +97.69% |
| 0.5B | value | 24.653 | -6.17% | 44.75 | +97.75% |
| 3B | native | 31.710 | 0.00% | 5969.81 | 0.00% |
| 3B | binary | 28.573 | +9.89% | 137.81 | +97.69% |
| 3B | value | 29.542 | +6.84% | 128.25 | +97.85% |
| 7B | native | 51.589 | 0.00% | 9286.38 | 0.00% |
| 7B | binary | 41.561 | +19.44% | 214.38 | +97.69% |
| 7B | value | 44.589 | +13.57% | 187.83 | +97.98% |

0.5B 中预填充计算较便宜，CPU offload 的传输和调度开销相对更明显，因此虽然写入量显著下降，TTFT 反而略有变差。3B 开始出现稳定正收益：Binary 和 Value 分别降低 9.89% 和 6.84% 平均 TTFT。7B 中收益进一步扩大，Binary 和 Value 分别降低 19.44% 和 13.57% 平均 TTFT。这支持本文的一个重要判断：结构感知 offload 对较大模型和较高 prefill 成本更有价值。

为避免只看 zipf 分布造成选择性解释，表 3 进一步给出 3B core 的完整结果。3B 在 uniform、zipf、reuse_long 中均呈现明显收益；在 reuse_short 中，所有策略的 LOAD 和 hit tokens 均为 0，说明 GPU prefix cache 已覆盖短复用距离场景，CPU offload 的主要收益空间有限。

**表 3：Qwen2.5-3B Agent/MCP core 实验结果**

| 分布 | 策略 | TTFT mean (ms) | TTFT vs native | STORE MiB | LOAD MiB | Hit tokens | Hit/Store |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| one_hit | native | 29.336 | +0.00% | 4375.69 | 5457.94 | 155248 | 35.5 |
| one_hit | binary | 28.438 | +3.06% | 137.81 | 5598.56 | 159248 | 1155.6 |
| one_hit | value | 28.762 | +1.96% | 98.62 | 5616.94 | 159771 | 1619.0 |
| reuse_long | native | 45.342 | +0.00% | 12924.00 | 0.00 | 0 | 0.0 |
| reuse_long | binary | 30.278 | +33.22% | 137.81 | 8682.19 | 246960 | 1792.0 |
| reuse_long | value | 32.964 | +27.30% | 120.38 | 7063.31 | 200912 | 1669.0 |
| reuse_short | native | 24.988 | +0.00% | 3885.00 | 0.00 | 0 | 0.0 |
| reuse_short | binary | 23.668 | +5.28% | 137.81 | 0.00 | 0 | 0.0 |
| reuse_short | value | 24.089 | +3.60% | 144.56 | 0.00 | 0 | 0.0 |
| uniform | native | 39.151 | +0.00% | 9508.88 | 3130.31 | 89040 | 9.4 |
| uniform | binary | 29.771 | +23.96% | 137.81 | 8402.25 | 238997 | 1734.3 |
| uniform | value | 31.744 | +18.92% | 131.62 | 7292.06 | 207419 | 1575.9 |
| zipf | native | 31.710 | +0.00% | 5969.81 | 4347.94 | 123675 | 20.7 |
| zipf | binary | 28.573 | +9.89% | 137.81 | 6018.19 | 171184 | 1242.2 |
| zipf | value | 29.542 | +6.84% | 128.25 | 5605.31 | 159440 | 1243.2 |

3B 结果也进一步说明 Binary 与 Value 的差异。Binary 在多数分布中延迟更低，例如 uniform 中降低 23.96% TTFT；Value 的写入量通常更少，例如 one_hit 中 STORE 从 Binary 的 137.81 MiB 降至 98.62 MiB。该趋势与 7B 一致，说明二者不是“谁全面优于谁”，而是面向不同优化目标的两种准入策略。

### 6.4 0.5B core 的边界结果

0.5B core 结果显示，SlotOffload 可以稳定减少写入，但 TTFT 不一定总是改善。在 one_hit、uniform 和 zipf 下，Value 相比 Native 的 STORE 减少约 97.60%--98.52%，但 TTFT 在 uniform 和 zipf 中分别下降 20.20% 和 6.17%。这说明对于小模型、短 prompt 和低并发场景，CPU--GPU 传输、调度开销和测量噪声可能抵消缓存收益。

本文不将该现象视为失败，而将其作为适用边界：SlotOffload 的主要收益来自减少写入放大和提高有效缓存密度；端到端 TTFT 收益依赖模型规模、缓存压力和访问模式。

### 6.5 讨论：Hit tokens 与写入效率

表 1 中部分场景的 hit tokens 高于 Native，而部分场景 hit tokens 为 0。前者说明在有限 CPU cache 下，过滤低价值写入可能减少缓存污染，使更多共享 instruction/schema 留在 CPU 中；后者则说明 TTFT 收益并不总是来自 CPU 命中，也可能来自减少 STORE 写入和调度开销。因此，本文不把 hit retention 作为单独的核心结论，而是结合 STORE MiB、LOAD MiB、HitTokens 和 TTFT 解释。

更稳健的指标是单位写入命中效率 \(E_{store}\)。例如在 uniform 中，Native 每 MiB STORE 仅带来约 6.0 个 hit token，而 Binary 和 Value 分别达到约 1114.8 和 1069.0；在 zipf 中，Native 为 13.3，Binary 和 Value 分别为 798.5 和 840.7。这表明 SlotOffload 并不是简单少写，而是显著提高了 CPU 写入的有效性。

## 7 相关工作

### 7.1 KV Cache 管理与 vLLM

vLLM 通过 PagedAttention 将 KV Cache 分页管理，减少显存碎片并提高服务吞吐，是本文实现的基础。现有工作还研究了 prefix caching、分离式 prefill/decode、跨实例 KV 传输和多级缓存等方向。SlotOffload 与这些工作正交：它不改变底层 KV 放置格式，而是在 CPU STORE 前增加准入策略。

### 7.2 语义或结构感知 KV 复用

KVShare、CacheBlend、CacheSlide 等工作关注语义相似、非连续片段或不变段/动态段的 KV 复用。本文与它们的差异在于：SlotOffload 不放宽 vLLM 的精确前缀匹配，不跨不同上下文融合 KV，也不对 KV 做近似；它只利用应用已知结构决定是否将已计算 KV 写入 CPU，因此实现更轻量，正确性边界更清楚。

### 7.3 分层存储与 SSD Offload

SolidAttention、SWARM 等工作探索 SSD 级 KV Cache offload，重点关注存储设备带宽、I/O 调度、多 SSD 协同和块放置。本文聚焦 GPU--CPU 两级路径，不使用 SSD。虽然硬件层级不同，但共同问题是：下层存储容量和带宽都有限，无差别写入会造成缓存污染。SlotOffload 的价值准入思想可作为未来 GPU--CPU--SSD 多级缓存的上游信号。

### 7.4 传统缓存准入

传统缓存系统常根据对象大小、访问频率、最近性和获取代价决定是否准入。SlotOffload 借鉴缓存准入思想，但引入 LLM prompt 的 token 结构信息和预填充重算成本，使策略能区分同一请求内部不同语义片段的 KV 价值。

## 8 局限性

**非 supercomputing 规模。** 本文是单机单 GPU LLM serving 优化，不涉及多 GPU、多节点、集群调度、资源隔离或跨节点 KV 传输。因此本文不适合作为 SC/HPC 系统论文投稿。更合适的定位是单机推理服务、私有化 Agent 部署或中文 CCF-C/工程系统方向小论文。

**依赖结构元数据。** SlotOffload 需要应用侧提供 prompt 片段类型。对于 Agent、模板平台和表单类业务，这类信息通常天然存在；对于完全开放聊天，可能需要额外解析。

**仍受精确前缀限制。** SlotOffload 不支持“前缀部分不同但后续相同”的 KV 复用。若 slot 出现在 prompt 前部，后续 schema 即使相同也不能直接命中。因此，推荐将稳定 instruction/schema 放在动态 slot 前面。

**Value 策略未全面优于 Binary。** 实验显示 Binary 在 TTFT 上经常更优，Value 在写入量上更优。这说明当前 value 模型偏保守。未来可在线校准 prefill cost 和带宽，或根据业务目标在延迟优先与写入优先之间自适应切换。

**合成 workload 与真实流量差异。** 本文 workload 是可控合成数据，用于系统分析模板频率和复用距离。真实生产流量可能具有更复杂的会话状态、工具调用链、多轮上下文和突发热点。后续应引入真实或半真实 trace，或者至少基于公开对话/工单数据映射模板 ID。

**max_tokens=1 的人工性。** 当前核心实验使用 `max_tokens=1`，主要用于隔离 prefill 和 KV 复用影响。在真实 Agent 服务中，输出长度、工具调用轮数和 decode 阶段开销会稀释 TTFT 收益。最终投稿应补充 `max_tokens=16/64` 实验，说明长输出下收益边界。

**正确性实验待补。** SlotOffload 不改变 KV 内容和模型计算，理论上不影响输出；但由于准入决策会改变 external cache 可用性，仍需通过实验验证 fallback 重算路径稳定。最终投稿应补充固定 prompt、temperature=0 下 native/binary/value 的输出一致性检查，以及 reset/fallback 路径测试。

## 9 结论

本文提出 SlotOffload，一种面向单机 Agent 结构化提示词服务的 KV Cache CPU offload 准入机制。SlotOffload 利用应用侧 prompt 结构元数据，将 KV block 的保存决策从“是否可 offload”推进到“是否值得 offload”。在 vLLM 原型中，SlotOffload 只过滤 CPU STORE，不改变 attention、KV 内容或前缀匹配语义。

实验表明，在 Qwen2.5-7B Agent/MCP workload 上，结构感知策略可以减少 96.37%--99.09% CPU KV 写入，并在多数分布下降低平均 TTFT；其中 Binary 策略延迟收益最高，Value 策略写入量最低。模型规模实验进一步显示，随着模型变大，结构感知 KV offload 的延迟收益更明显。本文结果说明，Agent 服务中的 prompt 结构是 KV Cache 分层管理的重要信号；即使采用轻量级 STORE admission，也能明显改善原生 offload 的写入效率，并在合适模型规模和复用距离下带来 TTFT 收益。

## 参考文献

> 以下参考文献条目需要在投稿前按目标会议格式核对作者、会议、年份和页码。

[1] Kwon W, Li Z, Zhuang S, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP, 2023.  
[2] vLLM Project. vLLM: Easy, Fast, and Cheap LLM Serving. https://github.com/vllm-project/vllm.  
[3] Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.  
[4] KVShare: Semantic-Aware Key-Value Cache Sharing for Efficient LLM Serving.  
[5] CacheSlide: Efficient KV Cache Reuse for Prompts with Static and Dynamic Segments.  
[6] CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion.  
[7] SolidAttention: Low-Latency SSD-based Serving for Large Language Models.  
[8] SWARM: Co-Activation Aware KVCache Offloading Across Multiple SSDs.  
[9] LMCache: An Efficient KV Cache Layer for LLM Serving.

## 附录 A：实验命令

7B core 实验命令：

```bash
export VLLM_USE_MODELSCOPE=True
export MODEL="/root/autodl-tmp/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct"
export GPU_KV_BYTES=58720256
export CPU_BYTES_OVERRIDE=234881024
export MAX_MODEL_LEN=1024
export MAX_TOKENS=1
export PROFILE=agent_core
export RESULT_ROOT=slot_agent_core_qwen7b_local_len1024
export RESUME=1

bash examples/features/run_slot_offload_agent_experiments.sh start
```

查询进度：

```bash
export RESULT_ROOT=slot_agent_core_qwen7b_local_len1024
bash examples/features/run_slot_offload_agent_experiments.sh status
```

查看汇总：

```bash
export RESULT_ROOT=slot_agent_core_qwen7b_local_len1024
bash examples/features/run_slot_offload_agent_experiments.sh summary
```

## 附录 B：待补实验

1. 长输出实验：补充 `max_tokens=16/64` 下的 TTFT 和 E2E latency，回应 `max_tokens=1` 过于人工的问题。
2. 输出一致性：同一 prompt 在 native、binary 与 value 下输出 token 是否一致。
3. 策略开销：决策耗时、metadata 大小、hotness tracker 内存。
4. 容量压力实验：改变 CPU/GPU KV 容量，验证 cache pressure 项。
5. 参数敏感性：`min_accesses`、`max_slot_ratio`、`base_threshold`、`pressure_scale`。
