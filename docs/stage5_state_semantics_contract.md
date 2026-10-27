# 阶段 5 输入与结果语义契约 v1

冻结日期：2026-10-22。版本：`ppd-state-semantics-v1`。

本文件完成第一步“冻结输入与结果语义”。它定义字段、采样边界与结果群体，并提供独立 CPU 参考计算；**不代表只读状态采集已接通，也不改变 Agent、路由、调度或缓存行为。** 不宣布 TTFT 优先，最终 Agent 主指标和 SLO 阈值仍未冻结。

后续实现状态（2026-10-23）：默认关闭的节点观察器、API 和独立 pending 账本已完成 CPU 验证，原生 GPU/IPC 与 proxy 对账待验收；见 [只读观测实现](./stage5_readonly_observer.md)。本文件的 v1 字段语义不因新增实现而自动改变。

纯计算实现：[state_semantics.py](/root/vllm-ppd/ppd/optimizer/state_semantics.py)。原运行契约继续见 [agent_serving_contract.md](./agent_serving_contract.md)；二者版本分离，旧 trace 不自动升级。

## 1. 不变项与命名规则

本阶段固定当前 Agent policy、reasoning 回放、工具 schema 与生成预算，保持 2P:2D、固定候选 P/D 和 strict routing map。不开 AMPD 前置门控，不实现 TTL pin/unpin，不重新校准、不启动 GPU 实验。

| 对象 | 读取时值或指纹 |
|---|---|
| Agent 配置 | Qwen3-30B-A3B-Thinking-2507；replay_reasoning=true；temperature=0；context=155648；episode completion budget=131072；per-turn cap=16384 |
| 服务默认配置 | BF16；MAX_MODEL_LEN=155648；GPU_MEMORY_UTILIZATION=0.97；端口 P 8100/8101、D 8200/8201 |
| uni-agent HEAD | `449f80f4383b1acf446bf5b7a6e2afce56093813` |
| vllm-ppd HEAD | `11dccb5d64d90910af42d9f46e06cc4e431c01dc`；开始时 decision_engine 已有用户未提交的空白修改，原样保留 |
| Agent YAML SHA-256 | `31dfd8485e13a1fc239b6ac34e925821d1947c25c9ea428bc3889ffc6b6f0d2d` |
| server/config.sh SHA-256 | `d05a9adb5a47222508246a311a5609b574eccee4625c7395dc9f8e329b372c3d` |
| first16 map SHA-256 | `0d2f3a9d50231b114bd1f963ccf0f1264b673c224804af1cbac154d8eaef459c` |

上述 SHA-256 是文件字节指纹，不冒充 `/status` 的规范化 map fingerprint，也不证明当前有相同参数的 server 正在运行。文件分别是 [Agent YAML](/root/uni-agent/examples/inference/task_config_ppd.yaml)、[server config](/root/vllm-ppd/scripts/server/config.sh)、[first16 map](/root/vllm-ppd/configs/routing/swebench_verified_first16_balanced_2x2.json)。本阶段不改变 concurrency；未来 matched run 仍须记录并匹配该参数。

所有计数为非负整数，时长以秒为规范单位；旧 benchmark 的毫秒字段只在显式转换时乘除 1000。`null` 表示未观测、不适用或不能证实；同时记录原因。零只表示确实观测到零，不用零替代未知。

## 2. 决策时刻、身份与覆盖

未来每份状态必须带 `semantics_version`、`node_id`、`node_epoch`、`snapshot_seq`、采样时刻、时钟类型和覆盖/有效性标志。输入只能来自本次首次路由决定之前的可用信息；本轮最终 completion、response cached_tokens 和事后负载峰值不能回填成决策前状态。

物理工作标识使用 `(run_id, logical_request_id, attempt_id, destination_node, node_epoch)` 或可无损对应的 `work_id`。P/D 阶段分开，retry 也分开。不得仅按 episode 或 logical_request_id 合并真实执行工作。参考函数处理单节点、单 epoch 的已规范化集合，不负责建立这些映射。

结果中的 `attempt` 是同一 logical request 跨 transport retry、required→auto phase 连续递增的 physical ordinal，优先采用 proxy 的 `logical_physical_attempt`。它不是 Agent 的 `X-PPD-Request-Attempt`：该 header 在新 phase 的 HTTP 调用中可能从 1 重新计数。Required 首次失败后 auto 成功，仍是 attempt 1 失败与 attempt 2 成功，不能有两个首 attempt。[proxy 计数](/root/vllm-ppd/ppd/comprehensive_proxy.py:1205)

CPU monotonic 时刻只在明确同源的时钟范围内相减。跨进程或跨节点的绝对时间不默认可比较；可以合并来源明确的时长，但需标明遗漏阶段。尚未定义 freshness 的数值阈值，本步骤不擅自新增路由超时策略。

## 3. N/H/A：完整输入与逻辑分界

| 量 | 冻结定义 | 不可替代来源 |
|---|---|---|
| N | 实际服务端规范化、tools schema、reasoning 策略和 chat template 生成的完整 prompt token IDs 长度 | 字符数、独立 suffix 长度、估算 context |
| H | 本次完整序列中可证明的历史/setup 边界位置 | 上一轮长度或跨轮 token LCP |
| A | 已证实 H 时的 N−H | 单独对 observation 使用模板得到的长度 |

消息边界沿用当前意图：有 assistant 时在最后一条 assistant 后切分；没有 assistant 时在最后一条 user 前切分，之前的 system/setup 属于历史侧。首轮的 H 因而可以非零；这不代表节点已有缓存。缺少可识别边界时，H/A 为 null。

在 token 层，只有“同一有效模板上下文中得到的历史侧 IDs”严格等于完整 IDs 的前缀，才可将其长度记为 H。最后的 generation prompt 以及分界后模板标记归 A。独立渲染可能改变历史 reasoning 或边界 token；此时 H/A=null，N 仍可有效。不能截取两者 LCP 后把它改名为 H。

`split_prompt_ids()` 只验证调用方提供的候选历史前缀，不实现 tokenizer 或模板对齐。空的已确认历史前缀给 H=0/A=N；未提供历史前缀给 H/A=null，两者不同。

现有 proxy 的 `context_tokens` 仍是字符估算，suffix tokenizer 的合成可加性检查也不是每个请求的前缀证明；旧 `input_tokens/context_tokens` 继续保留旧名，不能直接升级为本契约的精确字段。[旧提取代码](/root/vllm-ppd/ppd/comprehensive_proxy.py:480)

## 4. C/U：节点实际可以跳过的前缀

C_i 是同一完整 token 序列、模型/adapter、cache namespace/salt 和缓存组规则下，节点 i 当前 ready、连续匹配，并已考虑引擎 block/尾部限制后可跳过计算的前缀。U_i=N−C_i。它不是显存驻留 tokens 总数，也不是逻辑 H。

对本版本普通新生成请求，引擎将命中上限限制为 N−1，并可能因 block 对齐多重算一段，因此规范有效值为 `0≤C_i<N`、`U_i≥1`。物理前缀全驻留可以另记，但不能直接令 U=0。[前缀规则](/root/vllm-ppd/vllm-source/vllm/v1/core/kv_cache_manager.py:164)

没有观测、查询失败、epoch 不符、状态过时或完成状态不明时，C_i/U_i 都为 null；已确认冷缓存或已知关闭 prefix reuse 时，C_i=0 才合理。当前 P 无 D→P 读取，所以 C_P 不含“D 上有但 P 上没有”的 KV；C_D 也不把未来 Remote 路径将传来的 KV 当作决策时已有前缀。

hash 命中不自动证明 ready：本地 allocator 可以在 GPU 完成前登记计划计算的 blocks。未来 probe 需要 owner-thread 一致边界及完成状态证明，不能直接调用带命中统计副作用的 `get_computed_blocks()`。[提前登记位置](/root/vllm-ppd/vllm-source/vllm/v1/core/kv_cache_manager.py:315)

参考函数 `uncached_tokens()` 接受调用方已证实的有效 C，不负责查询，也不代替 block 对齐。C=N 的原始驻留长度会被拒绝，未知值原样保持未知。

## 5. B_D/K_D：需要 decode 服务的集合

v1 的精确参考范围限定为普通 n=1、非 speculative 生成，使用前一次完成结果已处理、下一次调度尚未开始的同步快照。若有无法解释的 async/batch-queue in-flight 状态，则 B_D/K_D 标 unavailable；不能通过 GPU 同步强行取得快照而改变 serving。

B_D 统计已有至少一个生成 token、尚未结束、KV 已恢复且当前可继续一步 decode 的请求。因本步 token budget 未被选中的 eligible 请求仍在这个需求集合中。Prefill/recompute、等待 FSM/远端 KV/容量的请求，以及工具中的 episode 不进入该集合；它们分别计入相应诊断。

L_j 是下一步 decode 输入之前，已有、可用的历史 KV 位置数。普通无 in-flight 情况下，已生成 G_j≥1 时：

```text
L_j = original_prompt_tokens_j + accepted_generated_tokens_j − 1
B_D = eligible decode 请求数
K_D = 对同一 eligible 集合求和 L_j
```

减一是因为最后一个 sampled token 将作为下一步输入，它的 KV 尚未产生。本步 attention 访问 L_j+1 个位置。K_D 不是所有 `request.num_tokens` 的直接和，也不是去重物理 KV 容量；不能由它推出精确带宽或延迟。

另记实际 scheduled decode 数、in-flight 数、blocked 数和各自 snapshot/step。`running` 含 chunked prefill，不能用 `len(running)` 或“本次只调度一个 token”判定 decode 阶段。[调度语义](/root/vllm-ppd/vllm-source/vllm/v1/core/sched/scheduler.py:228)

`summarize_decode()` 的输入是未来 collector 已正确识别的 eligible 集合，不自行从 engine 字段推断 phase。已完整观察的空集合输出 B_D=K_D=0；采集不可用时两者为 null；集合成员已知但个别 L_j 未知时，B_D 可有效，K_D 为 null，不输出未注明的部分和。

## 6. Prefill 积压：ready、scheduled、target 三个边界

每个本地 prefill/恢复工作定义：

```text
T_r：本次已知的处理目标
E_r：已完成并可使用的前缀（ready frontier）
S_r：已派发计算到的边界（scheduled frontier）
0 ≤ E_r ≤ S_r ≤ T_r

inflight = S_r − E_r
unscheduled = T_r − S_r
remaining = T_r − E_r = inflight + unscheduled
```

初次 prefill 的 T_r 是 prompt N；preemption 恢复可能包含已生成但丢失 KV 的 tokens，T_r 可大于原 N，但不包含未知未来输出。Preemption generation 改变后，旧 E/S 不能沿用。`num_computed_tokens` 会在 schedule 后增加，不能直接用作 E_ready。[调度进度](/root/vllm-ppd/vllm-source/vllm/v1/core/sched/scheduler.py:790)

集合 J_i 只包含节点 i 已可执行/正在执行的本地 prefill 或恢复，以及已知会在本节点计算的去重 pending。当前待选路请求排除；普通 decode、finished 和 blocked 不进入计算积压：

```text
R_r = T_r − E_r
Q_tok,i = Σ_(r∈J_i) R_r
Q_att,i = Σ_(r∈J_i) [E_r R_r + R_r(R_r+1)/2]
```

Q_tok 已包括 in-flight，不能再加一次。Q_att 是结构性 query–key 配对数，不是毫秒、FLOPs 或显存字节。某条计算工作的 E 未知时，整份 Q_tok/Q_att 标 null，同时报告 unknown_work_requests；不能把未知请求丢掉后声称得到完整积压。

等待远端 KV 的请求单列，即使连接器采用同步接收、没有进入 `WAITING_FOR_REMOTE_KVS` 枚举也一样。未来传输的历史既不算 E_ready，也不全算 D 本地 prefill；接收失败或 invalid blocks 后重新分类，接收完成后才计算实际本地 residual。[当前 P2P 外部匹配](/root/vllm-ppd/vllm-source/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:399)

同一物理工作同时出现在 engine 与 proxy pending 时只算一次，engine 状态优先，包括 blocked/finished。终止 ACK/tombstone 后不得因 active snapshot 不再出现而复活为 pending。集合缺失不等于完成 ACK；确认/水位线对账是下一步 collector 的职责。`summarize_backlog()` 假定单节点完整枚举；pending_requests 指最终纳入计算积压的 pending 数。

积压描述已接下的工作，不是严格排在新请求之前的 FIFO 队列。分块、合批和阻塞跳过都会改变等待，不在本步骤拟合等待时间。

## 7. 结果语义：三种 TTFT/TPOT 不混用

### 7.1 原始 score 的测量基础保持为 legacy client stream

`metric_basis=legacy_client_stream_text_v1`：

| 指标 | 冻结边界 |
|---|---|
| TTFT | client 调用 HTTP POST 前，到首个 SSE `choices[0].text` 非空 |
| E2E | 同一起点，到 HTTP stream 读取结束 |
| TPOT | (E2E−TTFT)/(G−1)，G>1；包括首文本后到 stream-end 的尾部开销 |

这是完整经过 proxy/P/D 的 client-stream 观测，不是 GPU 内第一枚生成 token 的时刻。原 writer 使用毫秒；参考函数以秒计算。保留现有 benchmark、loader 和 score 的行为，不在本步骤替换第一文本检测器。[旧实现](/root/vllm-ppd/scripts/benchmark/comprehensive_benchmark.py:433)

Chat 的 reasoning/tool 流可能有不同可见事件，不能因为也叫 TTFT 就直接并入上述 calibration。第一枚生成 token（含 reasoning）的 route-wide 语义可作为后续新测量基础，但需采集支持并单独命名，不静默替换旧 score 标签。

### 7.2 Node-local engine 指标只用于本节点诊断

`metric_basis=node_engine_generation_v1`：现有 `ttft_s` 是该节点 engine frontend arrival 到首次生成输出被 frontend 处理的时长，不是 GPU kernel 内精确结束时刻；`decode_elapsed_s` 是 engine 首/末生成输出事件间隔。普通单序列的 engine TPOT 为 `decode_elapsed_s/(G−1)`。[生成统计](/root/vllm-ppd/vllm-source/vllm/v1/metrics/stats.py:270)、[公开响应字段](/root/vllm-ppd/vllm-source/vllm/entrypoints/openai/serving_chat.py:83)

Remote 先执行 P，之后才发 D 请求，所以 D engine TTFT 不包含前面的 P 阶段，不能拿 D-only TTFT 给 Local/Remote 判端到端胜负。P 的 one-token 输出被丢弃，不是 Agent response 的首 token，也不计入 D 的 G。

当前非流式日志没有精确 route-wide 首生成 token 观测；HTTP first byte 通常接近完整响应就绪。`D HTTP 开始距 proxy 到达的间隔 + D engine TTFT` 只能命名为 estimate，它漏掉 D HTTP 发出到 D engine arrival 等阶段，不能升级成精确标签。不同来源的绝对时间不可直接相减。

### 7.3 ITL、request TPOT 与 poller 旧别名

Engine ITL 来自相邻输出事件间隔；一次事件有多个 tokens 时，它不是逐 token 间隔。节点 histogram 对事件加权，request TPOT 则先在请求内计算首末间隔/(G−1)，再按请求聚合，二者不同。

当前 poller 的 `tpot_sum/count` 读取旧名 `vllm:time_per_output_token_seconds`；该 fork 在此 histogram 写的是 ITL 事件，仅 hidden-metrics 开启时存在。真正每请求均值写入单独的 request-time-per-output-token histogram。旧字段保留兼容，但在新语义中应注明 `legacy_alias_of_itl`，不得作为 lookup 的 request TPOT。[poller](/root/vllm-ppd/scripts/profile_metrics_poller.py:33)、[ITL 别名写入](/root/vllm-ppd/vllm-source/vllm/v1/metrics/loggers.py:1125)、[request TPOT 写入](/root/vllm-ppd/vllm-source/vllm/v1/metrics/loggers.py:1162)

节点累计 histogram 可能包含 warmup、背景和失败前输出，不能代表匹配后的 Agent cohort。没有新 token 样本也不是零 ITL：应另记录 active decode 的 token age 和采样覆盖。

## 8. 统计群体、G、失败与 retry

第一版 score 的参考群体固定为 `candidate_turn2_first_attempt_v1`：同一 calibration 场景中，指定候选请求的逻辑 Turn 2、首 physical attempt、非 warmup。不混入 P 的 prefill-only 请求、Turn 1、背景 decode 或 retry。Online live 的 Turn 2+、全 attempts 和 episode 群体另报，不自动加入此均值。

`success` 是严格布尔值，表示该物理请求正常返回符合接口的完整响应，没有 HTTP/protocol/server 错误；不是工具语义成功或 SWE-bench resolved。正常返回的 `finish_reason=length` 仍计服务成功，并单列截断状态；如果截断触发服务器解析错误而返回 HTTP 400，则是失败。缺 usage 或缺可见首文本不一定改变服务成功标志，但对应 G/延迟必须单独判断有效性，不能据 success=True 填造测量值。

G 是该物理生成请求的真实 token IDs 数，包含 reasoning 和工具调用序列，不是可见 content 字数或 max_tokens。第一版限定 n=1；多 choice、speculative 或合并输出事件需额外契约。P 被丢弃的 token、其他 attempt 和此前轮次不能并入本次分母。

平均 TPOT 使用每个有效请求 TPOT 的算术平均，不用 `Σdecode_time/Σ(G−1)` 的 token 加权平均替换。每次汇总都保存 n_total、n_success、n_failed、n_ttft、n_tpot、n_missing_usage、n_zero_output。TPOT 纳入 score 均值时沿用旧代码的正值规则；G≤1 或缺 usage 时 TPOT=null，不能用零表示极快。G=1 仍可能有合法 TTFT；G=0 不存在首生成 token。

失败 attempt 留在分母和错误表，已有部分 timing/token 也保留，但不纳入成功请求 latency 均值。首 attempt 失败后的 retry 成功不能替代首 attempt，让对照表看起来没有失败。另报告全 attempts 消耗及 logical 完整完成时间；Agent 的整个 query 时间包含 retry/backoff，不能除以最终成功 response 的 G 当作 TPOT。

旧 writer 的无文本/缺 usage/失败可能以 0 写入，旧 TPOT 平均只取 >0，旧成功率还受 T1 失败后不发 T2 影响。这些 artifact 原样保留；新规范化记录采用 null+原因，不能没有证据地把旧 0 恢复为有效观测。Runner 还需单列 planned、attempted、T2 可达数，未产生的记录不能由 helper 猜测。

`summarize_calibration()` 只实现明确群体的参考聚合，并拒绝混合 measurement basis；它不替换 loader、不产生新路由，也不声称失败点可用作性能赢家。Node-engine 汇总只能作诊断，不能混入 legacy route score。

## 9. 验收边界与后续依赖

CPU 测试覆盖 token 前缀证明、null/0、有效 C 的尾部约束、decode 集合、E/S/T 不变量、pending 去重/终止和统计群体。通过只说明参考语义及算术一致，不能证明引擎已经能导出 ready C、B/K 或积压。

2026-10-22 验收：46 项新语义 unittest 全部通过，另有 4 项原 calibration filename/loader 契约 unittest 通过。额外计划的 decision metadata、routing 和 analyzer pytest 回归因当前 Python 未安装 pytest 而未执行，未为此安装依赖。Agent YAML、server config、first16 map 与原 decision engine 的文件指纹均保持开始时的值；原 decision engine 已有的用户空白修改未处理。

下一步才实现无副作用观测及映射：完成状态证明、collector 覆盖、跨进程身份对账与 clock provenance 都必须保留。分桶、freshness 阈值、表外策略和未来 calibration 点仍未实现；AMPD gate 与 TTL 仅保持后续接口边界。

运行 CPU 参考测试：

```bash
cd /root/vllm-ppd
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_state_semantics.py' -v
```

修改字段含义、时间边界或群体需要新版本。原 Agent/PPD baseline 与本契约应同时保存，不用重新命名旧字段掩盖不兼容。
