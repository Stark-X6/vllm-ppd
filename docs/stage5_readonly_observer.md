# 阶段 5 只读状态观测：实现与验收边界

更新日期：2026-10-23。状态：默认关闭的节点观测代码、API 接线、独立 pending 账本及 CPU 校验已完成；原生 EngineCore/IPC/GPU 联合验收未完成。新查表、AMPD gate 和 TTL 均未启用。

本文件是 [语义契约 v1](./stage5_state_semantics_contract.md) 的实现记录，不将 CPU 假对象测试称为实际 serving 验收。

## 1. 已实现内容

| 部分 | 实现 |
|---|---|
| 节点观察器 | [ppd_observer.py](/root/vllm-ppd/vllm-source/vllm/v1/core/ppd_observer.py)：owner-thread 快照、有效前缀、decode/prefill/blocked 分类和覆盖标志 |
| 只读 API | [api_router.py](/root/vllm-ppd/vllm-source/vllm/entrypoints/serve/ppd/api_router.py)：POST /v1/ppd/observe，经 AsyncLLM → EngineCore utility，不调用 GPU worker collective |
| 就绪记录 | [block_pool.py](/root/vllm-ppd/vllm-source/vllm/v1/core/block_pool.py) 与 [core.py](/root/vllm-ppd/vllm-source/vllm/v1/engine/core.py) 的可选 hook，只更新 observer 私有元数据 |
| Pending 账本 | [observation_pending.py](/root/vllm-ppd/ppd/optimizer/observation_pending.py)：单节点/epoch、物理 attempt 去重、ACK/tombstone、乱序/未知状态处理 |
| 开销脚本 | [benchmark_ppd_observer_cpu.py](/root/vllm-ppd/scripts/benchmark/benchmark_ppd_observer_cpu.py)：CPU 假负载与交替空操作对照，JSON 输出 |

开关为 `VLLM_PPD_OBSERVER_ENABLED=1`，默认不创建观察器、不注册接口。关闭时只留下少量条件判断；没有将新特征接进原决策器，Agent、server config、routing map 和原 decision engine 的文件指纹保持语义冻结时的值。用户原有 decision engine 空白修改原样保留。

## 2. 前缀就绪性与只读保证

缓存 hash 可以在 GPU forward 前登记；同一调度轮中被取消/抢占的请求可能没有真正完成对应写入。因此本实现维护独立证明：

1. 新 hash 登记时，将对应物理 block 标为未证实。
2. 同步 forward 完成后，只证明该次实际 scheduled、同一 preemption generation 且计算边界覆盖到的完整 blocks。
3. 回收/复用 block 时清除旧证明，即使后来写入相同 hash 也不能继承旧证明。
4. Cache reset 或 invalid KV 使旧证明失效；无法证明的命中返回 C/U=null，而不是零或乐观命中。
5. Abort 后已完成的计算可以保留证明；未执行的登记不会因 free 而自动变成 ready。

证明记录不增加 KV 引用计数，不 touch LRU，不分配/释放 KV，不调用带命中统计副作用的 get_computed_blocks。查询只读原索引，使用引擎原生 block hasher 构造 token-only 临时视图；未知或不完整的 hash 来源不会被误判为冷缓存。Hook 异常只使观察器失效，不向正常 serving 抛出新异常。

已有请求的 `skip_reading_prefix_cache` 被遵守。API token probe 则明确假定普通生成允许 prefix read，不适用于 prompt-logprobs、LoRA 或多模态变体。

## 3. 支持范围与返回字段

当前支持单 engine、TP=PP=DP=DCP=PCP=1、单 full-attention KV group、同步无 batch queue、非 speculative、无 LoRA/多模态；connector 为无 connector 或当前 P2pNcclConnector。不支持的布局返回 unavailable，不修改调度配置以迁就观察器。

返回 node_id、node_epoch、snapshot_seq、scheduler_step、cache_generation、monotonic clock scope 和采样时刻：

- `coherent_snapshot`：是否取得同步 owner-thread 边界。
- `coverage_complete/available`：节点请求是否都能按支持范围分类；未知成员使聚合 B/K、Q 为 null，避免返回伪完整部分和。
- `prefix.available`：独立表示本次 probe 的 C/U 是否有效；节点状态可用不代表 probe 一定命中或可用。
- `decode.b_d/k_d`、`prefill.q_tok/q_att`、blocked/unknown 数及各请求的 ready/scheduled/target 进度。
- `request_presence`：对指定原生 engine request ID 返回 accepted/terminal/unknown。缺席不是 terminal ACK。
- `observer_elapsed_s` 与累计 hook 时间：仅 observer CPU 工作，不包括 API/IPC 排队，更不是 GPU 延迟。

Prefill 最大 chunk 放不下、但更小 chunk 能放下时，仍保留其已知积压，并记 `capacity_risk=chunk_size_dependent`。这是容量风险提示，不保证实际 scheduler 会立即接纳。Decode 需求与实际 scheduled 不混称；同步查询时没有未完成模型 batch，in-flight 测试验证的是拒绝不安全状态和账本去重，未声称支持 async 调度。

## 4. API 使用和限制

开启时需要在原来的 server 启动环境中额外设置上述开关，沿用原 Agent、P:D、map 与模型参数。本次未运行启动命令。

状态查询示例：

```bash
curl --noproxy '*' -sS http://127.0.0.1:8200/v1/ppd/observe \
  -H 'Content-Type: application/json' -d '{}'
```

可选请求字段只有：

```json
{
  "prompt_token_ids": [1, 2, 3],
  "cache_salt": "optional-existing-salt",
  "expected_node_epoch": "epoch-returned-by-node",
  "request_ids": ["native-engine-request-id"]
}
```

上面只是字段形状示例，不代表真实模型 prompt。提交 IDs 的来源记为 caller_supplied_token_ids，`canonical_chat_verified=false`；不自动认定它等价于下一条 chat 请求。API 不返回 prompt 文本或 IDs。精确 N/H/A 的 chat 渲染对齐与实际请求关联仍需后续集成，不能由这条 token-only API 宣称完成。

接口继承 /v1 API-key 鉴权；只有已配置 key 才有鉴权，没有 key 时会告警。应限制在可信网络，开启鉴权后沿用既有 Bearer。关闭时 404；默认限制 2 MiB body、262144/model-length token cap、128 个跟踪 IDs。API 每进程 single-flight、最短 0.1 秒；HTTP 5 秒超时/取消后，未完成 utility 仍占用该槽，避免重复超时无限堆积。EngineCore 另在 hash 前频控，并限制 512 个 active records、32768 次 block lookup。

单进程 API 限流不是全节点服务质量保证。观察器查询在 owner thread 执行，也会占用 CPU 时间；10 Hz 上限不是推荐采样频率。初次 GPU 验证建议低频 state-only 和少量定向 prefix probe，不对每个长 prompt 高频轮询。

## 5. Pending 与 TTL 接口边界

PendingLedger 已通过独立 CPU 校验，但**还没有挂到 production proxy 的派发链路**。EngineCore 导出原生 request ID；接入方仍需核实 X-Request-Id → 原生 ID 的对应，并带上 run/logical/attempt/destination/epoch。不能猜测字符串转换后就宣称端到端 pending 去重完成。

账本的规则是 engine 观察优先；accepted 从 active snapshot 消失不会重新变 pending，而是 unknown，直到有终止 ACK 或新的有效证据。Timeout/disconnect 也不能推断任务结束。终止墓碑不静默淘汰；默认 4096 条上限耗尽后返回 unavailable，明确的新节点 epoch 才清除旧账。

Engine observer 也保守保留 accepted/terminal 身份，容量耗尽后失效，不通过丢记录伪造完整性。正常长时间运行需要后续可靠 ACK 水位压缩，当前版本不适合无限时长测试。

TTL 只返回 disabled 能力元数据，没有 pin、renew、evict 动作。节点就绪证明和 pending ledger 都不保护缓存驻留。

## 6. CPU 验收与开销记录

测试：新 observer 47 项、mock API 19 项、pending ledger 22 项、前序语义 46 项和 calibration 契约 4 项，共 138 项通过。API 测试使用真实 FastAPI/httpx ASGI，但 engine/mock/auth 方法通过隔离加载核验，不是原生 vLLM 进程启动测试；observer 测试使用明确的假 scheduler/cache 对象。

开销以 21 轮交替对照测量，无并发测试进程。查询 baseline 是最小 dispatcher stub；hash 是测试用 chained SHA256 替代品；不包含 native hasher、tokenizer、HTTP、IPC 或 GPU 工作。表中为中位数，单次进程内重复不是 21 次独立 serving 实验。

| 活跃请求数 | Probe tokens | State-only/ms | State+probe/ms | 64-token chunk 记账增量 | 8192-token chunk 记账增量 |
|---:|---:|---:|---:|---:|---:|
| 16 | 4097 | 0.060 | 1.716 | 31.1 μs | 1.337 ms |
| 64 | 32769 | 0.185 | 13.744 | 78.2 μs | 1.365 ms |
| 256 | 131073 | 0.736 | 106.783 | 236.5 μs | 1.640 ms |

8192-token hook 场景含 512 个 blocks 的登记与回收及 batch 生命周期，不能解释为每个 decode step 都有这项代价。不同数据行同时改变请求数和 probe 长度，不能由此分离各因素的独立影响。32K/128K probe 的 p95 约为 68.9 / 161.4 ms，宿主机抖动明显；不能据此声称线上开销低于某个百分比。

完整数据、命令与源文件指纹见 [CPU 原始结果](./measurements/ppd_observer_cpu_20261023.json)。配对与重复采样边界参考 experimental-design 技能；方法工具来源为 Timothy Kassis、Vinayak Agarwal、Yuhuan He、Darshil Patel、Aubrey M. Brueckner（2026），[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。该引用仅说明工作流程来源，不是 serving 性能证据。

## 7. 开卡后仍需验收

需要 GPU 后再完成以下工作，不自动开始完整 Agent/calibration 实验：

1. 用相同模型和启动参数验证原生 API → IPC → EngineCore utility 能运行，确认实际调度模式在支持范围。
2. 受控冷/热前缀、尾 block、重置和取消请求：重复 probe 不改变命中统计、引用或回收行为，probe 与后续真实请求的复用量一致或有可解释的时序差异。
3. 小规模 decode+append 混合负载，核对实际 B/K、prefill/recompute、容量风险和 NCCL 接收未就绪的分类。
4. 核实物理 request ID 映射，再接入 proxy pending 对账；真实 timeout、取消和迟到 ACK 不重复收费、不复活已结束任务。
5. 同负载下 observer off/on 配对测完整响应、ITL、吞吐及 CPU/IPC 开销。先测低频 state-only，再测定向长 prefix 查询，评估是否需要复用已生成的 hash、降低频率或缓存查询结果。

真实开销和只读性确认前，不能宣布第二步全部通过，也不进入新 lookup calibration 或更改生产选路。

