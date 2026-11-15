# VAL-LLM-BOOST-FRAMEWORK

Valhalla 之上的**智能层**：在 `val-llm-fusion-framework` 已实测的"可信记忆带宽"
（≥1495 万 token 规模不变、零编造、抗投毒、可审计弃权）之上，补齐当前缺失的
推理机制——迭代检索、关系图遍历、确定性聚合算子、冲突/时序通道、答案验证器、
离线适应闭环。

立项依据（全部本机实测，见 `docs/INTELLIGENCE_GAP_ANALYSIS_20261114.md`）：

- 当前天花板三处：多跳 0/2、计数/列举 0/2（集束吃枚举）、矛盾静默选后写；
- 根因不在"结构不够强"，而在流水线只有一趟、无关系语义、门只判接地不判
  推理完整、派生事实不回写、运行时权重不学习；
- 内部先例：unified-core `multihop_fate/v2` 切片 15/25、MCQ 阶梯证明智力
  随解码头缩放——本框架负责把这些窄机制做成通用智能层。

硬约束（不可回归）：极端三档 12/12、零编造、特征抓取 9/9、LoCoMo 不虚报。

2026-11-14 晚追加移植（原版优先）：

- `docs/ORIGINAL_FATE_DESIGN.md`——所有者原版 Fate 设计规范
  （被动收/判后学/集束/场强抓取/只读出口），从 rogue Valhalla
  原版代码与 2026-11-03 裁定记录移植；与 server 补救措施冲突时
  以原版为准；
- `docs/FATE_SILENCE_AND_UNLOAD_ANALYSIS_20261114.md`——过度沉默
  根源、判据离线校准、动态卸载 A/B（46MB→1.49MB）实测；
- `rust/`——已在 Valhalla-main 落地的 Fate 抓取出口与卸载修复
  源码副本及集成说明；`tests/` 为全部探针与结果。

状态：2026-11-15 已推送，组织归属：
Rogue-Intelligence-INC/VAL-LLM-BOOST-FRAMEWORK。`rust/` 引擎副本已
同步至 Valhalla-main 现行源码（2026-11-15）；一键复现见根目录
`reproduce.ps1`（构建引擎 → 起本地引擎 → 跑 560 行电池）。
