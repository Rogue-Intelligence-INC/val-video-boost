# 原版 Fate 设计规范（移植自 rogue Valhalla 原版，2026-11-14）

这是所有者的原始设计，从原版仓 `C:\Ai\rogue-products\Valhalla` 的
代码与试验记录移植而来。它优先于 server 侧一切后加的补救措施
（hybrid 弱放行、词法轨道、transformer 复述重抓、网关表示层补丁）——
那些是移植期产物，与本规范冲突时以本规范为准。

## 一条链、一种货币

```
信号入 → Fate 被动收 (receive_passive) → 收敛收口长出 Tile/Stem
      → 集束成区块 (bundle_fits) → 物理场演化 (场强=能量·√复杂度)
      → 刺激流过身体、区块被激活 → Fate 方程判抓取 (场强货币)
      → react 只读出口 → 反馈回场 (判后学)
```

## 五条不可动摇的规则（原版代码出处）

1. **被动收**：信号只流入，Fate 每 10 拍用自身簇方程判边际递减，
   收敛才收口；系统从不"去找"。
   `systems/metasystems/fate-f64/src/triple_fate.rs::receive_passive`
2. **判后学**：adapt 只吃方程自己刚判过的比率——先判定、后学习，
   方程不被自己的输出反向喂养。自适应发生在判定之后，不在判定里。
3. **集束是 Fate 的功能**（用户裁定 2026-11-03，提交 373a9dd）：
   邻居距离拟合者同束（`bundle_fits`：比率近 1、(x−1)² < 方程自身
   threshold），每束一条神经；束仅作压缩计数。
   `stem-cell/src/stemcell_complete.rs::rewrite_connections`
4. **抓取只认场强**（用户裁定 2026-11-03，提交 9e05869）：
   Fate 对外抓取不考虑束的汇总，只认所连信息的场强度——目标取
   束内场强最高成员，神经强度记该成员场强度。
5. **出口只读**：`react(&self)` 刺激不入 receive_passive、不 tick、
   不改方程不改存储；判不了就不送出（沉默是合法出口）。

## 被动抓取的正式定义（vent 探针，原版实测）

身体受刺激运转，各模块形成响应流；全局 Fate 用同一方程
`check_convergence` 逐条判：过判全拿、不封顶、不排名。
静默臂同规则抓取报噪声底。筛选是下一阶段的事，出口不查找不匹配。
记录：`manifestsys/hub-f64/src/bin/valhalla_probes.rs`（vent5/6/7），
产物存档 `C:\Ai\valhalla-bodies\grasp_vent{5,6}.json`。

## 与 server 现行实现的已知差距（移植时逐项消除）

- server 抓取是全量余弦扫描排名（主动检索），原版是被动激活；
- server 的门读全局 margin + ambient 代数，原版读方程收敛 vs 噪声底；
- 集束/场强只躺在 Stem 连接重写里，未接 QA 抓取；
- 身体存档（valhalla-bodies，2026-11-01）早于集束实现，加载后需
  按现行代码续跑重写连接才成区块形态。

## 本仓的移植件

- `rust/fate_qa.rs` + `rust/engine/`：2026-11-14 在 Valhalla-main
  hub-f64 落地的 Fate 抓取出口（candidate 序列 + 激活区块暴露、
  seg 编号连续、fragment 窗口卸载），见 `rust/README.md`；
- `docs/FATE_SILENCE_AND_UNLOAD_ANALYSIS_20261114.md`：过度沉默与
  动态卸载的实测分析（移植决策依据）；
- `tests/`：全部探针与结果，数字可复跑核对。
