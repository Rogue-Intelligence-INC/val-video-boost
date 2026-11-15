# val-video-boost

> ⚠️ **非开源 · NOT OPEN SOURCE** — 专有资产，见 [LICENSE](./LICENSE)

> **最新裁定（2026-11-15 晚，见 [docs/SUMMARY_REPORT.md](./docs/SUMMARY_REPORT.md)）：**
> 去插件拆解后，身份栈已收敛为无插件形态并巩固——圣经逐字钉死供给
> （detailed 身份块 + 否定事实负向）→ 每段 4 候选 → `FateInterface`
> 按指定 hero 脸 `select/verify`（τ=0.5 弃权）。实测 ArcFace 均值 0.657、
> 最低 0.603、6/6 过门，不用 IP-Adapter、不用 Triad 补丁（补丁通道已实测
> 只锁属性不锁人，退出身份路由）。外来指定真脸仍需参考通道，边界已实测写明。
> 回归：`code/test_fate_consolidation.py` 七项全过。下文为早期轮次记录，保留存档。

**定位（2026-11-15 改造）：自适应模型框架** — 不绑定任何单个生成模型。
框架自动发现本机全部生成模型、逐个测绘架构（区块/挂载点）、按探针
阶梯自适应定补丁强度（护栏内按目标实测选优，机制源自
valhalla-transformer / val-videopic-fusion 的 adaptive patch），并按
本项目实测记录路由任务。接新模型只需要：被发现 → profile → 跑一轮
探针，即纳入框架。入口：`code/adaptive_framework.py`（discover /
profile / plan / calibrate / route），实测：`code/test_adaptive_framework.py`
（已发现 7 个模型并分类，Juggernaut 探针选出强度 0.03，FLUX 57 块以
先验强度 0.0166 出图正常，5 条任务路由齐备）。硬限制以数据存档
（如 CogVideoX 显存墙 512×288、FLUX 必须 group offload），不写死在代码里。

**Valhalla 全套（LLM-BOOST 智能层 + Fate Triad 补丁）→ 生视频模型，超长上下文人物一致性实测**

新建于 2026-11-15，拷贝自本地两项目再融合：
`VAL-LLM-BOOST-FRAMEWORK`（Fate 之上的迭代/图/聚合/冲突/验证/适应智能层）×
`val-videopic-fusion`（Hub/Tile/StemCell/Triad 补丁注入 SDXL/CogVideoX 内部模块）。

## 直接回答：超长上下文能做到人物一致吗？

**分层能，不能一句话吹满**（2026-11-15 本机 CogVideoX-2b 实测）：

| 层 | 结果 |
|---|---|
| 超长上下文承载 | ✅ 20,033 行 / ~21.8 万词圣经放在视频模型之外（其 T5 上限仅 ~226 token），每段抓取可审计行号 → 39–44 词提示词；身份是恒定 256 维钉死补丁，与上下文长度无关（LLM 版已测 1495 万 token 同理） |
| 段内时序一致 | ✅ 相邻帧差 3.00 → 1.24（**-58%**），闪烁下降，不再逐帧变脸 |
| 属性一致（发色/年龄/服装） | ✅ hero 段 baseline 黑发翻车，补丁臂正确栗色发；与生图旧测同向（脸 L2 50.8→44.6） |
| 跨段同一个人（像素身份） | ❌ 两臂跨场景都是"同属性的另一个人"（补丁 114.8 vs 基线 92.9，未变紧）；256 维方向向量装不下人脸身份 |
| 漂移纪律 | ✅ YuNet 验证门：不过门弃权+留证不伪造；实测强度 ×2 毁脸，阶梯已改为降档 |

**第三轮 ✅ 机制补齐后极限通过**：装 insightface/ArcFace 后发现像素 L2 排名是假的
（真身份：基线 0.471 > 各补丁臂 0.417–0.433，异人 0.152）——纯机制问题。
接上本机 IP-Adapter Plus Face 参考脸通道（锚=已验证 hero 脸，Fate 钉死不更新）：
跨段 ArcFace **0.582**（单张最低 0.528）。极限测验：单锚点、10 万行圣经、
12 种光线/服装/表情 × 2 种子 = 24 段连续链——**均值 0.5516、最低 0.508、
24/24 ≥ 0.5、零重生、后半段与前半段同分（锚不衰减），EXTREME_PASS**。
即：身份信息以足够带宽进入每一帧且锚不被覆盖时，段数与上下文长度都不再是衰减项。

第二轮（Juggernaut XL 大生图模型 × Fate 记忆迭代）：基线跨段 47.2（CogVideoX 为 92.9，
模型规模是最大杠杆）；三版迭代——会话持久（已闭合片段冻结、新片段 0、补丁零漂移）、
累积语料重摄入（补丁 cos 0.989 真的动了）、身份强化回写——全部落在 44.7–45.3，
均不敌**冻结的 static_triad 42.93**。原因：文本回写不含人脸新信息，会话向量倾斜只是
噪声。下一步迭代必须回写图像特征（人脸嵌入编码器替换现 ResNet 级 dl_front）。

真·人物锁还缺参考图条件器（I2V / IP-Adapter-Face，本机无 CogVideoX 侧权重）：
补丁管属性与时序、hero 帧作 I2V 锚管像素身份、验证门管漂移——三件齐了才成立。
详见 [docs/DESIGN.md](./docs/DESIGN.md)。

![contact sheet](./reports/long_video_character/contact_try1.png)
（上排 baseline、下排 identity_lock；同种子同提示词，左→右 studio/park/harbour）

## 布局

```
code/     module_patch.py / bridge.py / download.py（拷贝自 val-videopic-fusion）
          character_bible.py  圣经+图边+冲突+抓取+身份补丁导出（新）
          video_boost.py      验证门/阶梯/账本 = BOOST 算子视频版（新）
          test_long_video_character.py  超长上下文人物一致性实测（新）
          test_*.py           拷贝的旧测试线
docs/     DESIGN.md（全套映射与裁定） + boost/（LLM-BOOST 原分析文档）
data/     identity_patch.json（身份 Triad 导出） + patches/（旧四体补丁）
reports/  long_video_character/（本轮视频/报告/对照图） + prior_*（旧实测存档）
```

## 安装与运行（公开发布准备版）

```powershell
.\install.ps1            # 建 venv → CUDA torch → 依赖 → doctor 自检（须全绿）
cd code
..\.venv\Scripts\python.exe doctor.py
# 无插件身份链一条命令：路由 → hero 登记 → 每段 K 候选 → Fate 选择 → τ 门 → manifest
..\.venv\Scripts\python.exe run_pipeline.py
# API 服务（可选 FATE_API_KEY 鉴权）：health/route/supply/enroll/verify/select/run
..\.venv\Scripts\python.exe fate_service.py     # http://127.0.0.1:8790/health
```

发布门槛与未达标清单见 [docs/LAUNCH.md](./docs/LAUNCH.md)（未清零前只做邀请制可用）。

## 运行（巩固后的正式入口）

```powershell
cd code
C:\Ai\models\.venv-sd\Scripts\python.exe run_pipeline.py
C:\Ai\models\.venv-sd\Scripts\python.exe run_pipeline.py --scenes park,harbour,library --seeds 7,8 --k 4
C:\Ai\models\.venv-sd\Scripts\python.exe run_pipeline.py --hero <同源hero图>   # 指定已有锚脸
# 回归（16 项）与证据对表
C:\Ai\models\.venv-sd\Scripts\python.exe test_fate_consolidation.py
```

产物：`reports/pipeline_run/manifest.json`（每段全候选分数、指纹、
选中帧与 ACCEPTED/ABSTAINED 状态）+ `contact_sheet.png`。
退出码 0 = 无弃权段。产品介绍见 [docs/PRODUCT_INTRO.md](./docs/PRODUCT_INTRO.md)，
全实验总结见 [docs/EXPERIMENT_SUMMARY.md](./docs/EXPERIMENT_SUMMARY.md)。

*Rogue Intelligence INC. · Proprietary*
