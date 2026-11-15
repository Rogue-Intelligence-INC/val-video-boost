# LAUNCH：公开发布门槛与运行手册

状态：准备中（2026-11-15 起）。本文件是发布清单，不是宣传稿：
每一项都要有实测或可复现证据才打勾，未达标项如实留空。

## 1. 已达标（有证据）

- [x] 核心机制实测：无插件一致性 0.657 / 24 段 0.630 / 48 段无衰减
      （`reports/fate_long_chain`、`reports/fate_pressure48`）
- [x] 正式运行入口：`code/run_pipeline.py` 一条命令全链 + manifest 审计
- [x] **API 服务化**：`code/fate_service.py`（FastAPI）
      `GET /health`、`GET /v1/route`、`POST /v1/supply|enroll|verify|select|run`，
      可选 `FATE_API_KEY` 鉴权；2026-11-15 本机实活验证全端点通过
      （select 0.6061 与管线账本一致，/run 端到端 ACCEPTED）
- [x] 配置外置：`code/fate_config.py` + `.env.example`
      （VAL_MODELS_ROOT / FATE_TAU / FATE_API_KEY / FATE_HOST / FATE_PORT）
- [x] 一键安装：`install.ps1`（建 venv → CUDA torch → requirements → doctor）
- [x] 环境自检：`code/doctor.py` 参考机全绿（GPU/VRAM/模型/ArcFace/ffmpeg/磁盘）
- [x] 回归防退化：`code/test_fate_consolidation.py` 16 项
- [x] 竞品机制同台：与参考图注入机制（IP-Adapter，平台竞品的共同路线）
      同裁判同协议三轮对表（0.581 vs 0.657；极限 0.552 vs 0.630）

## 2. 进行中

- [ ] **命名竞品同台**：ConsisID（开源指定脸生视频，CVPR 2025）
      权重已下载（25.1GB）、脚本已备 `code/test_consisid_benchmark.py`
      （同一指定脸、同一 ArcFace 裁判、同一门槛）。2026-11-15 本机
      尝试：4090 + CPU offload 下首条 49f 生成 45 分钟未完成
      （GPU 仅 130W，瓶颈在逐层搬运）已止损——**竞品在本机的
      本地时效本身即一条对标数据**（我方 I2V 同规格约 6 分钟/条）；
      正式对标改走夜间长跑或云端 GPU，不阻塞发布准备。
- [ ] 模型权重分发方案： juggernaut-xl-v9 / CogVideoX-2b /
      CogVideoX-5b-I2V / ConsisID 的下载脚本与校验（`download.py` 扩展）

## 3. 未达标（公开发布前必须补，禁止带病上线）

- [ ] 商业平台同台数字（Vidu/Kling/Runway，需购买 API 后按统一协议跑）
- [ ] 分钟级多窗口视频全链演示（I2V 接缝连 4+ 窗口的累积漂移数据）
- [ ] 双人同框高间隔（当前压缩供给归属 2/3，间隔薄）
- [ ] 角色多样性分布（男主/老年/古装/动漫风的候选池过门率未测）
- [ ] 服务化生产加固：异步任务队列、超时/重试、并发限流、
      上传式锚图接口（当前为服务器本地路径）、日志与指标
- [ ] 合规：真人脸授权流程、生成内容溯源标记、第三方模型许可清单
      （Juggernaut/CogVideoX/ConsisID/ArcFace 各自的商用条款核对）
- [ ] 一集端到端演示片（真实分镜脚本 → 成片 + manifest 全程留痕）

## 4. 运行手册（参考机已验证流程）

```powershell
# 安装（新机器）
.\install.ps1                                   # 含 doctor 自检
# 环境变量（按需）
$env:VAL_MODELS_ROOT="C:\Ai\models"; $env:FATE_API_KEY="change-me"

cd code
..\.venv\Scripts\python.exe doctor.py           # 全绿再往下
..\.venv\Scripts\python.exe run_pipeline.py --scenes park,harbour,library --seeds 7,8
..\.venv\Scripts\python.exe fate_service.py     # http://127.0.0.1:8790/health
..\.venv\Scripts\python.exe test_fate_consolidation.py   # 发布前回归，16 项须全过
```

冒烟顺序（任何环境变更后）：doctor → run_pipeline（2 场景）→
/health + /v1/select 各打一次 → 回归脚本。三步任一失败不发布。

## 5. 发布判定规则

- 回归 < 16/16、doctor 非全绿、manifest 出现未解释 ABSTAINED：
  任一成立即不发布；
- 对外数字只能引用 `reports/` 已存档 JSON，竞品对比必须同裁判同协议；
- 本文件第 3 节未清零前，只做邀请制可用，不做公开商业发布。
