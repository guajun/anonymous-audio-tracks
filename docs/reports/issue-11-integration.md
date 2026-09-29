# 端到端集成与固定真实音乐抽查（issue #11）

本文报告 issue #11 的交付与证据：参数化入口把「数据/checkpoint → E/P → 稳定轨迹 → 本地试听产物」串成可重跑链路，
并在固定选择规则下完成了 1 次真实冻结 AuT 的保留合成集评估与 2 个固定用户音乐片段的本地客观推理。
真实音频、权重、checkpoint 与预测数组只存在于忽略目录或授权远端目录，公共仓库只提交代码、文档与脱敏汇总。

**本文明确不声称**：M3 研究成功、M5 完成、LED 谱面接入；用户音乐没有真值，因此不报告总体准确率，也没有任何人工听感结论
（人工检查在本文与产物中均标记为 **pending**，等待 lead/用户实际试听）。

## 0. 模式区分（不得混用）

| 模式 | 入口 | 编码器 | `result_kind` / `data_kind` | 能证明什么 | 不能证明什么 |
|---|---|---|---|---|---|
| CPU fake smoke | `scripts/demo_pipeline.py smoke` | `FakeAutEncoder`（NumPy，无权重） | `fake-encoder-smoke (engineering pipeline only, not a real model result)` / `mock` | 数据→训练→checkpoint 重载→E/P→关联→序列化→viewer 校验的工程链路 | 任何来源识别/活动质量结论 |
| 真实冻结 AuT 合成评估 | `scripts/demo_pipeline.py synthetic` | 冻结 Qwen3-Omni AuT + issue #8 checkpoint | `aut-frozen-checkpoint-inference (real frozen AuT features)` / `model` | 真实特征上的保留集活动/边界/身份/来源数指标与三个常量基线 | M5 级迁移能力、长时间轴质量、统计显著性 |
| 本地用户音乐 | `scripts/demo_pipeline.py audio` | 同上，本地 CPU | `model` | 固定片段上的客观推理产物（轨迹、概率、viewer 会话） | 准确率（无真值）、身份收益、人工听感 |

## 1. 参数化入口与产物

`scripts/demo_pipeline.py` 是单文件参数化入口（CLI 参数 + 可选 `--config` 扁平 TOML 覆盖，命令行优先，不需要手改 JSON）：

```sh
uv run --no-sync python scripts/demo_pipeline.py smoke --out runs/demo/smoke --steps 3
uv run --no-sync python scripts/demo_pipeline.py synthetic \
    --checkpoint <checkpoint.pt> --index <index.json> --data-root <data-root> \
    --split test --songs <id-a>,<id-b> --model-dir <aut-model-dir> --device cuda:0 \
    --out runs/demo/aut-test
uv run --no-sync python scripts/demo_pipeline.py audio \
    --checkpoint <checkpoint.pt> --model-dir <aut-model-dir> --device cpu \
    --audio <local-audio> --start-seconds 30.0 --duration-seconds 8.0 \
    --hop-seconds 0.10 --out runs/demo/clip
```

每个输出根目录包含：

```text
pipeline_report.json      # 模式、结果类别、全部 provenance、指标/客观检查、失败与限制
artifacts.json            # 其余产物的 sha256 清单（报告自身不列入）
songs/<sample_id>/
  prediction/prediction.json + prediction.npz   # 协议 0.1.0 E/P 文档
  trajectory.json                               # 稳定 track_id 轨迹（原曲绝对时间）
  evaluation.json                               # 有真值时逐曲评估
  session.json                                  # 本地 viewer 会话（音频/trajectory/时间轴/隐私）
  manual_inspection.md                          # 人工检查模板（默认 pending）
train/checkpoint.pt + train_log.jsonl + summary.json  # smoke 模式的训练产物
```

报告始终记录：Git SHA 与 dirty、`uv.lock` SHA-256、`config_sha256`、数据集索引 SHA-256 与内容 digest、
checkpoint SHA-256 与 step、编码器 identity/provenance（model id、revision、dtype、attention backend、
extraction、window、batch 策略）、随机种子、选择的窗口/步长与阈值、设备与资源统计。
`data_kind=mock` 的 fake 轨迹在 viewer 中会显示“不能当作模型结果”的警告。

## 2. CPU CI smoke（fake，工程证据）

新增 `tests/integration/test_demo_pipeline.py`（`integration` + `ml` 双标记，模块顶部
`pytest.importorskip("torch")` 在任何 ML import 之前）。它通过生产 CLI 端到端执行：
`make_smoke_dataset`（明确标注 `synthetic=True` 的协议语料，非 DawDreamer）→ 3 步训练 → **从磁盘重载 checkpoint**
→ 对保留曲目滑窗推理 → `associate_sequence` → 保存 `prediction.json/npz`、`trajectory.json`、viewer 会话
→ 用真实 `viewer/js/protocol.js`（Node，若可用）校验轨迹，并检查 Git/lock/config/数据集 digest、编码器 pinned policy、
artifact 哈希、覆盖保护。

实测（本机 Windows，Python 3.12.11，torch 2.14.0+cpu）：

```text
tests/integration:                        4 passed in 19.78s
全量（base+ml，render 缺省跳过）:          681 passed, 2 skipped, 4 deselected
base-only 环境（无 torch，模拟 unit job）: 574 passed, 13 skipped, 15 deselected
  tests/integration 单独收集:             1 skipped（importorskip 生效，base/render CI 保持有效）
viewer Node 测试:                         74 passed
```

额外覆盖：checkpoint 两次重载的逐位一致预测；`origin_seconds=12.0` 的非零原点仍保持原曲绝对时间；
2 s 窗口在 4 s 音频两端产生 canonical 补零无效槽位（`center_valid=False`、`E=P=0`、`slot_valid=False`）；
`predictor_for_centers(..., audio_duration_seconds=...)` 与 `predict_at_times` 掩码一致，而缺少 duration 的
callback 只保护起点边界（端点边界未知）——与 `docs/TRAINING.md` 的既有说明一致。

持久化的本机 smoke 产物（忽略目录）：`runs/demo/smoke-final/`（checkpoint、prediction、trajectory、session、
manual_inspection.md、viewer 校验 `ok`）。这是 **fake 工程证据**，不是模型结果。

## 3. 真实冻结 AuT 保留合成集评估

### 3.1 固定条件与资源

- 授权远端 A6000，GPU 0（40 GB），仅新建 `issue-11-integration/` 子目录可写；独立 venv（Python 3.12.11，
  uv 0.8.18，torch 2.14.0+cu130，transformers 4.57.1）；只用现有 `ml` extra 的依赖集合，GPU 轮子为同版本
  CUDA 构建。
- 权重：issue #5 的只读模型目录，revision `26291f793822fb6be9555850f06dfe95f2d7e695`（explicit）；
- checkpoint：issue #8 已验收 `aut-short`，`config_sha256=fbe4a2ab392598f0556907477976c0f22d5aeeb5955b68e4db9b2173c807f800`；
- 数据集：issue #8 训练数据索引，索引 SHA-256 `dda89f11…`，内容 digest `c073f9cd…`（与 #8 记录一致），`leak_free=true`，
  `cross_split_assets` 全 0；
- pin：W=2.0 s 独立窗口、fp32、`block_diagonal`/`sdpa`、`independent_windows`、`batch_windows=8`、seed 20260929；
- 一次 GPU 任务，`timeout 600`：整体 `elapsed_seconds=201.3`（最终代码 4748be9 重跑；首次 195.9 为同一结果），进程内 torch peak allocated 2.94 GB / reserved 3.21 GB
  （scope：本进程，从编码器加载前到结束；其他进程不可见）。

### 3.2 固定顺序、无模型选择

在 `test` split 的 2 首曲目上按 `sample_id` 固定顺序 `test-01, test-02` 全部运行（未挑歌、未调参、未用 clip 选阈值）。
两首均为未见 composition/preset/sample 家族，与 train/val 共享资产数 0：

| sample | composition | preset | sample_origin | 来源数 | 时长 |
|---|---|---|---:|---:|---|
| `test-01` | `comp-test-a` | `aat/test/pad-v1`, `aat/test/pluck-v1` | `generated/test/*` | 2 | 9.5 s |
| `test-02` | `comp-test-b` | `aat/test/pad-v1`, `aat/test/pluck-v1`, `aat/test/snare-v1` | `generated/test/*` | 3 | 9.5 s |

### 3.3 活动、边界、身份、来源数指标

逐曲（阈值 0.5，整曲全局一对一声明映射，逐曲 376 个有效中心）：

| sample | TP | FP | FN | precision | recall | F1 | ID switch | 来源数绝对误差 | onset MAE (s) | offset MAE (s) | 边界漏段 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test-01` | 218 | 297 | 18 | 0.423 | 0.924 | **0.581** | 3 | 2 | 1.235 | 0.332 | 1/6 |
| `test-02` | 258 | 444 | 39 | 0.368 | 0.869 | **0.517** | 1 | 1 | 0.776 | 0.677 | 3/10 |

两首合计（micro）与同协议基线：

| 指标 | 模型 | all_inactive | all_active | no_identity |
|---|---:|---:|---:|---:|
| F1 | **0.544** | 0.0 | 0.163 | 0.538 |
| precision | 0.391 | null | 0.089 | 0.387 |
| recall | 0.893 | 0.0 | 1.0 | 0.884 |
| TP / FP / FN | 476 / 741 / 57 | 0 / 0 / 533 | 533 / 5483 / 0 | 471 / 746 / 62 |
| ID switch | 4 | 0 | 0 | 4 |
| 歧义归属帧 | 504 | 0 | 533 | 504 |
| 来源数绝对误差 | 3（预测 8 vs 参考 5） | 5 | 11（预测 16） | 11（预测 16） |

与 issue #8 已验收结果的连续性：`test-01` 的逐项指标与 #8 `eval_test.json` 记录一致（F1 0.581、P 0.423、R 0.924、ID switch 3、来源数绝对误差 2，均为四舍五入）；本流水线另用保存下来的 artifact 轨迹独立复算，两首的 `canonical_metrics_match=true`，`canonical_metrics_max_abs_delta=0.0`（判据 1e-6），即 `evaluate_split` 与「保存的 trajectory.json 重新评估」指标一致（两条路径的窗口分块不同，但仍数值相同）。

### 3.4 失败例与观察

- **误检主导**：FP 741 远大于 FN 57；预测轨迹数 8 大于参考来源数 5，其中 `test-01/trk-0003`、`test-01/trk-0004`、
  `test-02/trk-0004` 从未映射到任何 GT 来源（来源数膨胀）。
- **逐来源质量不均**：`test-01/s01` F1 0.298（FP 139、FN 12），`test-02/s01` F1 0.184（FP 104、FN 38）、
  `test-02/s03` F1 0.435（FP 191、FN 1）；相对较好的 `test-01/s02` F1 0.865。
- **身份没有可测收益**：去掉身份向量（`no_identity`）后 F1 0.538 与模型 0.544 几乎相同（逐曲 0.578 vs 0.581、
  0.509 vs 0.517），说明当前 checkpoint 的可测性能主要来自活动概率而非身份匹配。
- **身份切换与歧义**：模型 ID switch 合计 4（`test-01/s02` 3 次、`test-02/s03` 1 次）；同时在重叠区有 504 个
  「无法仅凭活动真值唯一归属身份」的帧（多 GT 来源或多预测轨迹同时活动），评估不猜 owner——重叠音乐的身份诊断覆盖有限。
- **边界**：onset MAE 0.78–1.24 s、有 4 段参考活动未配到重叠预测段；在 H=20 ms 的网格上仍明显偏大。
  未发现时间轴被缩放或逐帧重排 GT 的迹象（评估协议固定、逐曲无重新匹配）。
- **训练侧歧义监督**（#8 已记录，本次未改）：100 step 中 126/200 组出现并列最优匹配导致身份项被 mask；
  这与「身份无收益、FP 多」互相印证，但本任务不做损失/监督审计。

## 4. 用户音乐片段（本地、无真值）

### 4.1 固定输入与解码（在评估之前确定，无挑选）

| 片段 | 原始文件 SHA-256 | 原始时长 | 分析区间 | 解码后 SHA-256 |
|---|---|---:|---|---|
| clip-A（`1000APM`） | `67c56684…f219f18d` | 132.872 s | 30.000–38.000 s | `563793ec…4511ccade` |
| clip-B（`16-Mirror-inst`） | `d478a35b…107365b4e` | 294.000 s | 30.000–38.000 s | `f3a6f9cb…477bdb9a` |

- 解码命令固定为输入 seek：`ffmpeg -hide_banner -nostdin -y -ss 30.000000 -t 8.000000 -i <source> -ac 1 -ar 44100
  -c:a pcm_s16le <out.wav>`（完整命令在 `pipeline_report.json` 中）；FFmpeg `N-122571-g4ad20a2c09-20260128`。
- 解码前用 `ffprobe` 校验源时长 ≥ 区间终点，否则**报错退出**，不静默替换片段；原始 mp3 未被修改或上传。
- 音频与解码样本只在本机；本地 viewer 不上传任何内容。

### 4.2 推理设置

同一 issue #8 checkpoint，本地 CPU（i7-11800H，`OMP_NUM_THREADS=8`），W=2 s 不变；
**H=0.10 s**（预算约定的粗网格，明确比合成的 20 ms 粗；这是固定选择，未用片段选参）。
每片段 `timeout 600` 内完成：实测 28.8 s / 29.2 s（含模型加载）；阈值 0.5、关联参数全部来自 checkpoint 配置，未调参。

### 4.3 客观推理结果（不是准确率，不是听感）

| 片段 | 有效中心 / 网格 | 轨迹数 | 活动时段（原曲绝对秒） |
|---|---:|---:|---|
| clip-A | 61 / 81（20 个边缘补零窗口） | 8 | `trk-0001` 31.0–32.6、32.8–33.5…；`trk-0002` 31.0–37.0；`trk-0004` 31.0–37.0；其余 5 条为短时活动 |
| clip-B | 61 / 81 | 3 | `trk-0001` 31.0–37.0；`trk-0002` 31.0–32.2、32.4–32.5…；`trk-0003` 32.3–36.7 多段 |

- 两段都使用 K=8 容量；clip-A 占满 8 条轨迹、clip-B 3 条，说明「槽位数不等于来源数」的语义在真实音频上同样成立。
- 所有时间保持原曲绝对秒（30.0–38.0 起点不回零），viewer 同时显示本地播放时间与绝对时间。
- **没有观测到超出 10 分钟预算的问题**；`resources` 明确记录 CPU 不适用 CUDA 峰值（null + 原因），不伪造 0。

### 4.4 人工检查（pending）

每片段产物目录包含 `manual_inspection.md` 模板与 `session.json`，启动方式：

```sh
uv run python -m http.server 8123 --directory viewer
# 浏览器打开 http://127.0.0.1:8123/ 并依次加载解码后的 wav 与 trajectory.json
```

模板要求按「可分离层数、误检区间、漏检区间、身份交换、边界早晚」在原曲绝对时间上记录，且默认标记
**human listening status: pending**。在完成人工试听前，issue #11 的「真实片段抽查」与 PR 状态保持 `Refs #11`。

## 5. 限制与阻塞

- 真实合成评估只有 2 首、每首 9.5 s 的自合成数据，指标方差未知；`val` 在 #8 开发中被观察过。
- 没有做学习曲线/超参/损失审计；身份无收益、误检与来源数膨胀是当前 checkpoint 的可测行为，不是上限结论。
- 用户音频没有真值：不报告 precision/recall/F1/身份指标；人工听感仍未执行。
- 本地 CPU 推理与远端 GPU 推理之间只核对了两条链路的策略参数与保留集合指标，没有做数值级等价声明。
- 边界指标在 20 ms 中心网格上定义；本任务不改变时间对齐、不重采样真实音乐、不按节拍补物件。
- 本地用户片段推理依赖 FFmpeg 做 mp3→WAV 解码（未引入新的 Python 依赖）。

## 6. 主会话（lead）方向性决策（lead-directed，证据限定）

依据上述证据，**主会话决定：暂不扩大昂贵训练**。优先审计训练监督中的并列/歧义匹配（#8 有 126/200 组身份项被
mask）与误检/来源数膨胀（FP 主导、未映射轨迹、`no_identity` 与模型几乎相同），先把「活动概率过关但来源数虚高」
的问题定位清楚；若监督澄清后边界误差仍然偏大，再单独立项做一次受控的高时间分辨率特征对比（W/E/P 协议不变，
固定数据划分），而不是在本任务里顺手调参。上述后续实验**不在 issue #11 范围内**。

## 7. 验收清单映射

| issue 验收项 | 证据 |
|---|---|
| 配置→本地试听的整链路可重跑、不需手改 JSON | `scripts/demo_pipeline.py` 三模式 + `--config` overlay；`docs/REPRODUCE.md`；本文件 §1–§4 |
| CPU CI 端到端 smoke 与一次真实短运行均有证据且清楚标识 | §2（fake，`mock`）与 §3（真实冻结 AuT，#8 checkpoint），产物中 `result_kind`/`data_kind` 区分 |
| 真实片段仅抽查、不报总体准确率、不缩放时间或迁就节拍 | §4（无真值、无准确率、绝对时间轴、固定 H=0.10 粗网格说明） |
| 指标不好也提交真实结果，主会话据此决策 | §3.3–3.4 真实低分与失败例；§6 lead 决策 |

## 8. 证据位置（本地忽略目录，不随公开仓库提交）

- `runs/demo/smoke-final/`：CPU fake smoke 全链路产物；
- `runs/demo/aut-test-remote-final/`：远端真实冻结 AuT 合成评估回拷产物（2 首，含 prediction/trajectory/session/report，最终代码 4748be9）；
- `runs/demo/clip-1000APM/`、`runs/demo/clip-16-Mirror/`：本地用户片段产物（含解码音频，仅本机）；
- `runs/model-cache/qwen3-omni-aut/`：按 pinned revision 下载的本地只读权重缓存（隔离）；
- 远端授权目录 `issue-11-integration/`：独立 venv、git bundle 克隆、合成评估原始输出与日志。

公共仓库只含本文件与 `docs/REPRODUCE.md` 的脱敏数字、代码与测试，不含音频、权重、checkpoint、绝对私人路径或完整会话日志。
