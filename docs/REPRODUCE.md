# 复现指南（issue #11 端到端链路）

本文件说明如何从零复现「配置/checkpoint → E/P → 稳定轨迹 → 本地试听产物」链路，以及 fake CPU smoke
与真实冻结 AuT 两条证据链的固定命令。所有命令不需要手工编辑 JSON；参数通过 CLI 或可选 TOML overlay 传入。
真实音频、权重、checkpoint 与运行产物不得提交，公共仓库只保留代码、配置与脱敏汇总。

前置：Python 3.12（由 uv 管理）、Git、Node ≥ 22（viewer 校验可选）、FFmpeg/ffprobe（仅本地音频模式需要）。

## 0. 安装环境

```sh
uv sync --locked                     # base（协议/数据索引等，无 torch）
uv sync --locked --extra ml          # CPU 仿真与真实冻结 AuT 推理（CPU torch）
uv sync --locked --extra render      # DawDreamer 渲染（需要真实合成数据时）
uv sync --locked --extra ml --extra render   # 全部
```

CI 三个 job 与本地一致：base 跑 `-m "not integration and not ml"`，render 跑 `-m "not ml"`，
ml 跑 `-m ml`。`tests/integration/test_demo_pipeline.py` 同时带 `integration` 与 `ml` 标记，
并在模块顶部 `pytest.importorskip("torch")`，因此 base/render 环境只会 skip，不会因缺 torch 收集失败。

```sh
uv run --no-sync pytest -q -p no:cacheprovider -m "not integration and not ml"   # base
uv run --no-sync pytest -q -p no:cacheprovider -m ml                             # ml（含 GPU 无关的 CPU smoke）
node --test "tests/viewer/**/*.test.js"                                          # viewer 契约（74 项）
```

## 1. CPU fake smoke（工程链路，非模型结果）

```sh
uv sync --locked --extra ml
OMP_NUM_THREADS=4 uv run --no-sync python scripts/demo_pipeline.py smoke \
    --out runs/demo/smoke --steps 3
```

它会用 `aat.training.dataset.make_smoke_dataset` 写出明确标注 `synthetic` 的协议语料（非 DawDreamer），
训练 3 步，然后**从磁盘重载 checkpoint**，对保留曲目滑窗推理、关联、序列化产物并用 Node viewer 契约校验。
预期：退出码 0，stdout 中 `mode=fake-encoder-smoke`、`viewer_validation=ok`，`pipeline_report.json` 的
`result_kind` 含 “not a real model result”，所有 trajectory 的 `data_kind=mock`。产物布局见
`docs/reports/issue-11-integration.md` §1；查看器会把 `mock` 标记为不能当作模型结果。

同一链路的自动化版本（CI 使用）：

```sh
uv run --no-sync pytest -q -p no:cacheprovider tests/integration
```

覆盖：生产 CLI 全链路、checkpoint 重载逐位一致性、`origin_seconds=12.0` 非零原点、2 s 窗口两端补零的
canonical 无效槽位、`predictor_for_centers(..., audio_duration_seconds=...)` 与 `predict_at_times` 掩码一致、
无 duration 回调用途的边界限制、artifact sha256、覆盖保护、viewer 协议校验。

## 2. 真实冻结 AuT：保留合成集评估

准备（只读复用，不覆盖）：

1. **编码器权重**：issue #5 的独立 AuT 目录（`config.json`、`preprocessor_config.json`、
   `model.safetensors.index.json` 与含 `thinker.audio_tower.*` 的分片），revision 必须为
   `26291f793822fb6be9555850f06dfe95f2d7e695`。加载器只实例化 `Qwen3OmniMoeAudioEncoder`，不加载完整 Omni。
2. **checkpoint 与数据集**：issue #8 的训练产物（`checkpoint.pt`、`runs/train-index/index.json`、
   `runs/train-data/`）。重建方式见 `docs/TRAINING.md`；评估阶段不训练、不改权重。
3. GPU 环境：在独立 venv 中使用 CUDA torch（同版本 `torch==2.14.0`，可写子目录内 UV 缓存），
   例如：

```sh
uv venv --python 3.12 .venv
UV_CACHE_DIR=<writable-cache> uv sync --locked --extra ml --no-install-package torch
UV_CACHE_DIR=<writable-cache> uv pip install --python .venv --no-sources \
    --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.0"
```

运行（固定顺序、无选歌、无调参）：

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 timeout 600 \
    .venv/bin/python scripts/demo_pipeline.py synthetic \
    --checkpoint <issue-8>/checkpoint.pt \
    --index <issue-8>/runs/train-index/index.json \
    --data-root <issue-8>/runs/train-data \
    --split test --songs test-01,test-02 \
    --model-dir <issue-5-aut-model> --device cuda:0 \
    --out runs/demo/aut-test
```

`synthetic` 会先用 `evaluate_split` 走 issue #8 的固定协议（阈值 0.5、整曲全局一对一声明映射、
`all_inactive` / `all_active` / `no_identity` 三个基线），再对同一批歌单独跑一遍
`predict_at_times → associate_sequence` 并保存 `prediction/`、`trajectory.json`、`session.json`、
`manual_inspection.md`；报告里逐曲记录 `canonical_metrics_match`（两条路径的指标是否一致），并把
`--songs`/`--max-songs` 的选择写入 `provenance.dataset.selection`。**canonical 指标与基线只覆盖选中的歌**：
索引在评估前被过滤为精确的 `evaluated_songs`，不会用全 split 排序前缀冒充选择结果。
预期（A6000，两首 9.5 s 曲目）：整体约 3–4 分钟，`timeout 600` 不触发；报告含 GPU 峰值（本进程）与
encoder revision/attention/window policy。实测数字与失败例见 `docs/reports/issue-11-integration.md` §3。

完成状态：每个模式都会校验**每一条**产出的轨迹（Node 缺失为 `skipped`，不算失败）；任一 viewer 校验失败或
`canonical_metrics_match=false` 时，仍写入诊断产物但返回非零退出码（1），不会把失败当成功上报。

## 3. 真实冻结 AuT：本地用户音频（不含真值）

要求 FFmpeg/ffprobe 可用。音频只在本机解码与分析，不上传；输出目录在忽略路径下。

```sh
OMP_NUM_THREADS=8 uv run --no-sync python scripts/demo_pipeline.py audio \
    --checkpoint <issue-8>/checkpoint.pt \
    --model-dir <issue-5-aut-model> --device cpu \
    --audio /path/to/clip.mp3 \
    --start-seconds 30.0 --duration-seconds 8.0 \
    --hop-seconds 0.10 \
    --out runs/demo/clip-a
```

- `--hop-seconds` 默认 0.02 s（协议网格）；issue #11 的固定真实音乐抽查为了控制 CPU 预算明确使用
  **0.10 s**，报告会写出「比合成 20 ms 粗」的说明。W=2 s 与阈值/关联参数保持不变，禁止用片段调参。
- `--duration-seconds` 缺省时按 `ffprobe` 时长计算到文件末尾；`--start-seconds + --duration-seconds`
  超过源时长会**直接报错**，不会静默替换片段。`pipeline_report.json` 记录源文件 SHA-256、精确区间、
  解码命令与 FFmpeg 版本、解码后 SHA-256，并核对解码帧数与请求时长（声明容差 0.01 s）：截断/加长的输出
  会在推理前直接失败，不会把不足 8 s 的解码结果当成完整片段上报。
- 参数校验（布尔、正数、合法 split、命令相关键）在**任何目录归档、解码或模型加载之前**完成；`--out` 不得
  等于或位于 checkpoint/index/数据/音频/模型/仓库输入的祖先路径，`--label` 必须是安全的单层目录名。
- 无真值 → 不输出 precision/recall/F1；`manual_inspection.md` 默认 `human listening status: pending`。

## 4. 本地试听（viewer）

只开本地静态服务，页面无远程请求、无上传、无遥测：

```sh
uv run python -m http.server 8123 --bind 127.0.0.1 --directory viewer
# 浏览器打开 http://127.0.0.1:8123/
```

依次加载（必须同一分析片段）：`songs/<id>/` 下的解码 wav（或合成数据集的 `mix.wav`）与 `trajectory.json`。
页面显示窗口中心活动概率，同时显示本地播放时间与原曲绝对时间；`track_id` 是关联出的匿名身份，不是乐器名。
按 `manual_inspection.md` 的记录表在绝对时间轴上填写可分离层数、误检、漏检、身份交换与边界早晚；
听完前不要宣布任何听感结论，也不要把客观轨迹当作准确率。

## 5. provenance 与产物契约

每个模式的 `pipeline_report.json` 都包含：

- Git SHA + dirty、`uv.lock` SHA-256、训练/推理配置 SHA-256；
- 数据集索引 SHA-256、内容 digest、split/leak 摘要（有索引时）；
- checkpoint SHA-256 与 step，编码器 identity/provenance（model id、revision、dtype、attention backend、extraction、window、batch 策略）；
- pinned policy（W=2.0 s、fp32、block_diagonal/sdpa、independent windows）；
- 种子、中心步长、活动阈值、设备与资源统计（CPU 明确 `null` + 原因）；
- `viewer_validation`（总体 `status` + 每首歌 `per_song`）与 `checks`（完成检查的失败清单）；
- `artifacts.json`：其余产物逐文件 sha256。

`trajectory.json` 与 `prediction.json/npz` 均遵循 `docs/SCHEMAS.md` v0.1.0；所有时间为原曲绝对秒，
首尾补零窗口在 prediction 中为 `center_valid=false`、全零无效槽位，轨迹关联与评估都会跳过这些中心。

## 6. 隐私与边界

- 不提交音频、权重、checkpoint、NPZ、运行日志或私人绝对路径；`runs/`、`*.wav`、`*.mp3`、`*.pt`、`*.safetensors` 已被忽略。
- 用户音频与解码样本只留本机；如需远端 GPU 验证，只允许非音乐材料（例如合成数据、模型权重只读），
  音频不得复制到共享服务器。
- 本任务不改变协议版本、依赖锁或 CI；真实权重与 checkpoint 均按固定 revision/哈希引用。
