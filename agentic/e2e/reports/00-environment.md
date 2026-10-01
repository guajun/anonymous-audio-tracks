# 00 — 环境、预算与真实/mock 边界（issue #33）

> 全文脱敏：路径用 `<WS>`（workspace）、`<REPO>`（仓库 checkout）、`<SAM_ROOT>`（SAM checkout）、
> `<AUDIO>`（真实源音乐）占位；不含音频、权重、API key、完整会话。样本 hash 保留（单向，可复核）。

## 1. 环境（真实测量）

| 项 | 值 | 性质 |
|---|---|---|
| Pi | 0.87.1（native argv：Node + `dist/bundle/cli.js`，经 `pi_launcher`） | 真实 |
| 研究模型 | `google/gemini-3.8-flash`（`.pi/settings.json` 固定，run 命令**不传** `--model`） | 真实 |
| 实施 worker 模型 | `openrouter/xiaomi/mimo-v2.6-pro`（与研究模型不同角色） | 真实 |
| Google 认证 | `pi auth check --provider google --json` → `{"status":"ready"}`（只查状态，不打印 key） | 真实 |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU，8192 MiB，driver 616.64；run 前占用 2672 MiB / 空闲 5347 MiB | 真实 |
| SAM | checkout `<SAM_ROOT>`，torch 2.10.0+cu130（CUDA 可用），权重 `model-cache/`（上游 `verify_models.py` SHA-256 已校验，主 Agent `doctor --deep` 记录 18 ok/0 fail） | 真实 |
| SAM 工具 | `sam-audio` skill pin `dfbc40a9541f…`；上游 commit `c603de8794cc…`；CLI `audio-toolbox.sam/v1` | 真实 |
| bridge | `.pi/extensions/audio-bridge.ts` sha256 `dac9abe5…`（#30 冻结 pin；**≠ Pi 原生音频**） | 真实 |
| validator | `agentic/schema/validate.py`（#32 冻结），本 run 用 stdlib 引擎 | 真实 |
| harness 运行时 | Python 3.14（纯标准库，无第三方依赖） | 真实 |

## 2. 输入 clip（受控，守桥接预算）

| 项 | 值 |
|---|---|
| clip 名 | `e2e-clip-001.wav`（位于 `<WS>/audio/inputs/`，#31 音频 manifest 已登记） |
| 来源 | 真实复音音乐 `<AUDIO>`（用户自有；本机 `data/` 下，不入 git、不上传 GitHub） |
| 上传边界 | **裁剪后的 clip 以 inlineData 内联上传 Google/Gemini API 分析**（本 issue 明确允许；非 Files API）；源音乐不上传；**SAM 全程本地离线** |
| 裁剪 | offset **4.0 s**，时长 **16.0 s**（`harness/make_clip.py`，帧级裁剪，未转码） |
| 规格 | 44 100 Hz，2 声道，16-bit WAV；**2 822 444 bytes = 2.69 MiB < 4 MiB**（#30 单文件预算） |
| clip sha256 | `42f968e24dee43008a1e4164ae5567a503d45bc3211db5b6d6b96971b06dc7f0` |
| 时轴（冻结） | clip 的 `t=0` = 源文件 4.0 s 处；**所有 onset 用 clip 时轴的秒**；result 的 `duration_seconds=16.0` |
| 隐私 | clip/源音频只在 ignored 目录；公开材料只留 hash 与规格 |

## 3. 预算（本次实际消耗）

| 预算项 | 上限 | 实际 |
|---|---|---|
| 真实 Gemini run | ≤2（主 + 有限 followup） | **1**（主 run `e2e-real-20261001-02`，无 followup 需要） |
| 真实 SAM GPU 分离 | ≤3 次/clip | **3**（drums / bass / synthesizer；各先 `--dry-run` 再真跑） |
| 单次 SAM `--timeout` | 900 s | 实际上游推理 21.6–22.2 s/次 |
| Pi run 硬超时 | 2400 s | wall 354.9 s（未触发） |
| 校验修复循环 | ≤2 | **0**（一次通过） |
| 参数扫描 | 禁止 | 未做 |

## 4. 真实 / mock 边界（冻结口径）

| 产物 | 分类 | 说明 |
|---|---|---|
| `e2e-real-20261001-02` run 全部产物 | **真实** | 真实 Gemini + bridge 音频 + 真实 SAM GPU 分离 + 真实 DSP |
| `fixtures/make_mock_fixture.py` 产物 | **mock** | 自生成合成复音 + 构造真值，仅离线测试；`kind="mock"` |
| 离线测试（40 项） | **mock/离线** | 0 网络 0 API 0 GPU |
| `harness/spotcheck.py` 统计 | **DSP 抽查迹象** | 对真实 run 产物的离线统计；**不是**总体准确率 |

mock/native/bridge 三者在自动化里可区分（`tests/test_validate_result.py`）：
mock 结果在 `--level real` 必须被拒；结果把音频通路标 `native`（Pi 原生音频 unsupported）必须被拒；
真实音频通路只能是 `bridge`（`audio_attach`），run manifest 记 `audio_pathway.kind="bridge"`。

## 5. 复现命令（Junior 单条级；`<WS>` 见上）

```sh
# 0) 前置：doctor --deep 无失败（#31）；pi auth ready；GPU 空闲
python agentic/workspace-template/doctor.py --workspace "<WS>" --deep

# 1) 受控 clip（真实源音频在本机）
python agentic/e2e/harness/make_clip.py --source "<AUDIO>.wav" --offset 4.0 --duration 16.0 \
  --workspace "<WS>" --name e2e-clip-001.wav

# 2) 真实端到端 run（产生 API 费用 + GPU 负载）
python agentic/e2e/harness/run_e2e.py --workspace "<WS>" --run-id <RUN_ID> \
  --clip-name e2e-clip-001.wav --timeout 2400

# 3) 复核 + 抽查（离线）
python agentic/e2e/harness/run_e2e.py --workspace "<WS>" --run-id <RUN_ID> --clip-name e2e-clip-001.wav --verify
python agentic/e2e/harness/spotcheck.py stats --result "<WS>/outputs/e2e/<RUN_ID>/result.json" \
  --audio "<WS>/audio/inputs/e2e-clip-001.wav" --run-dir "<WS>/outputs/e2e/<RUN_ID>" --kind real
```
