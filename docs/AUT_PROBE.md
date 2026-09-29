# AuT 独立加载与特征时间轴探针（issue #5）

本文记录 Qwen3-Omni AuT 音频编码器的独立加载、权重筛选审计、真实短前向与资源实测。
**论文/模型卡数字与本文实测数字分开列出**；未实测的内容不写成结论。

范围：`src/aat/encoders/`、`scripts/probe_aut.py`、`tests/encoders/`。
协议仍为 v0.1.0（`docs/SCHEMAS.md`），特征产物遵循 `aat.contracts.FeatureData`。

## 0. 结论摘要（实测）

- 已确认独立类 `Qwen3OmniMoeAudioEncoder`，参数量 **647,927,168**（README 写“约 650M”），
  加载张量 **525** 个、BF16 载荷 **1,295,854,336 B (1.207 GiB)**；没有加载 Thinker/Talker/
  vision/Code2Wav 权重或模块（见 §3 证据）。
- A6000（sm_86，bf16，`torch 2.14.0+cu130`、`transformers 4.57.1`）上：
  - 0.5 s → 7 token，2.0 s → 26 token，8.0 s → 104 token，全部 finite；
  - 2.5 s 前向中位 **8.5 ms**（约 293× 实时），批量 8×2 s 中位 **16.8 ms**（约 951× 实时）；
  - 加载后峰值显存 **1.30 GB allocated / 1.43 GB reserved**（整模型 checkpoint 为 70.5 GB）。
- **窗口与整段不等价（实测）**：对齐到 1 s chunk 网格、完全位于曲内、使用同一权重时，
  独立 2 s 窗口与整段切片的 token 平均 |Δ| 约 0.020–0.028（特征平均 |x|≈0.018），
  平均余弦 0.95–0.97；窗口边缘 token 最小余弦约 0.70，内部最小余弦约 0.97。
  中间缓存整段特征再切片不能替代独立窗口编码。
- **上游 sdpa 路径的注意力掩码缺陷（源码审计 + 实测）**：`Qwen3OmniMoeAudioEncoder.forward`
  计算了 `cu_seqlens`，但从未调用它自己定义的 `_prepare_attention_mask`；
  `sdpa_attention_forward`/`eager_attention_forward` 忽略 `cu_seq_lens_*`，因此会退化为
  “整段全连接”。批量前向会把不同样本互相泄漏（实测 bf16 8×2 s：批量 vs 单条最大差
  0.083）。本实现默认通过 `forward_pre_hook` 注入上游自己的块对角掩码（FA2 varlen 语义），
  float32 下批量 vs 单条最大差从 0.110 降到 **2.6e-4**；13 s 整段 masked vs unmasked
  最大差 0.0615。详见 §4.4。
- `Closes #5` 相关真实验证已具备；**依赖集成（pyproject/uv.lock/CI）仍待主会话安排**，
  见 §7。权重、缓存、音频、完整日志均未进入 Git。

## 1. 选用版本（精确 pin）

| 组件 | 审计/运行版本 | 备注 |
|---|---|---|
| transformers | **4.57.1** | checkpoint `config.json` 记录 `4.57.0.dev0`；4.57.0 已被 PyPI yank。`qwen3_omni_moe/{modeling,configuration,processing}.py` 与 `whisper/feature_extraction_whisper.py` 在 4.57.0 与 4.57.1 **逐字节相同**（`modeling_qwen3_omni_moe.py` sha256 `809eaeb4d40cb0e59965a85b5f85ddc07e9ab7b6cdae84972c711f8cdadec296`）。 |
| torch | **2.14.0+cu130**（实测环境） | 本地 CPU 复现用 2.14.0+cpu；建议正式 pin 由主会话按 CI/GPU 环境确定 |
| safetensors | 0.8.0 | `safe_open` 按 key 取张量，不整 shard 实例化 |
| huggingface_hub | 运行环境随 transformers 安装 | 用于按 revision 下载 index/小配置/必要 shard |
| numpy | 2.5.3（>=2.0,<3） | 纯 NumPy 的 grid/resample/fake 路径 |
| 模型 | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | revision **`26291f793822fb6be9555850f06dfe95f2d7e695`**（HF API 报告 `lastModified 2025-09-22`） |

## 2. 架构与权重审计（下载前完成）

### 2.1 实际类名与参数前缀

- 官方 config：`model_type=qwen3_omni_moe`，音频配置位于 `thinker_config.audio_config`
  （`model_type=qwen3_omni_moe_audio_encoder`）。
- 独立 config/class：
  `transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe.Qwen3OmniMoeAudioEncoderConfig`
  与 `...modeling_qwen3_omni_moe.Qwen3OmniMoeAudioEncoder`。
- checkpoint 张量前缀：**`thinker.audio_tower.`**（525 个张量全部命中）。
- `positional_embedding` 是 **non-persistent buffer**：不在 checkpoint、不在 `state_dict()`，
  由 `SinusoidsPositionEmbedding` 在实例化时重算；审计时已单独记录，避免误判 missing key。
- 关键 config：`d_model=1280`、`encoder_layers=32`、`encoder_attention_heads=20`、
  `encoder_ffn_dim=5120`、`num_mel_bins=128`、`output_dim=2048`、`n_window=50`、
  `n_window_infer=800`、`conv_chunksize=500`、`downsample_hidden_size=480`、
  `max_source_positions=1500`。

### 2.2 需要的张量与形状（525 个）

| 数量 | key 模式 | 形状 |
|---:|---|---|
| 1 | `conv2d1.weight` | [480, 1, 3, 3] |
| 1 | `conv2d1.bias` | [480] |
| 1 | `conv2d2.weight` | [480, 480, 3, 3] |
| 1 | `conv2d2.bias` | [480] |
| 1 | `conv2d3.weight` | [480, 480, 3, 3] |
| 1 | `conv2d3.bias` | [480] |
| 1 | `conv_out.weight` | [1280, 7680] |
| 32 | `layers.N.self_attn.{q,k,v,out}_proj.weight` | [1280, 1280] |
| 32 | `layers.N.self_attn.{q,k,v,out}_proj.bias` | [1280] |
| 32 | `layers.N.self_attn_layer_norm.{weight,bias}` | [1280] |
| 32 | `layers.N.final_layer_norm.{weight,bias}` | [1280] |
| 32 | `layers.N.fc1.weight` | [5120, 1280] |
| 32 | `layers.N.fc1.bias` | [5120] |
| 32 | `layers.N.fc2.weight` | [1280, 5120] |
| 32 | `layers.N.fc2.bias` | [1280] |
| 1 | `ln_post.{weight,bias}` | [1280] |
| 1 | `proj1.{weight,bias}` | [1280, 1280] / [1280] |
| 1 | `proj2.weight` | [2048, 1280] |
| 1 | `proj2.bias` | [2048] |

共 525 个张量、11 种不同 shape、全部 BF16。审计方法：

1. 用 HTTP Range 只读取远端 shard header（`model-00001-of-00015.safetensors`，
   header JSON 187,528 B），得到每个张量的 `dtype/shape/data_offsets`；
2. 本地用官方 `audio_config` 实例化 `Qwen3OmniMoeAudioEncoder`，比较 `state_dict()`；
3. 结果：**missing 0、unexpected 0、shape 不一致 0**；参数量与 BF16 字节数与远端 header 一致。
   以上过程不下载权重、不实例化完整 Omni。

实现对应 `aat.encoders.checkpoint`（index 解析、shard 规划、前缀归一化、严格 key 校验、
字节核算）与 `aat.encoders.aut.load_filtered_audio_encoder`。加载时
`model.load_state_dict(..., strict=True)` 之前先显式检查 missing/unexpected 并抛
`EncoderCheckpointError`——不使用 `strict=False` 静默跳过。

### 2.3 审计引用

- 源码（4.57.1，与 4.57.0 逐字节相同）：
  `https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py`
  （sha256 `809eaeb4d40cb0e59965a85b5f85ddc07e9ab7b6cdae84972c711f8cdadec296`）；
  config/processor 同目录；Mel 前端：
  `https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/whisper/feature_extraction_whisper.py`。
- checkpoint 与 config/preprocessor/index：
  `https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/tree/26291f793822fb6be9555850f06dfe95f2d7e695`。
- 审计只用 HTTP Range 读取 shard header + 本地官方 config 实例化，未下载完整权重，
  对应 `tests/encoders/test_checkpoint.py` 与 `tests/encoders/test_aut_cpu.py` 中的合成
  checkpoint 校验。

## 3. 下载、磁盘与“只加载 encoder”证据

### 3.1 索引/shard 规划（下载前）

- 官方 index：15 个 shard，`metadata.total_size = 70,519,637,090 B`。
- 全部 525 个 `thinker.audio_tower.*` 张量都在 **`model-00001-of-00015.safetensors`**
  （其余 14 个 shard 不含 encoder 张量，不下载）。
- 该 shard 文件 **4,997,899,632 B (4.654 GiB)**，其中张量载荷 4,997,712,096 B；
  真正属于 AuT 的载荷 **1,295,854,336 B (1.207 GiB)**，占该 shard 的 **25.93%**。
- 另外 3 个小文件（index/config/preprocessor）合计 2,741,624 B。
- 下载前按“shard 大小 × 1.2 + 小文件”做 `shutil.disk_usage` 检查；
  实测授权工作盘当时剩余 242 GiB。
- `scripts/probe_aut.py --stage plan` 只用 HF 元数据/索引估算大小，不需要先下载 shard。

### 3.2 实测下载与加载

- `--stage all` 实际新增传输 **4,997,899,632 B**（shard1）+ 小文件；报告
  `transfer_summary` 记录 `bytes_newly_transferred` 与 `bytes_reused`，下载字节与
  “encoder 筛选后字节”分开记录。
- loader 只对 `thinker.audio_tower.` 前缀调用 `get_tensor`，materialise 525 个张量，
  **loaded encoder bytes = 1,295,854,336**，与审计值一致才继续。
- 加载报告中 `class_name=Qwen3OmniMoeAudioEncoder`；模块类名集合仅为
  `Conv2d/GELUActivation/LayerNorm/Linear/ModuleList/Qwen3OmniMoeAudioAttention/
  Qwen3OmniMoeAudioEncoder/Qwen3OmniMoeAudioEncoderLayer/SinusoidsPositionEmbedding`；
  没有 Thinker/Talker/Vision/Code2Wav 类。
- 实测（A6000, bf16）：加载耗时 5.75 s，**峰值 allocated 1,300,442,112 B，
  reserved 1,434,451,968 B**。checkpoint 全量 70.5 GB 与之不相干，未下载其余 14 个 shard。

## 4. 输入、Mel、token 时间与有效 mask

### 4.1 输入与 Mel

- 输入只重采样到模型要求的 **16 kHz 单声道**。渲染器默认 44.1 kHz，`aat.encoders.resample`
  提供 NumPy 窗口化 sinc（Blackman）重采样，测试覆盖 DC 增益、440 Hz 通过、10 kHz
  抗混叠（不允许折叠到 6 kHz）。
- Mel 复用 checkpoint 自带 `preprocessor_config.json` 指定的官方
  `WhisperFeatureExtractor`：`feature_size=128, sampling_rate=16000, hop_length=160,
  n_fft=400, dither=0.0, padding_value=0.0`，并核对 `n_samples=4,800,000` /
  `nb_max_frames=30,000`（即官方单段上限 300 s）。实现拒绝静默截断：mel 超过
  `nb_max_frames` 直接报错，要求切段。
- `torch.stft(center=True)` 要求 `n_fft//2 < 长度`，因此最短可处理音频为 **201 样本
  (12.6 ms)**；更短输入明确报错。

### 4.2 token 网格与时间公式（来自 pinned 源码，不是记忆）

pinned `modeling_qwen3_omni_moe.py` 中：

```
tokens(mel_len) = 13 * (mel_len // 100) + f(mel_len % 100)
f(r) = (( ((r-1)//2 + 1 - 1)//2 + 1 - 1)//2 + 1)   (r > 0), f(0)=0
```

- mel hop 160 样本 = **10 ms**；一个完整卷积 chunk 是 `n_window*2 = 100` mel 帧 = 1 s，
  经三层 kernel-3/stride-2/padding-1 卷积后是 **13** 个 token（100→50→25→13），
  所以“约 12.5 Hz（80 ms）”只描述 chunk 内的名义中心步长；真实 token 数是 **13 个/秒**，
  长音频平均 76.9 ms/token。
- chunk 内 token `k` 的卷积感受野是 mel 帧 `[8k-7, 8k+7]`，中心 `8k`；即 **80 ms 中心步长**。
  chunk 之间没有卷积上下文泄漏（`pad_sequence` + 逐 chunk 卷积）。
- chunk 边界处最后一个 token 中心 0.96 s、下一个 chunk 首 token 中心 1.00 s，
  **边界间隔是 40 ms**。`frame_times` 以感受野中心为准，是严格的绝对时间轴，
  但**不是** 10 ms 精度的能量包络或起音时间；不要把它当逐帧对齐真值。
- `frame_times = buffer_origin + mel_center * 0.01`，单位为原曲绝对秒。窗口若越过原曲
  起点会出现负时间；`AutFeatures.drop_before_track_origin()`/`subset()` 可裁剪，
  `FeatureData` 只接受非负、严格递增时间。

### 4.3 有效 mask 语义

- `valid=False` 仅表示该 token 的 sample-level 感受野（mel 帧 `[a,b]` →
  样本 `[a*160-200, b*160+200)`）碰到了**我们引入的零填充**（窗口首尾补零）。
- 纯 buffer 提取（无人工补零）时所有 token 均为 `True`；编码器自身的卷积边界零填充
  属于模型正常行为，不标记为 invalid。
- 对 chunk 尾部的 partial chunk，field 会被裁剪到真实 chunk 帧范围；模型内部
  `pad_sequence` 的 chunk 内补零不算“人工补零”。

### 4.4 上游注意力掩码缺陷与修正（需 review 的重要发现）

源码审计（4.57.1，与 4.57.0 逐字节相同）：

- `Qwen3OmniMoeAudioEncoder.forward` 计算 `cu_seqlens`（按 `n_window_infer=800` 及
  `aftercnn_lens` 划分为约 104 token ≈ 8.32 s 的块），但循环里只调用
  `encoder_layer(hidden_states, cu_seqlens)`；
- `_prepare_attention_mask` 虽然定义了块对角掩码，但**在 encoder forward 中从未被调用**；
- `sdpa_attention_forward`/`eager_attention_forward` 只使用 `attention_mask` 参数，
  完全忽略 `cu_seq_lens_q/k`。因此非 FA2 后端执行的是**整条拼接序列的全连接注意力**：
  单段变成“全局注意力”，批量前向则会跨样本泄漏。
- `flash_attention_2` 后端使用 `cu_seqlens` 做 varlen，不受该缺陷影响；但安装 FA2 成本高，
  且真实实验更容易命中默认 `sdpa`。

本实现的修正（`AutEncoder` 默认 `masked_attention=True`）：

- 对单个不超过 8.32 s（≤104 token）的输入，块掩码全 0，masked 与 unmasked **完全一致**；
  产生影响的是批量（防止跨样本泄漏）与超过 8.32 s 的整段（块边界）。
- 通过各层 `register_forward_pre_hook(..., with_kwargs=True)` 把上游自己的
  `model._prepare_attention_mask(hidden_states, cu_seqlens)` 结果注入
  `attention_mask` 关键字；不重写 forward、不改权重，并记录
  `last_attention_info`（mask 形状 + cu_seqlens）作为证据；
- 编码器第 31 层（最后一层）hook 输出与普通 forward 输出**逐位相同**（差 0.0），
  说明 hook 路径未改变最终语义。

实测（bf16, A6000；fp32 对照用于分离数值误差）：

| 对照 | 结果 |
|---|---|
| unmasked（上游 sdpa 原样）批量 8×2 s vs 单条，fp32 | 0.110（run2；跨样本泄漏） |
| masked（本实现默认）批量 8×2 s vs 单条，fp32 | **2.6e-4** |
| unmasked 批量 vs 单条，bf16 | 0.083 |
| masked 批量 vs 单条，bf16 | 0.039（主要是 bf16 GEMM 形状差异：bf16 批量 vs fp32 批量 0.041） |
| 13 s 整段 masked vs unmasked，bf16 | 0.0615 |
| 13 s 整段 masked 的块结构 | `cu_seqlens=[0,104,169]`，mask `[1,1,169,169]` |

结论/建议：#7/#8 使用特征时必须**默认启用块对角掩码**；批量组成会带来 bf16 数值差异
（同一条窗口单独跑与混在批量里跑最大差约 0.039），训练/推理应固定批策略，或对可复现性
要求高的场景用 fp32 特征。不要把“全连接注意力”版输出与 masked 输出混用。

## 5. 真实验证结果（A6000，实测）

运行配置：`GPU0 (RTX A6000, sm_86)`、`torch 2.14.0+cu130`、`transformers 4.57.1`、
`bf16`、`sdpa`、warmup 2 + repeats 5、CUDA synchronize、输入为进程内生成的确定性合成
多音信号（不含任何音乐/个人数据）。

### 5.1 不同长度短前向

| 时长 | mel 帧 | token | valid | 首/末 frame_time | 中位延迟 | 实时倍数 | 峰值 allocated Δ | 峰值 reserved |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0.5 s | 50 | 7 | 7 | 0.00 / 0.48 | 7.5 ms | 67× | 8.1 MB | 1.46 GB |
| 1.0 s | 100 | 13 | 13 | 0.00 / 0.96 | 7.2 ms | 139× | 11.9 MB | 1.46 GB |
| 2.0 s | 200 | 26 | 26 | 0.00 / 1.96 | 8.3 ms | 241× | 19.9 MB | 1.47 GB |
| 2.5 s | 250 | 33 | 33 | 0.00 / 2.48 | 8.5 ms | 293× | 27.9 MB | 1.49 GB |
| 8.0 s | 800 | 104 | 104 | 0.00 / 7.96 | 9.5 ms | 839× | 67.4 MB | 1.57 GB |

全部 `finite=True`，特征维度 2048；特征尺度（2.5 s 例）mean|·|≈0.0185、std≈0.0234、
max|·|≈0.177。批量 8×2 s：中位 **16.8 ms**（16 s 音频，**951×** 实时），
峰值 allocated Δ **128 MB**，峰值 reserved 1.64 GB。

### 5.2 44.1 kHz 输入与非零 origin

44.1 kHz 2 s（88,200 样本）→ 重采样 32,000 样本 → 26 token；
`origin_seconds=30.0` 时 `frame_times` 为 30.00 … 31.96，`finite=True`。

### 5.3 首尾补零窗口（3 s 曲内，2 s 窗）

| 窗口中心 | frame_time 范围 | valid token 数 |
|---:|---|---:|
| 0.5 s | −0.50 … 1.46 | 18（左侧补零区域 token 无效） |
| 1.5 s | 0.50 … 2.46 | 26（完全在曲内） |
| 2.5 s | 1.50 … 3.46 | 19（右侧补零区域 token 无效） |

### 5.4 独立窗口 vs 整段切片（13 s 整段，masked，同权重）

对齐到 1 s chunk 网格且完全在曲内；比较相同绝对 frame_time 的 token：

| 窗口起点 | 匹配 token | max\|Δ\| | mean\|Δ\| | 平均余弦 | 最小余弦 | 边缘 mean\|Δ\| | 内部 mean\|Δ\| | 边缘最小余弦 | 内部最小余弦 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 s | 26/26 | 0.093 | 0.026 | 0.953 | 0.707 | 0.033 | 0.016 | 0.707 | 0.973 |
| 3.0 s | 26/26 | 0.078 | 0.020 | 0.965 | 0.697 | 0.023 | 0.015 | 0.697 | 0.977 |
| 5.0 s | 26/26 | 0.104 | 0.025 | 0.972 | 0.726 | 0.028 | 0.021 | 0.726 | 0.968 |
| 7.0 s | 26/26 | 0.091 | 0.028 | 0.958 | 0.726 | 0.036 | 0.017 | 0.726 | 0.982 |

非对齐窗口（起点 1.5/3.5/5.5 s）在 1e-9 容差下没有 token 时间重合（匹配 0），
放宽到 0.04 s 后每窗匹配 26 个：mean|Δ| 0.049–0.057、平均余弦 0.878–0.882。

**结论**：即使 1 s chunk 对齐、使用块对角注意力，独立 2 s 窗口与整段切片差异仍显著
（mean|Δ| 与特征平均幅值同量级，边缘 token 余弦可低至约 0.70）。训练/推理必须固定
“独立窗口”或“整段提取再切片”中的一种，不能假设两者等价；README 的警告已由本实测确认。

### 5.5 中间层 hook

- 第 16 层（0-based，共 32 层）hook → 2048 维、26 token、finite；
- 第 31 层（最后一层）hook 与普通 forward 最大差 **0.0**（逐位一致）。
- 层范围显式限制为 `[0, 31]`，越界报错；语义为 `proj2(act(proj1(ln_post(layer_i 输出))))`，
  是“截断 + AuT 头”的特征，不是上游 `hidden_states` 接口。上游 forward 在 pinned 版本
  不提供 `output_hidden_states`，因此没有伪造该接口。

## 6. fake 路径与 CPU CI

- `aat.encoders.fake.FakeAutEncoder` 纯 NumPy，共享同一 grid/mask/时间轴，无 torch、
  transformers、网络或权重；特征维度默认 16。
- `tests/encoders/` 共 81 项测试：69 项纯 NumPy（grid/resample/fake/checkpoint/feature 容器/
  probe 辅助函数）在只装 numpy+pytest 的 base 环境全部通过；12 项真实类集成测试（tiny
  synthetic checkpoint，需要 torch/transformers/safetensors）在 base 环境整模块 skip，在含
  torch 的环境中全部通过。整套测试含 torch 时 476 passed / 1 skipped（render 集成），
  base 时 464 passed / 2 skipped。
- fake 测试覆盖：0.5/1.0/2.0/2.5/3.13 s、非整 chunk、左右补零 mask、非零 origin、
  批量与单条一致、对齐窗口内部 token 逐位一致、非对齐窗口差异 >0、FeatureData 落盘
  往返、以及子进程验证 `import aat.encoders` 不会引入 torch/transformers。
- 真实路径与 fake 明确分离：`FakeAutEncoder` 是局部确定性代理，不用于冒充真实模型
  结果；真实数据只在 §5 记录。

## 7. 依赖集成状态（重要）

- 本 PR 基于 `31c816a`，分支上**未修改** `pyproject.toml`、`uv.lock`、
  `.github/workflows/ci.yml`（由 #7 持有）。推送时 main 已合并 #6/#7（`5749102`）：当前 main 的
  `ml` extra 只有 CPU torch（显式 CPU wheel index），**仍缺 transformers/safetensors**；
  base/render CI 里真实类集成测试会因缺 transformers 被 `importorskip` 整模块跳过；
  ml job 只收集 `-m ml`（头部测试），不涉及本模块；任何 CI job 都不会联网下载权重。
- **合并前必须由主会话恢复本 session，基于新 main 串行集成 `ml` extra/lock**（增加
  `transformers==4.57.1`、`safetensors`、`huggingface_hub`），并在集成后重跑真实探针与
  `pytest tests/encoders`；在此之前本 PR 不具备合并条件。
- 建议集成内容（精确 pin）：
  - `transformers==4.57.1`（4.57.0 yanked；相关源码与 4.57.0 逐字节相同）；
  - `safetensors`、`huggingface_hub`（权重筛选与下载）；
  - `torch`：按实际 GPU/CI 环境 pin；本地/服务器实测分别为 2.14.0+cpu / 2.14.0+cu130；
  - 保持 base/普通 CI 不因本任务下载权重（满足“CPU CI 仅 fake、不联网”）。
- CI 映射：现有 unit job 的 `pytest -m "not integration and not ml"` 与 render job 的
  `-m "not ml"` 会执行 `tests/encoders` 的纯 NumPy/fake 测试；含 torch 的真实类集成测试
  需要 transformers 后才能从 skip 变为实跑（当前先 skip，避免 CI 下载模型栈）。
  真实探针不进 CI。

### 复现命令（脱敏模板）

```sh
WORKDIR=/path/to/workdir
uv venv --python 3.12 "$WORKDIR/.venv"
uv pip install --python "$WORKDIR/.venv/bin/python" \
    "transformers==4.57.1" torch safetensors huggingface_hub "numpy>=2.0,<3"
# 只读审计/规划（不下载 shard）
PYTHONPATH=src "$WORKDIR/.venv/bin/python" scripts/probe_aut.py --stage plan \
    --model-dir "$WORKDIR/model" --cache-dir "$WORKDIR/hf-cache" --output-dir "$WORKDIR/out/plan"
# 下载必要 shard 并跑真实探针
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$WORKDIR/.venv/bin/python" scripts/probe_aut.py --stage all \
    --model-dir "$WORKDIR/model" --cache-dir "$WORKDIR/hf-cache" --output-dir "$WORKDIR/out/run" \
    --device cuda:0 --dtype bfloat16 --layer 16
# 单元/集成测试
PYTHONPATH=src "$WORKDIR/.venv/bin/python" -m pytest tests/encoders -q
```

## 8. 未验证 / 不宣称

- 不做完整 Omni 问答、Talker/视觉前向、全量微调或大规模特征提取；未测量 30 s/300 s
  长音频的完整显存和吞吐（本文最长 13 s/8 s）。
- 不宣称 10 ms 时间精度；token 时间代表卷积感受野中心，见 §4.2。
- 不宣称窗口与整段等价（已实测不等价）；不宣称 fake 结果代表真实模型。
- 下载与加载字节按 pinned revision 审计；换 revision/换 transformers 需重跑
  `--stage plan` 与 key 审计。
- 合成信号只用于打通真实前向与资源测量，不代表音乐来源分离效果。
- 运行机器上的其他进程/显存占用未作为本任务测量的一部分；探针前后 GPU 占用快照显示
  本任务进程退出后释放了显存。
