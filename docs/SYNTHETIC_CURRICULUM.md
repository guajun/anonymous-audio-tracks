# 可复用的合成课程与连续包络管线

这部分从 #46 的研究实现提取，复用已有 DawDreamer 渲染器，供不同 backbone 和输出头实验使用。
它只生成混音、效果后分轨、控制事件、连续 RMS 标签与数据审计，不加载模型、不训练、不关联轨迹。
`aat.data.curriculum`、`aat.data.local_context`、`aat.labels.envelope` 均不依赖 torch 或 Demucs。

## 使用

```sh
uv sync --locked --extra render
uv run --no-sync python scripts/prepare_curriculum.py prepare-c0 --out runs/data/c0
uv run --no-sync python scripts/prepare_curriculum.py prepare-c1 --out runs/data/c1
uv run --no-sync python scripts/prepare_curriculum.py prepare-c1-local --out runs/data/c1-local
```

默认 train/val/test 数量为 4/2/2；`--counts 1 0 0` 可做真实渲染 smoke。
每组有独立默认 seed（4600 / 4610 / 4640），可用 `--seed` 指定。
输出目录必须不存在或为空，拒绝覆盖已有数据。counts 必须为三个非负整数且总和大于零。
渲染额外依赖只在真正渲染时需要；已有渲染样本可以仅用 numpy 提取标签：

```sh
uv run --no-sync python scripts/prepare_curriculum.py label --sample runs/render/sample \
  --hop-seconds 0.02 --energy-window-seconds 0.02
```

Python 入口为 `prepare_c0`、`prepare_c1`、`prepare_c1_local`（`aat.data.curriculum`）与
`label_sample`、`EnvelopeData.load`（`aat.labels.envelope`）。课程配置可通过 `c0_config`、
`c1_config`、`c1_local_config` 获取，供调用方基于通用渲染配置继续扩展。

## 课程及边界

| 课程 | 实际内容 | 数据验收 | 适用范围 |
|---|---|---|---|
| C0 | 单个固定 pad、3 个间隔 note、随机 ADSR/velocity/gain，8 秒+2 秒尾音 | 起音前尾音 RMS、跨来源 overlap、stem sum | 连续包络工程验证；只有一源，混音 RMS 即目标，不证明分离能力 |
| C1 | 固定 pad/pluck，5 个交错事件、随机首来源和起音扰动，14 秒+2 秒尾音 | 起音前尾音、实测非重叠 | 较长休止的课程；不能据此要求 2 秒局部描述子跨不相交上下文重连 |
| C1-local | 固定 pad/pluck，共36个交错短事件，每来源18个，12 秒+1 秒尾音 | 零 overlap，1..11 秒评分区间每个20 ms中心的真实2秒输入共现审计 | 局部对应验证；同来源事件间有真实零活动，不添加底音 |

当前只实现这三组固定音色、非重叠课程。overlap、音色保留集和更真实音乐需要调用方扩展，
不能将现有产物描述成广泛音色泛化基准。
train/val/test 保留的是独立时序、ADSR、velocity 与 gain，固定音色有意共享。
这与 `aat.data.index` 的按资产隔离数据协议不同：课程 `index.json` 是独立版本，不能直接当成
通用数据索引或未见音色测试。旧 binary activity 数据路径保持独立。

## 连续标签

每个样本在通用渲染产物之外包含 `envelope.json` 与 `envelope.npz`：

- 版本 `aat-envelope-v1`，单位 `linear_rms_full_scale`；数组为 `center_times[N]`、`rms[N,S]`、`valid[N]`。
- 从效果后分轨实测 centered rectangular mean-square 再开平方；多声道先平均声道功率，避免反相抵消。
- 默认20 ms测量窗、20 ms中心步长；保留原曲绝对时间轴、真实静音及所有来源的共同幅值尺度。
- 不按单来源峰值归一化；ADSR 是渲染控制，RMS 是实际声音标签，二者不等同。
- 中心越过音频末尾时 `valid=False`；这只描述测量中心有效性，不代表模型上下文完整。
- 校验分轨路径不逃逸、音频采样率/长度一致、manifest分轨摘要、标签数组摘要与元数据形状。

`overlap_ratio` 是门限1e-3下，至少两来源活动的中心数 / 至少一来源活动的中心数。
它是数据审计指标，不是模型预测的存在概率。

## 真实上下文审计

`aat.data.local_context.audit_local_context` 接受连续标签、实际推理中心、音频帧数及原时间轴起点。
输入长度、边缘余量和 RMS门限可配置，默认分别2秒、50 ms、1e-3。
边界按测量窗与标签网格作保守扩展。每个评分输入须完整覆盖每来源至少两个事件；
活动中心须有相邻同来源事件证据，事件间的静音中心须同时看见前后端点。
音频外补零不算证据，模型的执行segment不进入该定义。

C1-local 同时输出 `local-context.json` 和 `local-context.npz`，包含真实窗口边界、完整事件数量、
活动相邻证据、静音端点覆盖及摘要。生成器只审计20 ms标签网格；不同模型若使用不同推理中心，
必须再调用此函数审计实际输入，不能自动继承标签网格的验收结果。
该约束保证可见证据，不能保证模型正确匹配。

## 复现与研究隔离

数据index保存seed、配置中的ADSR/gain、起音/来源信息、mix/envelope摘要和stem sum证据。
历史版本字符串保持不变，以保留 #46 产物语义；此处仅迁移模块路径并添加独立CLI。
音频、数组和数据目录不入Git。没有预训练权重下载，没有 backbone cache 或训练依赖新增。

原研究记录保留在 [#46](https://github.com/guajun/anonymous-audio-tracks/issues/46) 和
[草稿 PR #47](https://github.com/guajun/anonymous-audio-tracks/pull/47)。Demucs、向量头、shape/PIT loss、
关联器、TimeCycle、TensorBoard、实验报告和数值结果均不属于这次项目级提取。
