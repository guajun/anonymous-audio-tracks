# Issue #46 C0 连续包络实验

设计与损失选项见 [ISSUE_46_DESIGN.md](ISSUE_46_DESIGN.md)。首个可执行阶段为 C0，
包含 4 个训练、2 个验证、2 个测试片段；一个固定 pad 音色，随机 ADSR/gain/力度/时长。
这是参数与演奏保留集，不是未见音色泛化测试。所有产物只写在忽略的 runs 或任务目录。

## 运行环境

使用独立 Python 3.12 环境。服务器验证环境使用 torch/torchaudio 2.9.1+cu128、
demucs 4.0.1、DawDreamer 0.9.0。不要给既有任务环境升级 torch。
完整依赖入口为 configs/research/envelope-requirements.txt；CUDA wheel 单独指定索引：

```sh
uv venv --python 3.12 .venv-envelope
uv pip install --python .venv-envelope/bin/python torch==2.9.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-envelope/bin/python -r configs/research/envelope-requirements.txt
```

Windows 将解释器路径替换为 `.venv-envelope/Scripts/python.exe`。
新环境安装外部包，训练/测试通过 scripts 的源码路径运行，不修改主项目 uv.lock。

## 复现

```sh
python scripts/envelope_experiment.py prepare-c0 --out runs/envelope/c0
python scripts/envelope_experiment.py cache --data runs/envelope/c0 \
  --out runs/envelope/cache --device cuda:0 --hop-seconds 0.04 --batch-windows 4
python scripts/envelope_experiment.py train --cache runs/envelope/cache \
  --out runs/envelope/sweep --device cuda:0 --steps 100 --seed 46 \
  --families l1 huber area_iou huber_iou multiscale
```

每个输出目录必须为空，避免覆盖历史实验。标签是 20 ms 网格；上述预算探针的模型输出为
40 ms 网格，cache 可设 0.02 恢复密集输出。缓存步长必须是标签步长的整数倍。
缓存包含原曲中心时间、FP16 冻结特征、原始线性 RMS 目标、输入摘要与骨干身份。

## 特征与输出头

- 冻结官方 htdemucs 单模型，记录下载权重摘要，不使用其四种分离输出作为轨迹。
- 首版使用原样 eval 前向并 hook cross-transformer 的频域与时域输出。
  为保留官方归一化和 segment 补零行为，参考路径仍执行 decoder，不能称为 encoder-only 加速。
- 将时域分支线性重采样至频域时间格，再沿通道拼接。只保留应用窗口内的名义 STFT 帧。
  时间格记录为名义 bin center，尚未宣称通过脉冲探针认证所有感受野边界。
- 应用上下文为 2 秒；报告同时记录模型实际执行 segment、原始特征尺寸、显存与耗时。
- 共享 128 维卷积投影、逐槽位竞争分配、来源上下文和中心证据池化、共享向量输出。
  没有可学习 Query，也没有每槽独立描述子输出层。额外背景分配通道可吸收未归属内容。
- 共享映射的原始向量乘以该槽位中心分配占比，得到 z；其方向为 E，半径 r
  映射为 A=r/(1+r)。未分配中心证据的槽位不能直接复制全混音强度。A 拟合线性 RMS，不是概率。
- Demucs 会对输入归一化，因此输出头还接收混音中心 RMS 作为绝对幅度线索。
  单来源的混音 RMS 本身就是目标基线，必须报告，不能把 C0 拟合当作来源解耦成功。

## 损失与验证边界

首轮实现 R0–R4：L1、Huber、面积 IoU、Huber+IoU、多尺度 Huber。
默认 Huber delta=0.05、IoU weight=0.1、空槽/静音 weight=1，均可通过 CLI 改动。
Gaussian 尺度默认 0/20/50/100 ms，含原尺度；TimeCycle 为 0。
默认权重是预声明的探针起点，不是已经调优或完成量级标定的结论。

每段一次完整排列匹配，包含未分配槽位的代价；并列最优对称平均。
C0 直接比较整段槽位包络，不声称验证了 E 的区分能力或跨槽位关联。
可微双随机关联展开已提供，且有“形状误差能回传至描述子”的测试；C1 才启用与评估它。

输出包括每种损失的初始验证、最终验证/测试 MAE、相对 L1、面积 IoU、空槽均值、
混音 RMS 基线、参数量、日志、耗时及 checkpoint。初步只做一个固定种子与短预算；
不能称为损失排名的统计结论，也未自动进入更复杂课程。

## 独立包络产物

envelope.json / envelope.npz 使用 `aat-envelope-v1`，单位 `linear_rms_full_scale`。
它与旧 activity.json / activity.npz 的 probability 语义独立。
记录来源顺序、绝对中心时间、20 ms 测量窗、有效中心掩码、原始分轨摘要及数组摘要。
双声道按等功率平均，不将反相声道相加导致能量抵消。
连续强度尚未直接写入旧 PredictionData/Trajectory 或 viewer；后续接入必须显式标注语义。

测试覆盖连续值/时间轴/幅度比例/摘要完整性、空目标梯度、IoU 几何、整段接错轨迹、
含空槽的精确匹配目标、多尺度梯度及描述子经软关联得到梯度。
