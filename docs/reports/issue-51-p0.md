# #51 P0 启动与 patch 特征诊断

历史记录：下文描述最初因frame入口失败而停在P0的执行轮。
用户随后提供本地frame权重；新的frame P0和P1结果见
[frame / C0报告](issue-51-frame-c0.md)，不以此历史阻塞描述当前状态。

2026-10-04，用户以 SSH 服务器路径和“开工”授权执行草案。
本次完成远程环境检查、P0 代码、真实音频 smoke 与 patch 探针；
**frame 权重来源门槛未通过，P1/P2 未启动，没有训练结果。**
研究 [#51](https://github.com/guajun/anonymous-audio-tracks/issues/51) /
[draft #52](https://github.com/guajun/anonymous-audio-tracks/pull/52)。

## 隔离与来源

本地仍使用本会话 managed worktree，远程新建
`/mnt/ssd/user/lgy/issue-51-ssast`，从已推送研究分支克隆。
没有写共享本地 checkout、远程旧研究目录或现有SSAST目录。
GPU 0 为 RTX A6000（48 GB），GPU 1 的桌面进程保留。

现有SSAST `.venv` 的torch/torchaudio报undefined symbol，不能直接使用。
本研究 `.venv-p0` 添加自己的timm0.4.5/torchvision0.24.1/绘图依赖，
通过只读`.pth`使用既有研究runtime的torch2.9.1+cu128、torchaudio2.9.1+cu128
及renderer。此环境尚非完全自包含；重建依赖见研究目录requirements-p0.txt。
一次独立完整安装因`/home`磁盘不足失败，随后下载缓存改到本研究SSD目录；
失败尝试没有用于探针。不改变main pyproject/lock。

官方固定源码commit为`a1a3eecb94731e226308a6812f2fbf268d789caf`。
frame Dropbox入口实际返回HTML，直接内容域名404，OneDrive403。
找到的HF frame转换镜像只有自动生成model card，无法据此独立确定转换来源；
未加载或拿来替代官方frame。访问记录见
[checkpoint access](../../evidence/issue-51-p0-checkpoint-access.json)。

patch采用服务器原有文件，SHA256为
`f03ee45f984ce778fb75741e5a51c4b78a8ce639e71f407ae6a9fb4e96e242bf`。
全部166个预训练键严格加载、无missing/unexpected keys，形状符合16×16 base；
**本次未独立建立其原始下载来源，也没有官方校验摘要进行对照**。
严格加载证明文件与架构兼容，不自动证明来源真实性。

## 实测结果

P0调用main `prepare_c0(counts=(1,0,0))`新渲染一个样本，不复用旧模型研究代码。
从44.1kHz原音频裁剪实际2秒输入，显式重采样至16kHz；
测试真实音频、半/双增益、静音、脉冲、阶跃、10ms循环移位和左边界补零共8项。
移位是循环移位诊断，不用于训练或原时间轴标签。

| 核查 | 实测 |
|---|---|
| 输入log-fbank | `[8,198,128]`，固定官方统计，无频谱padding |
| 未池化token | `[8,12,19,768]`，无cls/dist、无分类embedding |
| 时间中心（裁剪局部坐标） | 87.5ms至1887.5ms，100ms步长 |
| 投影支持 | 175ms；attention仍看整段2秒 |
| 下游位置张量 | `[1,230,768]`，时间裁剪起点23，频率插值8→12 |
| 坐标核验误差 | `1.11e-16`秒，与离散卷积区间中点一致 |
| 相同输入重复最大误差 | `0.0` |
| batch1每窗encoder耗时 | `4.311ms` |
| batch4每窗encoder耗时 | `3.281ms`（batch约13.124ms） |
| CUDA峰值allocated / reserved | 约414MiB / 792MiB |
| 进程峰值RSS | 约1.88GiB（包括加载/渲染过程） |

计时为5次暖机后20次、每段同步GPU，只测encoder，不包含重采样/fbank/磁盘/头。
60秒曲目H=20ms的3000窗仅encoder线性估算约9.84秒，不是端到端实测。
峰值含batch8探针，不能当作仅batch4训练内存。
真实波形与token投影支持有效掩码保存到远程validity.npz；掩码不改变attention。

增益变化使特征改变：半增益/双增益相对真实特征平均向量距离约1.044/1.943。
特征模长仍约20.25，而静音特征模长约21.60，因此encoder模长不能直接当活动A。
保留原44.1kHz多声道功率平均的full-scale RMS旁路；
这些响应没有证明可线性解码准确RMS，更没有证明匿名分离。
机器结果见 [P0 patch JSON](../../evidence/issue-51-p0-patch.json)。

## 验证与后续

4个frontend/网格/实际Conv2d展平顺序/错类型checkpoint测试通过。
探针两次运行均完成；第二次增加真实波形与token支持掩码和RSS，最终计时采用第二次。
共享数据、标签和上下文审计模块没有修改。TimeCycle未实现、未接入。

frame路径的下一步是可核验原始权重；否则可由用户选择patch独立P1基线。
候选镜像缺少来源说明及已有patch兼容性不能替代frame阶段门槛。
用户已收到这两条路径的具体选择；本次没有把patch实测写成frame验收，
没有越过P0门槛启动训练。
