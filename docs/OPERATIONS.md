# 运行与共享服务器约定

## 本地

使用 uv 管理 Python 3.12 环境。仓库已有数据渲染、活动标签、数据集索引、冻结 AuT 探针、E/P 训练、
轨迹关联与本地试听入口：

- base：协议、窗口、数据索引等纯 NumPy 路径（`uv sync --locked`）；
- `render` extra：DawDreamer 渲染（`scripts/render_sample.py`、`scripts/label_sample.py`）；
- `ml` extra：CPU torch、transformers、safetensors（`scripts/probe_aut.py`、`scripts/train.py`、`scripts/demo_pipeline.py`）。

真实权重、checkpoint、音频与运行产物只写在忽略的 `runs/` 等目录；不要将本机 VST 路径、模型目录或音频路径
硬编码进提交的配置与代码。GPU 运行需要的 CUDA torch 与锁文件中的 CPU torch 是同一版本的构建变体，
只在独立虚拟环境中安装，不修改 `pyproject.toml`/`uv.lock`。端到端复现命令与“fake vs 真实 AuT”标识见
[docs/REPRODUCE.md](REPRODUCE.md) 与 [docs/reports/issue-11-integration.md](reports/issue-11-integration.md)。

## 共享服务器

具体 SSH 别名和授权工作根目录保存在本地 `.local/compute.toml`，该文件不提交。

- 所有工作限于授权根目录内的项目子目录，使用独立 `.venv`、缓存和输出路径。
- **用户音乐音频与解码样本不得复制到共享服务器**；只能在本地分析，除非另有明确授权。服务器上只处理合成数据、代码、
  只读模型权重与脱敏运行产物；新建任务使用新的子目录（例如 `issue-11-integration/`），已有任务目录（模型、checkpoint、
  venv、数据）只读复用，不删除、不覆盖他人产物。
- Git 身份只通过 `git config --local` 写入本项目仓库，不改变共享账户的全局 Git 配置。
- 不向共享账户复制个人 `.env`、API key、SSH 私钥、GitHub token 或认证缓存。
- 公开仓库可匿名 clone/fetch；也可通过审核过的 Git bundle 同步。服务器不保存个人 GitHub 凭据，推送统一在本地完成。
- 运行前检查 GPU 与进程使用情况。空闲显存快照不等于长期资源预留；不终止他人进程、不修改驱动或系统环境。
- 短任务优先，长任务记录 GPU 编号、PID、启动时间、日志与运行配置，并遵守服务器既有调度约定。
- GPU 不足时先降低批大小、冻结骨干或分阶段缓存特征，再决定使用远端计算。

共享 Unix 用户不构成个人隐私隔离，即便目录仅该用户可读也不能防止同账户其他会话访问；该目录只放本项目获准共享的非敏感材料。

## 数据与提交

不提交音频、模型权重、实验原始数据或运行密钥。推送前检查 `git status` 和 staged diff。`.gitignore` 只是防误提交，不替代检查。

远端初次初始化采用已审查提交的 bundle，并保留 origin URL。仓库公开后可匿名拉取；个人连接信息仍只保存在忽略的本地配置中。
