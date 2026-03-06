# UniSRec_1 Project

## 简介 (Introduction)

本项目是一个基于 [RecBole](https://recbole.io/) 和 PyTorch 实现的通用序列推荐（Universal Sequence Recommendation, UniSRec）系统。项目支持模型的预训练（Pre-training）和微调（Fine-tuning），并针对特定数据集（如 `lianhua`）进行了适配。

## 项目结构 (Project Structure)

```text
UniSRec_1/
├── config.py           # 配置加载逻辑
├── finetune.py         # 模型微调入口脚本
├── pretrain.py         # 模型预训练入口脚本
├── predict.py          # 模型预测脚本
├── unisrec.py          # UniSRec 模型定义
├── data/               # 数据加载与处理核心逻辑
│   ├── dataloader.py
│   ├── dataset.py
│   └── transform.py
├── dataset/            # 数据集目录
│   ├── downstream/     # 下游任务数据（处理后）
│   ├── preprocessing/  # 数据预处理脚本 & BERT模型
│   └── raw/            # 原始数据
├── props/              # 配置文件目录
│   ├── UniSRec.yaml    # 主配置文件
│   ├── finetune.yaml   # 微调特定配置
│   └── pretrain.yaml   # 预训练特定配置
├── saved/              # 保存的模型权重 (.pth)
└── wandb/              # Weights & Biases 日志
```

## 环境要求 (Requirements)

- Python 3.8+
- PyTorch (建议 1.12+)
- CUDA (如果使用 GPU)

主要依赖库：
- recbole
- transformers
- torch
- wandb
- pandas
- numpy
- tqdm

## 安装说明 (Installation)

1.  **克隆项目**
    ```bash
    git clone <repository_url>
    cd UniSRec_1
    ```

2.  **创建并激活虚拟环境 (推荐)**
    ```bash
    conda create -n unisrec python=3.9
    conda activate unisrec
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```
    *(注意：如果根目录下没有 `requirements.txt`，请参考 `wandb` 目录下的日志文件或手动安装上述核心依赖)*

## 数据准备 (Data Preparation)

原始数据应放置在 `dataset/raw/` 目录下。

数据预处理脚本位于 `dataset/preprocessing/process_or.py`。该脚本负责加载原始数据（如 `lianhua.csv`），执行 K-core 过滤，生成文本特征，并转换为模型所需的原子文件格式。

**运行预处理：**

```bash
cd dataset/preprocessing
python process_or.py --dataset lianhua --input_path ../raw/
```

参数说明：
- `--dataset`: 数据集名称
- `--user_k_min`: 用户最少交互数过滤阈值
- `--item_k`: 商品最少被交互数过滤阈值
- `--input_path`: 原始数据路径

## 配置说明 (Configuration)

项目使用 YAML 文件进行配置，主要位于 `props/` 目录下：

- **`props/UniSRec.yaml`**: 包含模型架构参数（如 `hidden_size`, `n_layers`）、训练参数（`epochs`, `batch_size`）等通用配置。
- **`props/finetune.yaml`**: 微调阶段特定的配置。
- **`props/pretrain.yaml`**: 预训练阶段特定的配置。

可以在运行脚本时通过命令行参数覆盖部分配置，或者直接修改 YAML 文件。

## 使用说明 (Usage)

### 1. 模型预训练 (Pre-training)

使用 `pretrain.py` 脚本进行模型预训练。

```bash
python pretrain.py -d lianhua
```

- `-d`: 指定数据集名称（默认为 `lianhua`）。

### 2. 模型微调 (Fine-tuning)

使用 `finetune.py` 脚本加载预训练模型并进行微调。

```bash
python finetune.py -d lianhua -p ./saved/UniSRec-LIANHUA-6.pth
```

- `-d`: 指定数据集名称。
- `-p`: 指定预训练模型的路径（`.pth` 文件）。
- `-f`: 是否固定编码器参数 (True/False)。

### 3. 模型预测 (Prediction)

使用 `predict.py` 进行推理（具体用法请参考脚本实现）。

## 结果与日志 (Results & Logs)

- **模型权重**: 训练好的模型会保存在 `saved/` 目录下。
- **日志**: 训练日志和指标通过 `wandb` 记录，保存在 `wandb/` 目录下，也可在控制台查看。

## 常见问题 (FAQ)

- **OOM (Out of Memory)**: 如果遇到显存不足，请在 `props/UniSRec.yaml` 中减小 `batch_size` 或 `MAX_ITEM_LIST_LENGTH`。
- **数据路径错误**: 请确保 `dataset/` 下的目录结构符合预期，且配置文件中的数据路径正确。
