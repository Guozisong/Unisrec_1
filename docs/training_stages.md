# UniSRec 预训练与微调任务对比

本文档详细对比了 UniSRec 项目中预训练（Pre-training）和微调（Fine-tuning）两个阶段在数据集构建、优化目标、输入输出以及数据处理流程上的异同。

## 1. 任务概述 (Overview)

| 阶段 | 核心目标 | 适用场景 |
| :--- | :--- | :--- |
| **预训练 (Pre-training)** | 从交互序列与文本特征学习序列表示。 | 在当前数据集上训练序列编码器。 |
| **微调 (Fine-tuning)** | 适配下一个商品预测任务。 | 从当前数据集的预训练权重继续训练。 |

## 2. 数据集与输入 (Dataset & Input)

| 特性 | 预训练 (Pre-training) | 微调 (Fine-tuning) |
| :--- | :--- | :--- |
| **数据来源** | 当前流水线读取 `<dataset>.train.inter`。 | 当前流水线读取同一次预处理得到的 `<dataset>` 训练、验证和测试文件。 |
| **商品表示** | **纯文本特征 (PLM)**。利用 BERT 等模型提取的通用文本向量 (`plm_emb`)。 | **PLM + ID Embedding**。在 Transductive 模式下，结合 PLM 特征和领域特定的 Item ID Embedding。 |
| **ID 映射** | 当前入口使用 `UniSRecDataset` 的单数据集商品映射。 | 使用同一数据集的商品映射。 |
| **输入 X** | 原始序列 (`item_seq`) + **增强序列** (`item_seq_aug`)。 | 仅原始序列 (`item_seq`)。 |
| **数据加载类** | `UniSRecDataset`（通过预训练配置加载增强特征） | `UniSRecDataset` |

## 3. 优化目标与损失函数 (Optimization Objectives)

| 特性 | 预训练 (Pre-training) | 微调 (Fine-tuning) |
| :--- | :--- | :--- |
| **任务类型** | **自监督对比学习 (Contrastive Learning)** | **监督学习 (Supervised Learning)** |
| **损失函数** | **多任务对比损失**:<br>1. `Seq-Item Contrastive`: 序列 vs 下一个商品<br>2. `Seq-Seq Contrastive`: 序列 vs 增强序列 | **交叉熵损失 (Cross-Entropy Loss)**:<br>预测下一个商品的概率分布。 |
| **标签 y** | **隐式标签**。通过 Batch 内的正负样本对定义：<br>- 正样本：自身的增强视图或真实下个商品<br>- 负样本：Batch 内其他序列 | **显式标签**。真实的下一个商品 ID (`pos_items`)。 |
| **负采样** | **Batch 内负采样** (In-batch Negatives)。 | **全量 Softmax** (计算所有候选商品的 Logits)。 |

## 4. 数据处理流程示例 (Data Processing Workflow)

### 4.1 预训练数据处理
假设用户序列为 `[A, B, C, D]`，预测目标为 `E`。

1.  **输入构建**:
    *   原始序列 `S`: `[A, B, C, D]` -> 对应特征 `[Vec_A, Vec_B, Vec_C, Vec_D]`
    *   正样本商品 `I_pos`: `E` -> 对应特征 `Vec_E`
2.  **数据增强 (Augmentation)**:
    *   随机 Mask (例如丢弃 `B`) -> 增强序列 `S_aug`: `[A, C, D]`
    *   增强特征: `[Vec_A, Vec_C, Vec_D]`
3.  **模型计算**:
    *   计算 `S` 与 `Vec_E` 的相似度 (Seq-Item)。
    *   计算 `S` 与 `S_aug` 的相似度 (Seq-Seq)。
    *   最大化上述相似度，同时最小化与 Batch 中其他样本的相似度。

### 4.2 微调数据处理
同样的用户序列 `[A, B, C, D]`，预测目标为 `E`。

1.  **输入构建**:
    *   原始序列 `S`: `[A, B, C, D]`
2.  **特征查找**:
    *   查找 PLM 特征 `[Vec_A, ...]` + (可选) ID Embedding。
3.  **模型计算**:
    *   编码器输出序列表示 `H_s`。
    *   计算 `H_s` 与**所有候选商品** (Item A ~ Z) 的得分。
    *   计算交叉熵损失，使得商品 `E` 的概率最大。

## 5. 代码实现参考 (Code References)

*   **数据集定义**: [`UniSRecDataset`](../recbole_data/dataset.py)
*   **数据增强**: [`PLMEmb`](../recbole_data/transform.py)（Item Drop / Word Drop）
*   **模型与损失**: [`UniSRec`](../unisrec.py)（预训练对比损失与微调交叉熵损失）
