# 纯推理预测重构设计

## 目标

将现有预测流程改为纯模型 TopK 推理，并最大化单机 GPU 推理效率。预测阶段只读取预处理数据和微调模型，只生成一份扁平 CSV：

```text
user_id,item_id,score
```

## 结果口径

- 为数据集中的全部预测用户生成结果。
- 屏蔽 padding 商品和用户输入序列中的全部历史商品。
- 按模型分数直接选择每个用户的 TopK 商品。
- 不读取商品品类或有效商品名单。
- 不执行品类打散、在售过滤或最多 150000 用户筛选。
- 不读取或写入 ODPS。
- 不生成 details CSV。

唯一产物为：

```text
<work-dir>/results/<dataset>-recommendations.csv
```

## 接口变化

独立预测命令保留：

```text
--stage predict
--dataset
--finetuned-checkpoint
--top-k
--work-dir
--python
```

从 `run.sh` 和 `predict.py` 的预测接口删除：

```text
--item-metadata-csv
--item-metadata-table
--eligible-items-csv
--eligible-items-table
--output-table
```

`--env-file` 继续供 `fetch` 的 ODPS 输入使用，不再用于预测。完整流水线 `all` 只要求一种交互数据来源，不再要求预测商品信息来源。

## 推理架构

### 模型层

在 `UniSRec` 中提供两个推理方法：

1. 计算并归一化全量商品向量；每次预测任务只执行一次。
2. 使用缓存的商品向量计算一个用户 batch 的全量商品分数。

训练方法和现有 `full_sort_predict()` 保持原有行为，避免影响预训练、微调和 RecBole 评估。

### 预测层

`predict.py` 依次执行：

1. 使用 `configs/UniSRec.yaml` 和 `configs/finetune.yaml` 构建数据集及模型。
2. 加载微调检查点并进入 `eval()` 模式。
3. 预先构建 RecBole 内部 ID 到原始 ID 的数组映射。
4. 计算一次全量商品向量缓存。
5. 在 `torch.inference_mode()` 中逐 batch 推理。
6. 将 padding ID 和输入序列中的历史商品分数设为负无穷。
7. 获取每个用户的 TopK，过滤非有限分数。
8. 立即将该 batch 追加写入 CSV。

预测过程不累计全量用户、候选商品或分数。

## 性能策略

- 全量商品向量从“每 batch 计算一次”改为“每次任务计算一次”。
- 使用 `torch.inference_mode()` 关闭梯度和版本计数。
- 使用 `config['device']` 统一模型、输入和缓存向量设备。
- 使用 GPU 原地 `scatter_` 屏蔽历史商品。
- 只计算请求的 TopK，不再固定获取 800 个候选。
- 预先计算 ID 映射数组，避免在内层循环重复查找 JSON 和 RecBole token。
- 使用标准库 `csv.writer` 流式写出，删除 pandas 中间对象。

## 边界行为

- `top_k` 必须为正整数。
- 实际可推荐商品少于 `top_k` 时，只写出有限分数对应的商品。
- 微调检查点、预测原子文件、商品向量或 ID 映射缺失时立即失败。
- 输出目录不存在时自动创建。
- 每次运行覆盖同名结果 CSV，避免混入上一次结果。
- 模型配置与检查点结构不一致时由严格权重加载直接报错。

## 代码范围

- `unisrec.py`：增加商品向量缓存和缓存推理方法。
- `predict.py`：重写为纯推理与流式 CSV 输出。
- `run.sh`：删除预测业务数据和 ODPS 参数，简化 `all` 与 `predict`。
- `tests/`：增加历史屏蔽、TopK 和 CSV 输出测试，更新流水线参数测试。
- `README.md`：更新完整流水线和独立预测说明、参数表、数据流与输出。

## 验证标准

1. 历史商品和 padding 不出现在推荐 CSV。
2. 每个用户最多输出 `top_k` 行，按分数降序排列。
3. 输出只包含 `user_id,item_id,score` 表头和数据。
4. 预测接口不再接受商品元数据、有效商品或 ODPS 输出参数。
5. `all` 无需预测商品信息参数即可执行到预测阶段。
6. 全量商品向量在一次预测任务中只计算一次。
7. 流水线测试、预测单元测试、Bash 语法和 Python 语法检查通过。

## 不在本次范围内

- 近似最近邻索引或分布式推理。
- 混合精度和模型量化。
- 改变训练、微调或预处理业务逻辑。
- 恢复品类打散、有效商品过滤或 ODPS 输出。
