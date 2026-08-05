# StageDouble 升级计划

## 目标

将现有原型从 2 输出升级为 6 输出，提供三种机器学习模型可选（随机森林、梯度提升、加深 MLP），通过命令行参数切换。

---

## 改动清单

### 文件 1: `model vol.1/ml_trainer_v2.py`（双输入版，主要改动）

| 改动点 | 位置 | 说明 |
|--------|------|------|
| `N_OUTPUTS = 2` → `N_OUTPUTS = 6` | 第 110 行 | 输出参数从 2 个扩到 6 个 |
| 新增 import | 第 92-100 行 | 添加 `RandomForestRegressor`、`GradientBoostingRegressor`、`MultiOutputRegressor` |
| 重写 `build_model()` | 第 182-201 行 | 根据 `--model` 参数返回三种模型之一 |
| 新增 `--model` 参数 | `parse_args()` | 命令行参数，可选 `random_forest`/`gradient_boost`/`mlp` |
| 更新 `_run_loop()` 打印 | 第 266 行 | 6 个输出值的打印格式 |
| 更新 `HELP_TEXT` | 第 306-317 行 | record 命令示例改为 6 个参数 |
| 更新模块 docstring | 第 1-86 行 | 文档中的 2 → 6 |

### 文件 2: `prototype vol.1/audio/ml_trainer.py`（单输入版，同步改动）

| 改动点 | 同上 | 与 v2 保持一致 |
|--------|------|---------------|
| 所有改动 | 同上 | 与 v2 完全一致的改动 |

---

## 三种模型实现

```python
def build_model(model_type="random_forest"):
    if model_type == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "gradient_boost":
        return MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, random_state=42)
        )
    elif model_type == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), solver="lbfgs", max_iter=2000),
        )
```

## 命令行参数

```bash
python ml_trainer_v2.py --model random_forest    # 默认，推荐
python ml_trainer_v2.py --model gradient_boost
python ml_trainer_v2.py --model mlp
```

## 验证方式

1. 启动程序后检查 `record` 命令提示是否显示 6 个参数
2. `record 0.1 0.2 0.3 0.4 0.5 0.6` 录制一条数据
3. `train` 训练，检查是否报错
4. `run` 运行，检查输出是否 6 个值
5. 换 `--model` 参数，重复上述步骤