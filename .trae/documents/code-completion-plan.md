# 代码补全计划

## 问题清单

### 1. ml_trainer_v2.py -- 残留的"2 输出"文案
| 行号 | 现状 | 改为 |
|------|------|------|
| 31-32 | "同样输出 2 个数值" | "同样输出 6 个数值" |
| 72 | "predict 2 values" | "predict 6 values" |
| 83-84 | "预测出 2 个数值" | "预测出 6 个数值" |
| 84-85 | "消费这 2 个输出参数" | "消费这 6 个输出参数" |
| 253 | "返回 [out1, out2]" | "返回 6 个输出值" |
| 429 | "record <out1> <out2> [seconds]" | "record <mouth>..<speed> [seconds]" |

### 2. ml_trainer_v2.py -- 输出值未限制范围
**问题**：RandomForest 和 GradientBoost 可能预测出 <0 或 >1 的值，直接发给 MetaHuman 会导致表情异常。
**改法**：在 `predict()` 方法中对输出做 `np.clip(out, 0.0, 1.0)` 截断。

### 3. audio_extractor.py -- 引用旧文件名
| 行号 | 现状 | 改为 |
|------|------|------|
| 28-29 | "ml_trainer.py" | "ml_trainer_v2.py" |
| 34-35 | "ml_trainer.py" | "ml_trainer_v2.py" |

### 4. motion_extractor.py -- 注释中的旧路径
| 行号 | 现状 | 改为 |
|------|------|------|
| 29 | "model vol.1/models/" | "models/" |
| 38 | "model vol.1/models/" | "models/" |
| 122 | "model vol.1/models/" | "models/" |

## 验证
逐行检查修改后的文件，确认无 "2 个数值"、"model vol.1"、"ml_trainer.py"（非 v2）残留。