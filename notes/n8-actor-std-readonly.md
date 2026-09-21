# N8：Actor 标准差读取不应改写参数

日期：2026-09-21

## 结论

已修复 `Actor.get_std` 的读取副作用。修复前，`std` 参数化会在首次构造动作分布时执行 `self.std.clamp_()`；全量 B 的 seed0/step1000 checkpoint 因而有 25/64 个元素被原地改写，最大改变量为 `2.408e-5`。这会让只读评估改变已加载的模型状态，也会破坏依赖前后 state hash 相同的诊断。

修复后，getter 用函数式投影产生相同的有界标准差，模型参数和 `state_dict` 保持不变。该修改不改变 checkpoint schema，已有包含 `std` 或 `log_std` 参数的 checkpoint 可继续严格加载。

## 实现选择

`std` 路径使用直通投影：

```text
projected = clamp(std, lower, upper)
result = stop_gradient(projected) + (std - stop_gradient(std))
```

这样同时保留以下行为：

- 前向数值仍依次遵守 `std_clamp_min/std_clamp_max` 和 `max_noise_std`；
- 对 `std` 的局部梯度仍为 1，包括边界外元素。这与旧实现“先在 `no_grad` 中原地投影，再返回参数本身”的梯度语义一致；
- 读取、推理和构造分布不再写参数。

区别是优化器持有的原始参数不会在 forward 时被强制投影；输出标准差始终受边界约束。对旧实现而言，forward 中暗改优化器参数本身正是 N8，因此没有保留这项副作用。

`log_std` 正常有限值路径原本已经采用函数式 clamp。本次也去掉了异常值分支中的原地“修复”：NaN/Inf 在当前读取中替换为 `log(0.5)`，但 checkpoint 状态保持原样，以便调用方显式发现和处理损坏参数。异常位置经 `torch.where` 选择安全常数，因此该位置梯度为 0；有限位置保持原梯度。

## 验证

新增 CPU 回归测试 `tests/test_actor_std_readonly.py`，覆盖：

1. 首次 `update_distribution` 前后完整 `state_dict` SHA-256 不变；
2. 两层边界配置下的输出与旧版依次原地 clamp 完全一致；
3. 边界内外的 `std` 梯度均为 1，明确锁定旧版局部梯度语义；
4. 旧格式 `std` checkpoint 可 `strict=True` 加载，读取后状态不变；
5. `log_std` 含 NaN/Inf 时返回安全值且不改写状态。

测试只依赖 CPU，不启动仿真、训练或 GPU 任务。

## 结论强度

- **已验证**：N8 的直接原因是 getter 内的 `clamp_`；函数式直通投影消除了读取副作用，并保持输出边界与局部梯度。
- **有证据支持**：现有 checkpoint 的轻微越界来自训练更新后参数未恰好落在配置边界，首次评估触发了隐式投影。
- **未验证**：本修复对完整 PPO 训练曲线的影响；本轮按要求不启动训练。边界外原始参数不再于 forward 中回写，长期优化器轨迹可能与旧实现不同，但策略所用标准差始终满足相同边界。
