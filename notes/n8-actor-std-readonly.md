# N8：Actor 标准差读取不应改写参数

日期：2026-09-21

## 结论

已修复 `Actor.get_std` 的读取副作用。修复前，`std` 参数化会在首次构造动作分布时执行 `self.std.clamp_()`；全量 B 的 seed0/step1000 checkpoint 因而有 25/64 个元素被原地改写，最大改变量为 `2.408e-5`。这会让只读评估改变已加载的模型状态，也会破坏依赖前后 state hash 相同的诊断。

修复后，getter 用函数式投影产生相同的有界标准差，模型参数和 `state_dict` 保持不变；训练器在明确的训练边界调用原地投影。该修改不改变 checkpoint schema，已有包含 `std` 或 `log_std` 参数的 checkpoint 可继续严格加载。

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

Actor 新增显式的 `project_std_()`。PPO trainer 在每个 rollout 之前、schedule 更新之后调用一次，并在每次 `optimizer.step()` 之后调用。这样 optimizer 持有的原始 `std` 也保持边界语义，同时写操作只发生在训练生命周期中可审计的位置。

`log_std` 正常有限值路径原本已经采用函数式 clamp。本次也去掉了异常值分支中的读取副作用：NaN/Inf 在只读评估中以 `log(0.5)` 计算，但不改 checkpoint；训练开始或 optimizer step 后的显式投影仍按旧语义把整个异常向量重置为 `log(0.5)`。异常位置在只读 fallback 中梯度为 0；有限位置保持原梯度。

## 训练调用链审查

当前 PPO 每个 batch 先进行一次 rollout，再进行 `num_ppo_epochs × num_mini_batches × num_micro_batches` 次训练 forward。一次 rollout 内有 `num_steps_per_env` 次 policy forward；同一 microbatch 在图像增强模式下还可能有两次 policy forward。它们之间都没有参数更新，因此每个 rollout 前、每次 optimizer step 后投影一次，保证下一次所有 forward 看到的参数与旧 getter 投影后相同。rollout 前投影位于 schedule 更新之后，也覆盖调度器或回调改写状态的入口。

`accelerator.accumulate()` 下代码会在每个 microbatch 调用包装后的 `optimizer.step()`；非同步 microbatch 的 step 是 no-op。其后投影在参数未更新时也是 no-op；真正更新的 step 则立即投影。梯度非法而跳过 step 时，参数此前已经处于边界内。

与旧轨迹的时序关系是：旧版在“下一次 forward 前”投影，新版在“上一次 optimizer step 后”投影。两点之间没有别的参数更新，故下一次 forward、其梯度以及下一次 optimizer 输入相同。首次训练、resume、schedule 或回调改写是额外入口，因此 trainer 在每个 rollout 前再投影一次。新版 checkpoint 会直接保存有界参数；旧版可能保存 optimizer step 后的轻微越界参数，并在 resume 后首次 forward 才投影，但二者下一次训练 forward 的状态相同。

上游 `upstream/main` 和 `upstream/zhengyi/sonic-control-fixes` 仍是 getter 内 `clamp_`，没有可直接移植的只读实现。本修复只在 Actor 增加显式投影接口，并在现有唯一 PPO `optimizer.step()` 调用点接入。

## 验证

新增 CPU 回归测试 `tests/test_actor_std_readonly.py`，覆盖：

1. 首次 `update_distribution` 前后完整 `state_dict` SHA-256 不变；
2. 两层边界配置下的输出与旧版依次原地 clamp 完全一致；
3. 边界内外的 `std` 梯度均为 1，明确锁定旧版局部梯度语义；
4. 连续六次 Adam 更新中，上界向外、下界向外两种梯度下，新版显式 post-step 投影与旧版 pre-forward 投影在每次 forward 前参数一致，optimizer 状态一致；
5. 旧格式 `std` checkpoint 可 `strict=True` 加载，读取后状态不变；
6. `log_std` 含 NaN/Inf 时只读 fallback 不改状态，显式训练投影保留旧版全向量重置语义。

测试只依赖 CPU，不启动仿真、训练或 GPU 任务。

## 结论强度

- **已验证**：N8 的直接原因是 getter 内的 `clamp_`；函数式直通投影消除了读取副作用，并保持输出边界与局部梯度。现有 trainer 的连续多步 CPU 模型验证表明显式 post-step 投影保持旧版下一次 forward 前的参数与 optimizer 状态。
- **有证据支持**：现有 checkpoint 的轻微越界来自训练更新后参数未恰好落在配置边界，首次评估触发了隐式投影。
- **未验证**：完整 PPO 训练曲线的逐步复现；本轮按要求不启动训练。DeepSpeed/FSDP 的分片参数模式未在本项目当前训练配置中启用，也未做专项运行验证。
