# general32 同观测 CPU 反事实：A201/on、A501/on，2026-10-08

**residual 的即时目标作用已复现；腕部到边界不能推出整臂退让都受限。** 同一已记录状态下，既有 nominal 自身在端点，也有 residual 将输出推到端点，另有 residual 将肩/肘拉离端点或改变内部目标。肩肘目标变化可明显大于腕部；这修正了只看腕部饱和的解释。实际跟随/负载问题仍与目标作用分开，不能由本诊断证明闭环柔顺净增益或恢复。

## 输入能否恢复、回放精度

- 12个代表tick，每run播放/末帧各3个；按右腕pitch端点/内部样本及直接输入/稀疏残差锚点选择，不是随机样本，不据此估算整段频率。只用CPUExecutionProvider，ORT1.23.2、intra/inter线程各1，复用既有venv；未安装依赖、未启动GPU、机器人、DDS、仿真，未改部署资产/参数。
- obs930字段齐全：当前及此前9个成功CONTROL tick，旧→新、term-major：angular30、q290、dq290、previous_action290、gravity30。硬件q先减原default、转换Isaac顺序；action历史保留该行“上一tick输出”，当前回放输出对齐下一CSV行。gravity用原wxyz共轭旋转，无额外归一化，历史按原stride=1而非时间近邻猜测。
- `token_state.csv`是**nominal**：源码LogPostState在ResidualEngine.Apply之前；与四份snapshot nominal_token完全一致。fused policy前64维从未当nominal使用；本轮不重算/缓存encoder，不伪造物理pelvis输入。
- CSV为9位小数，因此10个样本为精度受限重建；四个直接policy float32锚点测得obs最大差 **1.49e-8**，nominal token差0。两个选中的末次PLAY样本有直接snapshot，优先用其未受residual修改的raw suffix，属原始float32输入；另10个没有完整直接原始位串，不声称bit-exact。NPZ另留obs_reconstructed供核对，缺失的optional direct记录用NaN表示，不作为推理输入。
- 原actor ONNX内嵌checkpoint RMS、归一化clip和`clip(.15*mean,±.5)`，直接输入raw obs930，不重复归一化/.15；gain=1；decoder输入固定为cat(nominal64或nominal64+delta64, **raw** obs930)，不FSQ fused。
- 四个锚点隔离了CPU与现场TRT误差：对直接捕获policy输入，CPU raw最大差1.43e-6；CSV重建相对CPU直接输入raw最大差4.77e-7。12个样本的actor+decoder回放相对现场raw最大差 **2.3842e-6**，最终float32 q_target最大差 **1.0431e-6 rad**。四个锚点中首PLAY处处于真实warmup、delta=0，校准按原gate复现；12个主样本全部gate≥10、原on/gain1确有actor。日志全部稀疏warmup计数已对照恢复序列。
- 没有连续记录delta64，但重放的64维delta已保存。tick1150两run有稀疏delta_l2/max可独立核对；误差详见JSON，含打印精度影响。

## 12个样本

PLAY使用0-based nominal frame；FINAL_HOLD保持最后nominal frame，不能据此标为松手/无力。CONTROL_HOLD首snapshot实际仍play=true，仍归PLAY。

| run（20261007） | tick | 阶段/frame | 右腕pitch nominal→fused raw | 该tick全左/右臂最大目标变化 rad |
|---|---:|---|---:|---:|
| 174302 | 615 | PLAY/334 | -2.4828→-2.9630 | 0.2684/0.1855 |
| 174302 | 705 | PLAY/424 | -3.0000→-3.0000 | 0.5897/1.4175 |
| 174302 | 1120 | PLAY/839 | -2.8527→-3.0000 | 0.1338/0.0953 |
| 174302 | 1130 | FINAL_HOLD/839 | -2.2494→-2.8044 | 0.1028/0.1962 |
| 174302 | 1150 | FINAL_HOLD/839 | -3.0000→-3.0000 | 0.1478/0.3289 |
| 174302 | 2328 | FINAL_HOLD/839 | -2.8608→-3.0000 | 0.1186/0.1438 |
| 174626 | 720 | PLAY/381 | -1.9822→-3.0000 | 0.4922/0.3064 |
| 174626 | 726 | PLAY/387 | -0.9196→-1.8079 | 0.3247/0.5095 |
| 174626 | 1103 | PLAY/764 | -2.7001→-3.0000 | 0.0889/0.0923 |
| 174626 | 1119 | FINAL_HOLD/764 | -2.4798→-2.9582 | 0.1206/0.1019 |
| 174626 | 1150 | FINAL_HOLD/764 | -2.7433→-3.0000 | 0.0918/0.0898 |
| 174626 | 1816 | FINAL_HOLD/764 | -2.7349→-3.0000 | 0.0836/0.0980 |

## 同状态下三类作用，含肩/肘

下列nominal是**同一个on运行状态**的固定观察反事实，不是另一次off运行；目标差为fused−nominal，不是两相邻真实命令跳变量。

| run/tick | 关节SDK索引 | nominal→fused（clip后raw） | Δq_target rad | 解释 |
|---|---|---:|---:|---|
| 174302/705 | right_wrist_pitch_joint (27) | -3.000000→-3.000000 | +0.000000 | nominal本来就在同一端点，最终目标无变化 |
| 174626/720 | right_shoulder_pitch_joint (22) | -3.000000→-3.000000 | +0.000000 | nominal本来就在同一端点，不能全归residual |
| 174302/2328 | right_wrist_pitch_joint (27) | -2.860824→-3.000000 | -0.010369 | residual把内部输出推到−3 |
| 174626/1816 | right_wrist_pitch_joint (27) | -2.734932→-3.000000 | -0.019748 | residual把内部输出推到−3 |
| 174302/705 | right_shoulder_pitch_joint (22) | 3.000000→-0.231929 | -1.417451 | 右肩从+3拉离边界，目标变化明显 |
| 174302/705 | right_elbow_joint (25) | 1.023101→0.022780 | -0.438718 | 右肘内部目标变化 |
| 174626/720 | right_elbow_joint (25) | -3.000000→-2.301427 | +0.306378 | 右肘从−3拉离边界 |
| 174626/726 | right_elbow_joint (25) | -2.501723→-1.339992 | +0.509509 | 右肘内部目标变化 |
| 174626/726 | left_elbow_joint (18) | -0.301291→-1.041604 | -0.324684 | 左肘内部目标变化 |

A201/tick705右肩目标差约−1.41745rad、右肘−.43872rad；A501/tick726右肘约+.50951rad、左肘−.32468rad。它们是固定观察下即时模型目标差，**不是已实际执行的突跳，也不是推荐真机动作**。腕端点会限制该方向的进一步最终目标变化，但这里明显不能把它等同整臂没有目标作用。

## 目标变化与实际跟随分开

| run/tick/关节 | 反事实Δtarget rad | 既往held target−实际q rad | 实际dq rad/s | SDK tau_est Nm |
|---|---:|---:|---:|---:|
| 174302/705/right_shoulder_pitch_joint | -1.417451 | +0.503293 | +0.009204 | +7.2500 |
| 174626/726/right_elbow_joint | +0.509509 | -0.209436 | -2.164447 | -1.0625 |
| 174626/726/right_shoulder_pitch_joint | +0.000000 | -0.533071 | -0.013806 | -7.3750 |
| 174302/2328/right_wrist_pitch_joint | -0.010369 | -0.006708 | +0.006136 | -0.0875 |
| 174626/1816/right_wrist_pitch_joint | -0.019748 | -0.006828 | -0.006136 | -0.0875 |

A201/705右肩：即时目标作用明显，同时当时对既往writer目标有约.503rad误差；A501/726右肘也有即时目标变化并伴约.209rad既往误差。这支持“目标作用与执行/负载偏差可并存”，不能称目标变化都已被实际实现，也不能由单点推出电机吞掉指令。A501/726右肩既往误差约−.533rad，但nominal/fused都−3，此处不能把误差归因于该tick新增residual目标差。

未对齐真实人手输入、固件命令生效时刻或动态跟随滞后，所以“跟随不足”仅是已有目标—状态差描述，不是唯一硬件根因。tau_est是SDK电机估计力矩，不是外部腕力；PD代理仅由原命令/状态计算。该诊断不估刚度、admittance、力精度、撤力恢复或闭环净增益。

## 已取得与未取得的证据

29关节完整结果在 `results.json` / `joint_results.csv`：两路clip后raw、各自最终q_target、变化量、原记录raw/target、held目标与实际q/dq/tau_est。delta64、fused token64也在JSON。分类是同观察下的端点变化，不是机械限位验证。

**裁剪前值：unknown。** 原decoder接口只输出clip后的action，本轮环境没有ONNX图中间张量提取器，未修改/另导部署图获取preclip；29关节的nominal_preclip_action/fused_preclip_action均null，不能从±3反推超过多少。非snapshot tick的精确原始obs位串也缺失，仅有经锚点核对的CSV重建。施力/放手时刻、方向、力度、固件限幅状态仍unknown，未补造事件。

最小结论：下一步模型侧应同时复核腕端点、肩肘目标作用及PD负载，不能只用“腕饱和”解释整臂手感；本轮已经有同观测即时作用证据，仍需将闭环恢复/退让与这种即时作用分开。保留现有可用部署，不据本报告改限幅、PD、scale或要求重训。

## 包内文件与复现

- `samples.npz`：12份obs930/nominal64、原输出/目标、相关10帧源数据及四份校准输入。无pickle/object，未包含完整原始日志。
- `identity.json`：actor/codec/obs/config/控制参数、exact source/SHA、输入排列、精度与校准证据；actor SHA `f9ad4973f3238db1caf56a821c90fa4ce44b01aedcb9b6f8575393fea23fd107`，decoder `7e27a100cd540ea6037d2708ff87a5e2d97b79246d4c658fb3c0522b56b37a6d`。模型未装入小ZIP。
- `results.json`、`joint_results.csv`、`analyze_counterfactual.py`与校验记录。原checkpoint/global42k、gains/scales/default/map不变，原报告/ljc/旧ZIP保留。

在已有含numpy/onnxruntime的Python中，解包后只需同SHA原actor与decoder，可完全从样本重放，不需原日志/encoder/DDS：

```bash
python analyze_counterfactual.py --samples samples.npz --identity identity.json --actor /已有资产/general32_native42_residual.onnx --decoder /已有资产/sonic_original_decoder.onnx --out /独立目录/replayed
```

本轮已做这种portable重放并核对结果一致。无需为review安装模型或运行真机。
