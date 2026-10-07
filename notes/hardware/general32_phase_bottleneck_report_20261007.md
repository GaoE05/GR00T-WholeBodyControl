# general32：已有真机日志的分阶段瓶颈分析，2026-10-07

现有证据优先指向 **decoder 输出/PD 目标端长期停在策略输出边界**，不能把柔顺幅度有限统一归因于 residual 达到 ±0.5，也没有证据证明执行链普遍不响应。A201、A501 的末帧阶段没有 residual `.5` 日志采样，却仍出现腕部 raw action 长期 ±3；A201 腕部实际角接近已发布目标。肩部另有明显目标—实际误差和 PD 恢复负载，执行响应与姿态负载也可能影响手感，但没有固件饱和状态或真实人手力，不能宣布电机故障、硬件力矩饱和或量化柔顺增益。

本次只读取原来的 **36 次**真机运行：全部做 phase、稀疏 residual 和 29 关节 raw action 检索，其中优先 **9 次** onehand_A508 / A501 / A201 做 writer、新目标、实际 q/dq、SDK tau_est 的完整时间对齐。其他27次没有做 writer 响应细分析，不能将9次结果扩写为36次均相同。没有启动机器人、DDS、仿真、推理或训练；没有修改模型、gains/scales、default-angle、映射或控制代码，也没有为本次分析安装依赖。

## 身份、范围与数据字段

- actor SHA256：`f9ad4973f3238db1caf56a821c90fa4ce44b01aedcb9b6f8575393fea23fd107`；general32/native12/history930/global42k，gain=1。checkpoint `dbdf998a9f2210372db958b2a98f952bd0801e0a42b4b93c03204f74359c7485` 为既有溯源，未重新导出。
- encoder `19010ef7d5a89540e9b6909a94191829e64e167240e5108a8047a87a2bbb8ecc`；decoder `7e27a100cd540ea6037d2708ff87a5e2d97b79246d4c658fb3c0522b56b37a6d`。
- 当前现场 controller `d9d9a7f03d1454bc0358403d5766fcc3f5772d91`、binary `7170662a5aa864627d4466ee328072be5c5cdab0c58a7ab0a3d77c4d9a3e9af7`，本轮未改；历史运行各自的版本/参数快照见原逐run JSON。报告 commit 与控制器 commit 分开。
- onehand_A508 exact source 为 `231114/big_light_one_hand_hold_R_001__A508.csv`，当前32 seen，574帧/11.48s。A501 为 `231110/watering_plants_high_walk_ff_stop_360_R_002__A501.csv`，765帧/15.30s；A201 为 `230216/lift_crate_walk_ff_start_360_R_001__A201.csv`，840帧/16.80s。后两条 current32-unseen、future-full 训练候选，不是 future-full held-out。
- `q.csv` / `dq.csv` / `motor_torque.csv`：约50Hz、同一 GatherRobotStateToLogger 快照，SDK/PR 硬件29关节顺序，分别 rad / rad/s / Nm（`tau_est`）。q 日志已加回 default，不能再次加偏移。tau_est 不是外部人手力或腕部六维力；源 LowState 的真实采集时间/年龄未知。
- `action.csv`：约50Hz，IsaacLab29顺序、无量纲，**已是 decoder clip(-3,3) 后的 raw action**，不是 clip 前网络输出。每行记录上一tick输出。`motion_playing.csv` 同步记录该tick观测的播放状态。
- `targets_direct.csv`：约50Hz 的候选提交（sent=0）及约500Hz writer 发送尝试（sent=1），SDK29顺序，每个完整block为29行，含 q_target(rad)、dq_target(rad/s)、tau_ff(Nm)、Kp/Kd、sequence/stage、steady_ns。sent=1 不是独立 DDS 收件证明。
- `residual.csv`：每50 tick约1Hz的稀疏统计及 warmup 采样；只有 delta_l2/max、inference 等，没有连续delta64各分量。文本使用有限打印精度，`.5` 指日志打印的 max(abs(delta)) 等于 .5；无法据此计算连续限幅时间占比或多少 latent 维度饱和。
- 启动 INIT→WAIT 只有 `startup_transition.jsonl` 的稀疏 q/dq/target/PD 快照，没有连续 q/dq/tau/actor/decoder日志。INIT/WAIT 不执行 residual/decoder；下表的“启动”指按 ] 后 CONTROL_PREPLAY，含 warmup，不能混为 INIT 模型失败。

## 分段与时间对齐

优先9次均核对单次连续 PLAY 行数恰好等于原帧数；最后 play=true tick仍归 PLAY，即便 writer 的 stage 名为 CONTROL_HOLD。之后 play=false 才归 FINAL_HOLD；这只意味着名义末帧保持，**不等于操作者松手或无力保持**。36次概览对无法核实完整单次reference的末尾非播放段标 FINAL_NONPLAY_UNVERIFIED，循环/中断运行不伪造末帧。

原 state CSV index=i 对应 Control tick=i+1；`action.csv` index=i+1 是在 state index=i 时产生的 decoder 输出。重新排为 SDK 硬件顺序后使用 `float32(default + raw * scale)` 与当前候选/已记录writer目标核对。9次共25,881个可对齐 raw/target tick，最大差 **1.1921e-7 rad**；不纠正上一tick错位会产生约1rad的假误差。最后state tick的本次raw不在下一行日志中，明确缺失1行/每run，不补造。

实际响应使用 **不晚于 state logger steady timestamp 的最近 sent=1 目标**，避免拿未来刚生成目标与当前状态假装是已执行的响应。CSV steady time 取整不确定性±0.5µs；恰靠近writer时间的样本从跟随统计排除。LowState源采集时间未知，仍不能精确声称固件响应延迟。候选诊断丢行仅用已实际记录的同sequence writer命令补齐；sequence−Control tick=150在9次中一致，block完整/同sequence向量一致。

PD代理为 `tau_ff + Kp*(held_target-q) + Kd*(dq_target-dq)`。它是从日志命令与SDK状态计算的名义需求，没有重力/摩擦/固件内部限幅状态，不能当电机实测输出或外力估计。绝对位置误差只作描述，不将未知人手输入的off/on曲线差异当因果净增益。

## 稀疏 residual 到限幅的样本

各格为“.5采样数 / 日志采样数”；off行actor没有推理，分母仅说明日志采样量。CONTROL_PREPLAY 为启动后等待 T；PLAY 为完整播放；FINAL_HOLD 为名义末帧保持。

| run（均20261007） | profile/mode | 启动 | 播放 | 末帧保持 | 末帧记录tick数 |
|---|---|---:|---:|---:|---:|
| 141649 | onehand_A508/on | 11/13 | 9/12 | 53/83 | 4107 |
| 150609 | onehand_A508/off | 0/6 | 0/13 | 0/32 | 1543 |
| 150721 | onehand_A508/on | 2/3 | 8/12 | 35/52 | 2565 |
| 151808 | onehand_A508/off | 0/3 | 0/12 | 0/22 | 1067 |
| 151858 | onehand_A508/on | 1/4 | 8/12 | 38/53 | 2575 |
| 174302 | lift_walk_start_A201/on | 0/6 | 6/18 | 0/49 | 2406 |
| 174456 | lift_walk_start_A201/off | 0/20 | 0/17 | 0/12 | 529 |
| 174626 | high_watering_walk_A501/on | 6/7 | 9/17 | 0/29 | 1416 |
| 174737 | high_watering_walk_A501/off | 0/3 | 0/17 | 0/13 | 612 |

141649/on末帧83条采样中82条实际推理，另1条门控零输出；其余重点on采样均有推理。A201/on末帧49条delta_max在0.5以下；A501/on末帧29条同样如此。它们的delta仍非零，不应将“没有到.5”写成 actor 没效果。

进一步按**同一producer tick**核对：A201/on末帧49个residual采样的max范围.379319–.425205，其中右腕pitch在48个采样tick同时raw=−3；A501/on末帧29个采样的max范围.294154–.373920，其中28个采样tick右腕pitch同时raw=−3。两条在tick1150分别记录delta_max=.421886/.352386、raw=−3、new target=−.223503rad。这直接显示若干同tick decoder端点并不要求latent先到.5；仍不能排除未采样tick的latent限幅。对应计数保存在JSON每phase residual的 `same_tick_residual_below_limit_raw_endpoints`。

## raw action ±3：关节、符号及阶段

关节简写均为 SDK/PR：LP=左腕pitch(20)、LY=左腕yaw(21)、SP=右肩pitch(22)、SR=右肩roll(23)、RP=右腕pitch(27)、RY=右腕yaw(28)。下表是**可观测raw行**的端点占比%，分母/正负计数/首tick/连续段长在JSON中；未出现记0。跟随CSV另排除了时间取整歧义样本，分母可能略不同。

| run/mode | 阶段 | LP20 | LY21 | SP22 | SR23 | RP27 | RY28 |
|---|---|---:|---:|---:|---:|---:|---:|
| 141649/on | 启动 | 96.0 | 92.6 | 99.0 | 87.4 | 96.8 | 98.2 |
| 141649/on | 播放 | 97.9 | 67.8 | 99.7 | 16.7 | 98.3 | 100.0 |
| 141649/on | 末帧 | 91.5 | 66.9 | 97.1 | 18.4 | 99.3 | 99.4 |
| 150609/off | 启动 | 0 | 0 | 96.9 | 0.3 | 0 | 95.9 |
| 150609/off | 播放 | 0 | 0 | 65.5 | 0 | 0 | 100.0 |
| 150609/off | 末帧 | 0 | 0 | 86.1 | 0 | 0 | 100.0 |
| 150721/on | 启动 | 80.4 | 65.4 | 92.5 | 1.9 | 83.2 | 89.7 |
| 150721/on | 播放 | 96.9 | 59.1 | 84.7 | 7.5 | 98.4 | 96.0 |
| 150721/on | 末帧 | 97.8 | 67.6 | 98.6 | 5.2 | 97.5 | 100.0 |
| 151808/off | 启动 | 0 | 0 | 91.2 | 1.0 | 0 | 87.3 |
| 151808/off | 播放 | 0 | 0 | 74.6 | 0 | 0 | 100.0 |
| 151808/off | 末帧 | 0 | 0 | 95.7 | 0.2 | 0 | 100.0 |
| 151858/on | 启动 | 82.6 | 65.8 | 96.1 | 0 | 87.1 | 92.9 |
| 151858/on | 播放 | 96.2 | 68.5 | 99.7 | 0 | 98.1 | 100.0 |
| 151858/on | 末帧 | 93.4 | 77.2 | 100.0 | 1.2 | 98.6 | 99.5 |
| 174302/on | 启动 | 90.4 | 91.8 | 0 | 0 | 90.7 | 89.3 |
| 174302/on | 播放 | 89.0 | 48.3 | 0 | 0 | 94.8 | 78.7 |
| 174302/on | 末帧 | 99.5 | 98.4 | 0 | 0 | 99.6 | 99.3 |
| 174456/off | 启动 | 0 | 33.5 | 0 | 0 | 0 | 0 |
| 174456/off | 播放 | 0 | 13.7 | 0 | 0 | 0 | 20.1 |
| 174456/off | 末帧 | 0 | 0 | 0 | 0 | 0 | 12.5 |
| 174626/on | 启动 | 90.2 | 92.6 | 98.8 | 97.3 | 34.3 | 93.8 |
| 174626/on | 播放 | 68.2 | 57.3 | 75.0 | 75.9 | 29.0 | 85.4 |
| 174626/on | 末帧 | 54.6 | 0 | 0 | 0 | 98.9 | 0 |
| 174737/off | 启动 | 10.8 | 49.3 | 97.3 | 93.9 | 83.8 | 89.9 |
| 174737/off | 播放 | 56.6 | 30.1 | 75.3 | 73.6 | 73.9 | 85.4 |
| 174737/off | 末帧 | 0 | 0 | 0 | 0 | 0 | 0 |

重点符号与时间：A508/on 的双腕pitch、右腕yaw、右肩pitch主要 **−3**；左腕yaw常为+3，部分阶段也出现−3，不能把正负翻转的总端点比例称为同一固定目标。A201/on末帧左腕pitch、右腕pitch/yaw为−3，左腕yaw主要+3；A501/on末帧双腕pitch为−3。A501/off播放时双腕pitch为+3且右肩pitch/roll主要−3，和on的目标方向不完全一样。

A508/on 首右肩pitch−3在 tick4 已发生，off也是 tick4，均早于T播放；腕pitch随后在启动CONTROL就到−3。A201/on末帧右腕pitch最长连续−3为2395 tick（47.90s），左腕pitch2392 tick（47.84s），左腕yaw+3最长2351 tick（47.02s）。A501/on末帧右腕pitch连续−3为1399 tick（27.98s）。A508/141649/on末帧右腕pitch连续−3为2760 tick（55.20s）。这些都是输出端点持续时间，不是人手持续施力时间。

**baseline本来就有的饱和**：A508/off的右腕yaw在播放/末帧约100%为−3，右肩pitch也大量−3；A501/off/on播放右肩pitch约75%，右肩roll约74–76%。因此不能把所有端点归给general32 residual。另一方面，A508/off双腕pitch没有±3，而on在播放/末帧约92–99%；A201/off双腕pitch无±3，而on末帧约99.5%。这提示actor注入与decoder/闭环状态相互作用值得优先复核，不等于已经通过同一观测反事实证明actor造成全部变化。

±3是网络导出策略边界，不是机械限位。腕pitch/yaw经现有scale/default后端点约±0.223503rad，右肩pitch −3对应−1.115732rad（default=.2）。在同方向端点上，进一步增加decoder未裁剪输出不能继续改变该关节最终目标；反方向仍可变化。没有 clip 前输出，不能量化丢失了多少目标变化，也不应扩大边界或全局clip“修复”。

## 最终目标与实际响应

下面均为 FINAL_HOLD，误差=最近已记录writer目标−同次SDK实际q，报告绝对误差 p50/p95。不是未来新目标的误差，不带未知人手力归一化。

| run/mode | 关节(hw) | absolute error p50/p95 rad | held target std rad | actual q std rad | tau_est min/max Nm |
|---|---|---:|---:|---:|---:|
| 141649/on | right_shoulder_pitch_joint (22) | 0.2843/0.7818 | 0.0159 | 0.1895 | -12.19/0.62 |
| 141649/on | right_wrist_pitch_joint (27) | 0.0040/0.0792 | 0.0063 | 0.0276 | -1.63/0.43 |
| 150609/off | right_shoulder_pitch_joint (22) | 0.3169/0.8134 | 0.0591 | 0.2099 | -13.94/5.44 |
| 150609/off | right_wrist_pitch_joint (27) | 0.0157/0.0417 | 0.0785 | 0.0900 | -1.18/0.64 |
| 150721/on | right_shoulder_pitch_joint (22) | 0.2963/0.8619 | 0.0070 | 0.2560 | -13.62/5.25 |
| 150721/on | right_wrist_pitch_joint (27) | 0.0038/0.0348 | 0.0097 | 0.0159 | -0.94/0.98 |
| 151808/off | right_shoulder_pitch_joint (22) | 0.3044/0.8702 | 0.0164 | 0.2363 | -14.44/2.94 |
| 151808/off | right_wrist_pitch_joint (27) | 0.0074/0.0241 | 0.0355 | 0.0414 | -0.91/0.59 |
| 151858/on | right_shoulder_pitch_joint (22) | 0.2890/0.5865 | 0.0005 | 0.2241 | -13.31/7.12 |
| 151858/on | right_wrist_pitch_joint (27) | 0.0034/0.0875 | 0.0031 | 0.0277 | -1.82/1.56 |
| 174302/on | left_wrist_pitch_joint (20) | 0.0008/0.0093 | 0.0074 | 0.0075 | -0.39/0.56 |
| 174302/on | left_wrist_yaw_joint (21) | 0.0104/0.0104 | 0.0098 | 0.0079 | -0.20/0.17 |
| 174302/on | right_shoulder_pitch_joint (22) | 0.1294/0.1411 | 0.0619 | 0.0553 | -6.94/3.62 |
| 174302/on | right_shoulder_roll_joint (23) | 0.1332/0.1428 | 0.0296 | 0.0251 | -5.31/0.06 |
| 174302/on | right_wrist_pitch_joint (27) | 0.0067/0.0067 | 0.0024 | 0.0018 | -0.44/0.29 |
| 174302/on | right_wrist_yaw_joint (28) | 0.0085/0.0085 | 0.0018 | 0.0008 | -0.18/0.15 |
| 174456/off | left_wrist_pitch_joint (20) | 0.0088/0.0123 | 0.0052 | 0.0025 | -0.28/0.18 |
| 174456/off | left_wrist_yaw_joint (21) | 0.0056/0.0103 | 0.0082 | 0.0046 | -0.12/0.16 |
| 174456/off | right_shoulder_pitch_joint (22) | 0.1372/0.1710 | 0.0172 | 0.0021 | -3.00/-1.31 |
| 174456/off | right_shoulder_roll_joint (23) | 0.1465/0.1553 | 0.0098 | 0.0001 | -3.50/-1.94 |
| 174456/off | right_wrist_pitch_joint (27) | 0.0032/0.0055 | 0.0029 | 0.0007 | -0.18/0.14 |
| 174456/off | right_wrist_yaw_joint (28) | 0.0035/0.0129 | 0.0134 | 0.0141 | -0.26/0.16 |
| 174626/on | left_wrist_pitch_joint (20) | 0.0115/0.0161 | 0.0309 | 0.0240 | -0.36/0.28 |
| 174626/on | left_wrist_yaw_joint (21) | 0.0079/0.0108 | 0.0233 | 0.0223 | -0.26/0.19 |
| 174626/on | right_shoulder_pitch_joint (22) | 0.0195/0.0250 | 0.0225 | 0.0083 | -1.06/1.75 |
| 174626/on | right_shoulder_roll_joint (23) | 0.1290/0.1362 | 0.0183 | 0.0032 | -2.19/-1.06 |
| 174626/on | right_wrist_pitch_joint (27) | 0.0068/0.0070 | 0.0078 | 0.0063 | -0.24/0.20 |
| 174626/on | right_wrist_yaw_joint (28) | 0.0070/0.0168 | 0.0429 | 0.0439 | -0.29/0.39 |
| 174737/off | left_wrist_pitch_joint (20) | 0.0038/0.0122 | 0.0140 | 0.0107 | -0.12/0.19 |
| 174737/off | left_wrist_yaw_joint (21) | 0.0028/0.0082 | 0.0209 | 0.0191 | -0.08/0.18 |
| 174737/off | right_shoulder_pitch_joint (22) | 0.0056/0.0127 | 0.0080 | 0.0004 | -0.06/0.44 |
| 174737/off | right_shoulder_roll_joint (23) | 0.1087/0.1105 | 0.0029 | 0.0001 | -1.81/-1.50 |
| 174737/off | right_wrist_pitch_joint (27) | 0.0139/0.0152 | 0.0243 | 0.0259 | -0.04/0.29 |
| 174737/off | right_wrist_yaw_joint (28) | 0.0099/0.0101 | 0.0028 | 0.0001 | 0.03/0.23 |

A201/on末帧四个腕pitch/yaw的误差中位数约0.0008–0.0104rad，右腕pitch目标std约.0024、实际std约.0018rad：长期端点对应实际也接近目标，不能归为“网络给了变化但电机完全不执行”。同run的肩pitch/roll误差中位数约.08–.15rad，和腕不同；这可兼有负载/PD/姿态影响，不能用腕结果代替全臂结论。

A508/on末帧右肩pitch目标std仅.0005–.0159rad，实际std .1895–.2560rad，误差中位数.2843–.2963rad；off也有约.304–.317rad中位误差。这说明实际关节在近端点目标下仍在动并承受恢复PD，不能仅凭位移称actor主动退让，或因为有目标误差就断言故障。物理载荷与人为施力事件没有区分。

下面三个是**数值事件**，不是已确认的手拉/松手。raw为该tick新生成输出；held是此前writer目标，因此刚首次到−3时两者不同属正常时间关系。

| run | 阶段/事件/关节 | tick/nominal frame | raw | new target / held target / actual q rad | dq rad/s | tau_est / nominal PD Nm |
|---|---|---:|---:|---:|---:|---:|
| 174302 | FINAL_HOLD/first_raw_minus3/right_wrist_pitch_joint | 1131/839 | -3 | -0.223503/-0.208927/-0.183658 | -0.024544 | -0.3688/-0.3977 |
| 174626 | FINAL_HOLD/first_raw_minus3/right_wrist_pitch_joint | 1120/764 | -3 | -0.223503/-0.220389/-0.174550 | -0.595185 | -0.1063/-0.1334 |
| 141649 | FINAL_HOLD/largest_held_target_error/right_shoulder_pitch_joint | 3786/573 | -3 | -1.115732/-1.115732/-0.247774 | 0.026078 | -12.1250/-12.3926 |

141649右肩例子的Kp=14.250623、Kd=.907223，名义PD≈−12.3926Nm、tau_est≈−12.125Nm；计算出的PD需求和tau_est常高度相关。但tau_est是SDK估计，不是独立实测/外力通道。没有固件实际命令、饱和标志或可靠外力，所以不宣布达到力矩极限，不用tau估算腕部N或cm/N。

## Control时延与瓶颈判断

9次已完成Control的P99约0.721–1.281ms，最大10.3694ms，没有已记录执行耗时>20ms；A501/on最大10.0189ms、A201/on9.91259ms。9次start interval最大23.9801ms，显示调度间隔另有抖动，不能把执行时间小于周期写成硬实时无抖动。全36次计时复用原报告：113,896完成回调，P99=1.087ms/max=12.002ms，start interval最大25.053ms。现有数据不支持把主要手感问题归为持续Control算力超时。

按可见证据排序：

1. **直接可见的目标端瓶颈**：腕/肩长期decoder后clip端点；部分阶段没有residual `.5`采样仍有目标端点，同tick稀疏配对已见latent max显著低于.5而decoder仍到±3，latent限幅不能作为唯一解释；未采样tick是否限幅仍未知。
2. **需复核模型—decoder交互**：on相对off的部分腕目标方向/端点频率变化明显，A508出现更多腕pitch端点；baseline也有肩/腕yaw端点。现有off/on是不同闭环状态和未知人为输入，不能严格分离actor、nominal codec与观察分布，也未运行固定obs的counterfactual decoder重放。
3. **执行/负载并存**：A201腕能贴近已发布目标，排除这部分“普遍执行不动”的解释；肩部存在较大PD误差、恢复负载，两者可一起限制手拉体验。没有外力和驱动状态，不能把肩误差唯一定位为驱动限幅、摩擦、重力或施力。

最小处理建议是先将本报告的端点关节/phase交给模型与codec维护方，复核同一已记录观察下nominal/fused token经过decoder后的端点与目标变化（若原观察能准确重建，明确标离线反事实；本轮尚未执行）。先保留原策略/参数与实际可用部署，不通过加大gain/scale、扩±3/XML范围或改PD去“解锁”，也不把本报告当停止正常现场试用或必须重训的门槛。未知输入时先做描述性对照，不计算残差净增益、刚度或admittance。

## 可复核文件、复现与缺失项

- `general32_phase_analysis_20261007.json`：36次分阶段/29关节端点与residual采样；9次详细对齐audit和跟随摘要；源日志SHA与来源。
- `general32_phase_joint_metrics_20261007.csv`：783行=9run×29关节×3phase，含目标/实际角分布、dq、tau_est、PD代理；时间歧义排除后的分母明确列出。
- `general32_phase_numeric_events_20261007.csv`：首±3及最大已发布目标误差的数值样本、tick/phase/frame、new/held target、q/dq、tau_est、PD。全部human_event=unknown，不作为真实受力事件。
- 现场完整分布和九个aligned NPZ保留在 `/home/user/code/ge/guard_acceptance/hardware_phase_analysis_20261007/artifacts/`；其SHA/path在JSON中，不装入ZIP。原36run日志仍在 `SoftSONIC_GuardedBundle_20261006/logs/`，视频位置沿 `EVIDENCE_LOCATIONS_20261007.md`；没有新真机视频/截图或施力事件标注。
- 原身份、实验表、报告、native receipts与源码diff附件仍保留。encoder/timing/posehold/日志警告修复保持，未触碰ljc或训练任务。

复现仅需现场已有numpy与原绝对路径数据：

```bash
python3 /home/user/code/ge/SoftSONIC-hardware-results-20261007/notes/hardware/analyze_general32_phases.py --report-dir /home/user/code/ge/SoftSONIC-hardware-results-20261007/notes/hardware --artifacts /home/user/code/ge/guard_acceptance/hardware_phase_analysis_20261007/artifacts
```

此脚本先核验原库存SHA、只做文件I/O和数组运算，不加载ONNX、GPU或DDS。导出契约证据为当前 `assets/export_core_manifest.json`，SHA `af52f449b435724704d14e7ed586e20059bf593f381846edbb0679f365787e8d`：decoder输入cat(token64,raw_obs930)，输出已clip±3；residual输出已clip(.15*actor,±.5)。不能把token顺序反过来作未来重放。

仍unknown：施力/放手时刻、左右手/方向/力度、真正无力阶段、恢复事件、clip前decoder值、每tick64维delta及归一化输入clip情况、固件已施加/饱和力矩、LowState源采集年龄、真机视频关联。未代填恢复PASS或每条效果均PASS。
