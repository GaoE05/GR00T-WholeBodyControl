# SoftSONIC general32 现场真机结果，2026-10-07

## 已有证据支持的结论

**未见参考动作上的初步真机柔顺迁移/泛化迹象**：操作者报告 high_watering_walk_A501（15.30 s）和 lift_walk_start_A201（16.80 s）在播放和手拉时似乎有一定退让。两条唯一匹配的 on 日志确认 general32、gain=1、当前32-motion residual 训练未见 exact source，沿用原 actor，没有新增训练。放手后的恢复尚缺时间标记/视频，不能据此填 PASS。

用户还报告几个 motion 有效果、幅度存在差异；这一概括保留为 motion 层面的定性观察，未扩写成每条运行均 PASS 或同等效果。原单 motion stand/walking 的真机成功由用户确认，单独列为既有基线，不冒充 general32 的结果。

本次仅离线整理已有材料，没有启动新的 DDS、MuJoCo 或 G1 控制进程，没有修改模型、控制器算法、依赖或训练任务。新增报告分支基于远端 `codex/softsonic-target-guard-20261006` 的 `163d7f4cab0585ce16488bf596f2c8ff4a4d9f63`；报告所记现场控制器 commit 是另一套现场本地 Git 身份，不能把报告分支 HEAD 当作现场运行源码版本。

## 运行与资产身份

现场仓库 `/home/user/code/ge/SoftSONIC-controller-guard-v1`，分支 `feature/site-target-guard-v1`。当前 commit `d9d9a7f03d1454bc0358403d5766fcc3f5772d91`；本次检查 working tree clean，当前 dirty diff 为空，SHA256 为 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。各次运行时的 dirty diff 没有独立记录，标为 unknown；不得用现在 clean 代填历史 clean。

| 身份 | SHA256 / 版本 |
|---|---|
| 当前 controller binary | `7170662a5aa864627d4466ee328072be5c5cdab0c58a7ab0a3d77c4d9a3e9af7` |
| general32 actor | `f9ad4973f3238db1caf56a821c90fa4ce44b01aedcb9b6f8575393fea23fd107` |
| encoder | `19010ef7d5a89540e9b6909a94191829e64e167240e5108a8047a87a2bbb8ecc` |
| decoder | `7e27a100cd540ea6037d2708ff87a5e2d97b79246d4c658fb3c0522b56b37a6d` |
| checkpoint provenance（manifest，非本次另载 checkpoint 校验） | `dbdf998a9f2210372db958b2a98f952bd0801e0a42b4b93c03204f74359c7485` |
| gains/scales/default/mapping header | `b9332adf07c2c9b75c9b1e0756e57c7a1c2a890d8bb0aa53c3f1905fb739b791` |
| hardware limits CSV | `062784a79da100d7c0717a040c9a53de85abdfe241c4f555693dd5639825f9de` |
| native codec check binary | `b4812d54c050062d7a5aaa59aeeed4ea657e9509bae1cd80c2677efa798531c6` |

资产实际位于 `/home/user/code/ge/SoftSONIC_GuardedBundle_20261006/assets`。当前 actor/codec/obs-config 已实读核验；36 个 `run_identity.json` 的 actor 身份和 native 绑定一致，22 次 on 的 sparse residual 日志确认为 on、gain=1 且有非零输出；14 次 off 的采样 delta 为零。没有每 tick 的模型文件 SHA 或单独历史 codec 文件副本，因此历史不可超出运行身份与 native 绑定作绝对文件溯源声明。

本组记录包含三个 controller 版本：`842cab5cde281c913dd58c90491ca40aff12ccda` / binary `c7f3d83bb18835f0ef4d9668caf961bf9bcfd64bb477247ee2354a0260250d6c`；`ee1e777b6acfa83bee921be56ac1fb2e378982a5` / binary `c8bc614d129fa557dd70f7a2d126e6615c33f0740e5b502acbcdb3f49b4c2b6d`；当前 `d9d9a7f` / binary 如上。三个 commit 的 policy header SHA 相同；所有已记录 CONTROL writer 的 Kp/Kd 与当前 header float32 数值一致。参数数组、29 个 SDK/PR 关节名/索引、映射、action_scale、default_angles、Kp/Kd、CLI 与 reference SHA 均在身份 JSON / 每 run JSON 中。

现有 CPU source parity、TRT codec native PASS、general32 native golden 64 行 gain=1 PASS 复用已有真实 receipt，本次未重新运行 native 或旧错误配置矩阵。模型为 native12/history930→delta64/global42k：four-motion shared28k + general32 continuation14k，checkpoint RMS 内嵌，ONNX 已含 `clip(0.15*mean, ±0.5)`。现有 C++ gain=1 注入路径未改，不再乘 .15、不对 fused token FSQ；实时 encoder/当前物理 pelvis 输入、encoder 名称兼容、完整 Control 计时、仅准备阶段 posehold、日志仅告警与原停止/锁存流程保留。XML 活动范围不被重新当作所有 PD target 的硬阻断/clip。

`controller_playback_diff_842cab_to_d9d9a7f.patch` 保存现场两个后续播放修复提交的累计 diff（40 行新增、1 行删除）；这些不是本次报告任务新增的控制算法修改。其余现场源码身份在 JSON 中按文件 SHA 记录，完整源码/旧 ge/ljc 回退仍在现场。

## 实验盘点

共 **36 次真实硬件配置运行（22 on、14 off，0 shadow）**，覆盖 9 个 reference。判据为 `run_identity.sim=false`、hardware purpose/NIC，以及实际产生的 state/Control/writer 日志；“运行记录存在”不等于任务或柔顺效果 PASS。表中 frames/fps 是原 reference 身份，播放 ticks/事件见逐 run CSV/JSON。

| Profile（去掉 general32_ 前缀） | exact source | 当前32 split | 帧数 / ticks 时长 | off / on 日志数 |
|---|---|---|---|---|
| stand | `big_light_two_hands_hold_R_001__A508` | seen，协调端 exact-source 更正 | 847 / 16.94 s | 2 / 4 |
| walking | `230216/lift_crate_walk_ff_loop_180_R_003__A200.csv` | seen | 600 / 12.00 s | 1 / 2 |
| button_A480 | `231018/high_button_over_push_001__A480.csv` | seen | 322 / 6.44 s | 3 / 4 |
| onehand_A508 | `231114/big_light_one_hand_hold_R_001__A508.csv` | seen | 574 / 11.48 s | 2 / 3 |
| sideway_A021 | `220713/walk_sideway_045_loop_001__A021.csv` | seen | 592 / 11.84 s | 2 / 1 |
| door_open_A508 | `231114/outside_door_handle_right_side_open_full_R_002__A508.csv` | seen | 267 / 5.34 s | 2 / 4 |
| sideway_stop_A021 | `220713/walk_sideway_045_stop_001__A021.csv` | seen | 499 / 9.98 s | 0 / 2 |
| high_watering_walk_A501 | `231110/watering_plants_high_walk_ff_stop_360_R_002__A501.csv` | current32-unseen，future-full 训练候选 | 765 / 15.30 s | 1 / 1 |
| lift_walk_start_A201 | `230216/lift_crate_walk_ff_start_360_R_001__A201.csv` | current32-unseen，future-full 训练候选 | 840 / 16.80 s | 1 / 1 |

未找到真机日志的已准备 profile：`behind_twohand_pickup_A508`、`twohand_turn_walk_A508`、`onehand_turn_walk_A508`（seen），以及 `wave_walk_A492_dev`（development-unseen）。分别记为“已准备、真机未确认”，没有虚构 run。A501/A201 不能称为未来 full 的 held-out；这里的 unseen 只指当前32 residual 训练 exact source，不推断 SONIC 预训练是否见过。

历史 stand 元数据中早期“迁移/heldout”文字已依据协调端 exact source 更正为 seen，原 run 文件没有改写。A508 只是演员别名，不能据别名把 hold 与 pickup source 混为一条。

36 次都有 play-state 推进、末帧 sparse trace、随后 exact damping；这支持控制链播放/停止日志完整性，不独立证明全身追踪任务成功或没有物理跌倒。button 的部分历史运行启用了无限循环；door 最新三次启用 repeat=3。`173345` door/on 记 800 个 playing rows，而 `173529` off 与 `173630` on 记 801；早期 `165210` door/off 记 269。故逐 run 保留原计数，不以预期 `3×267` 自动改写，也不仅以 CONTROL_HOLD 标签推断播放完成（该标签在最后一帧 play=true 时也会出现）。

所有 run 的左/右手、施力方向、力大小、施力/放手区间、操作者姓名和拉动所处播放/末帧阶段均为 **unknown**。run_id 是启动目录时间，不是人手事件时间；日志 timestamp 不能代替缺失事件标记。A501/A201 各仅一条 on，可与用户 motion 级观察关联，但不伪造精确手拉时刻。

## A501/A201 的可复核结果

| 指标 | A501 on | A201 on |
|---|---|---|
| run_id | `20261007_174626_general32_high_watering_walk_A501_on` | `20261007_174302_general32_lift_walk_start_A201_on` |
| 实际 playing rows | 765 | 840 |
| 首末 playing timestamp 跨度 | 15.279947 s | 16.779781 s |
| Control P99 / max | 1.143 / 10.019 ms | 1.222 / 9.913 ms |
| 超过 20 ms 的 Control 执行 | 0 | 0 |
| q/dq/action/IMU/token 有限性 | 已记录行全部 finite | 已记录行全部 finite |
| motor_error / target fault | 0 / 0 | 0 / 0 |
| residual sparse 非零 norm 行 | 53/53 | 73/73 |
| sampled delta_max 到 .5 的行数 | 15/53 | 6/73 |
| sampled residual inference max | 0.296 ms | 0.328 ms |
| writer 无效 active / 非 exact damping / 停止后旧 active | 已记录行均 0 | 已记录行均 0 |
| target diagnostic dropped_rows 最大值 | 58 joint rows | 29 joint rows |
| 手拉退让 | 用户报告似乎有一定退让 | 用户报告似乎有一定退让 |
| 放手恢复 / 无力保持 / 物理跌倒 | unknown，缺事件/视频 | unknown，缺事件/视频 |

off 对应 `20261007_174737_general32_high_watering_walk_A501_off`、`20261007_174456_general32_lift_walk_start_A201_off`，均记完整 765/840 ticks，采样 delta=0。它们具备同 exact reference 的轨迹记录，但人手干预未知、启动姿态/阶段不能假定相同；**不作为已标注无力或等力的量化反事实对照**。

## 日志字段、采样与解释

CSV 的 `index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp` 分别为样本索引、首样本归零 ms、Unix wall-clock ms、steady-clock ms、ROS timestamp s（本组非 ROS 接口为 0 占位）。可用 index/steady clock 联合匹配，wall clock 转换为现场 Asia/Shanghai；不要把 ros_timestamp=0 当真实时刻。记录的 q median rate 为约 50 Hz；各文件行数、dt P50/P95/P99/max 和 index gap 在 JSON 中，不能只依赖 metadata.dt=.02。

| 文件 | 内容、单位、顺序与来源 | 采样/限制 |
|---|---|---|
| q.csv / dq.csv | SDK LowState 实际 q（rad）/ dq（rad/s），29 个 hardware PR/SDK index 0–28；q 已加回 default-angle，不应再次加 | Control 收到的状态快照约50Hz，非独立传感采样频率 |
| action.csv | **上一 Control tick 的 decoder raw action，IsaacLab 顺序、无量纲** | GatherRobotStateToLogger 先于本 tick Infer；不是当前 target，也不是硬件顺序缩放 offset。旧字段注释不可信，以源码为准 |
| targets_direct.csv | float32 后最终 q_target rad、dq_target rad/s、tau_ff Nm、Kp Nm/rad、Kd Nm·s/rad；hardware PR 顺序 | sent=0 是候选检查，sent=1 是 writer 发布尝试约500Hz；不等于独立 DDS 收件确认；有诊断丢行 |
| base/torso_quat, ang_vel, accel | SDK 两组 IMU，quat wxyz；gyro rad/s、accel m/s² 按接口 SI 约定解释 | 原 CSV 未附传感器标定；未独立校准、没有可靠 world root translation；不是世界坐标腕位姿 |
| motor_torque.csv | `LowState.motor_state[i].tau_est()`，hardware order，Nm 电机估计力矩 | **不是外部手拉力，不是腕部六维 wrench**；手指状态另记，actor 不负责手指 |
| motor_error.csv | SDK motor error code，hardware order，无量纲 | 全部已记录值为0；不代替物理任务/跌倒观察 |
| token_state.csv | residual 注入前的 nominal encoder token64 | 约50Hz；不是 delta 或 fused token，不证明固定 nominal 被缓存 |
| residual.csv | tick/frame/warmup、mode/gain、inference_ms、delta_l2/delta_max（无量纲 latent） | 每50ticks及warmup事件的 sparse norms/maxima；没有完整 delta64、component clamp count 或每 tick actor输出 |
| motion_playing / motion_name / encoder_mode | 实际 play 标志、reference 名、编码模式 | 约50Hz；reference frame 仅 residual/startup sparse trace，不能假称每tick frame直接记录 |
| startup_transition.jsonl | INIT/WAIT/Control 阶段 actual q/dq、raw action、final target、last_action/history、nominal/reference/PD 等诊断快照 | 稀疏记录，不能填充为连续 history930 或给放手自动标记 |
| control_loop.csv | tick/start_ns/elapsed_us/start_interval_us/completed | 完整 Control callback 计时，不是 residual 时间；不包含独立500Hz writer、异步磁盘线程 |

源码核对点：现场 `g1_deploy_softsonic.cpp` 中 GatherRobotStateToLogger、CreatePolicyCommand、Control；`state_logger.cpp` 的 appendStateToCSV_；`softsonic_residual.hpp` 与 `softsonic_target_guard.hpp`。JSON 保存各文件 SHA。映射通过两份数组互逆/29索引唯一性和模型 joint name/qpos address 验证，目标计算沿用 `float32(default_angles[hw] + raw[isaaclab_to_mujoco[hw]] * action_scale[hw])`。

36 runs 的完整 Control 汇总：113,896 个 completed rows，P50 **0.538 ms**，P99 **1.087 ms**，max **12.002 ms**，执行耗时 >20 ms 为0；回调起始间隔 max **25.053 ms**，反映调度抖动，不能宣称每 tick 严格硬实时保证。已记录 writer 没有非 finite/negative-gain/fault active，没有 CONTROL gains 差异，没有停止后旧 active 或非 exact damping。target 诊断总 dropped_rows 最大值逐 run 合计1972 joint rows，状态字段仍 finite；任何“全部 writer”结论仅限已记录行。

## 腕轨迹与可算指标的边界

已用原可用 MuJoCo 环境仅做离线 `MjModel.from_xml_path` + `mj_forward`，解析官方 `scene_43dof.xml` 的 includes/defaults，29 个实测 hardware q 按验证名称映射赋值。使用 left/right_wrist_yaw_link 相对 pelvis 的 `R_pelvis^T (p_wrist - p_pelvis)`；free-root 设模型默认，不用它冒充真实 root。XML SHA/编译后 qpos address 与29关节身份在 JSON。

36 条完整相对腕 FK CSV 保存在现场 artifact 目录。它们包含实际记录到的运动、名义参考变化及可能的人工干预，**是 FK 估计，不是视频/世界绝对误差/外力导致位移**。A501/A201 on/off JSON 包含整段播放各轴范围及有限差分速度描述统计；可以查看轨迹连续性，不能用整段范围比较柔顺强弱。放手时刻未知，暂不定义恢复区间或恢复时间；末帧后的轨迹可查看，但不能凭“停止播放”推断操作者已撤力。后续如果有事件标记，动态 motion 必须对齐相同 nominal phase。

没有真实外力传感器，不计算 cm/N、刚度、admittance、力跟踪精度；现有 off/on 虽同 reference，但无同步/等力/无力标注，不计算 residual 净增益。采样 delta 非零说明 actor 在参与，sampled max=.5 只说明该条 sparse sample 有分量到模型 clamp，不能给出全程64维饱和率。不能据这些记录声明任务成功、统一刚度、严格跨 motion 量化泛化。

## 文件、视频与接力

Git 中仅轻量材料：本报告、`general32_hardware_experiments_20261007.csv/json`、`general32_hardware_identity_20261007.json`、`general32_reference_inventory_20261007.json`、现有 native receipt 副本、小型离线分析脚本和播放 diff。每条 run 保存原文件路径/byte count/SHA；原始文件分析前后 size/mtime 都未变化。不提交 actor/codec/binary、大日志或视频。

现场原始日志：`/home/user/code/ge/SoftSONIC_GuardedBundle_20261006/logs/<run_id>/`。离线 FK、完整逐run分析及空 dirty diff：`/home/user/code/ge/guard_acceptance/hardware_results_20261007/artifacts/`。当前 profiles/launcher/shell wrapper 仍在 `/home/user/code/ge/guard_acceptance/`；实际每run完整命令在 JSON。

code/ge 中盘点到55个 AVI（含重复 archive），均来自 MuJoCo prepared simulation runs/归档。**没有找到能关联到这些真机 run 的录像/截图**；手机录像存在与否 unknown，未伪装成已收集。仿真例子 `/home/user/code/ge/guard_acceptance/runs/20261007_170623_prepared_general32_high_watering_walk_A501_on/fullbody_force.avi` 和 `20261007_170731_prepared_general32_lift_walk_start_A201_on/fullbody_force.avi` 只可标为仿真，不能做真机定性证据。现阶段代表性真机证据是两条 on 的状态/命令/时延日志和用户观察。

无需因缺指标停止既有正常真机试用。可选增强：由现场操作者选1条已见、1条未见，在近似相同 nominal phase 做 off/on 及放手观察，每组2–3次；轻量事件标记/关联手机录像即可，保持 Control 回调不做同步大文件 I/O。不需要重测全部 motion。真机撤力是操作者放手；`]` 起控、`T` 播放、`O` 停止沿现有流程，每次模型/参考/模式切换退出后重启，没有新增暂停键，不自动执行本建议。

协调 session 可引用本报告和结构化表：2条 unseen on 日志身份已确认；退让为初步定性报告，撤力恢复/精确力/严格量化迁移仍缺证据。报告分支 commit 与状态在现场接力 receipt 中记录，未通过工具向未知 session 地址自动发消息。

同步目标为用户指定的 `GaoE05/GR00T-WholeBodyControl` / `codex/general32-hardware-results-20261007`。此前 Git CLI HTTPS 因无本地凭据失败，连接器写 blob 返回 `403 Resource not accessible by integration`，现场 SSH 探测超时。用户随后明确选择本地 Git + patch/ZIP 接力：**停止本 session 的远端发布尝试，不配置 GitHub 账号，由协调 session 接包同步**。已有本地 commit 保留，尚未发布 GitHub，不提供虚构的远端 commit/PR 链接；仓库原有可见性未改。最终源 commit、patch 与 ZIP 文件清单在 ZIP 顶层 DELIVERY.json 中，接入说明见 DELIVERY_20261007.md。
