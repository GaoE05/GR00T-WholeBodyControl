# 现场证据位置（大文件不打包）

此清单仅列位置和证据类型，不将仿真视频当作真机视频。原日志文件的 bytes/SHA256 见实验 JSON 每 run 的 evidence。

## 真机原始日志

根目录：`/home/user/code/ge/SoftSONIC_GuardedBundle_20261006/logs/`。以下36次 sim=false/hardware 记录；完整命令、actor/native绑定、reference文件SHA均已保存。

| run_id | mode | profile |
|---|---|---|
| 20261007_123242_general32_stand_on | on | general32_stand |
| 20261007_123408_general32_walking_on | on | general32_walking |
| 20261007_130954_general32_button_A480_on | on | general32_button_A480 |
| 20261007_131916_general32_button_A480_on | on | general32_button_A480 |
| 20261007_141649_general32_onehand_A508_on | on | general32_onehand_A508 |
| 20261007_142524_general32_sideway_A021_off | off | general32_sideway_A021 |
| 20261007_142738_general32_stand_on | on | general32_stand |
| 20261007_150508_general32_button_A480_off | off | general32_button_A480 |
| 20261007_150609_general32_onehand_A508_off | off | general32_onehand_A508 |
| 20261007_150721_general32_onehand_A508_on | on | general32_onehand_A508 |
| 20261007_150926_general32_stand_off | off | general32_stand |
| 20261007_151013_general32_stand_on | on | general32_stand |
| 20261007_151223_general32_button_A480_off | off | general32_button_A480 |
| 20261007_151317_general32_button_A480_on | on | general32_button_A480 |
| 20261007_151556_general32_stand_off | off | general32_stand |
| 20261007_151701_general32_stand_on | on | general32_stand |
| 20261007_151808_general32_onehand_A508_off | off | general32_onehand_A508 |
| 20261007_151858_general32_onehand_A508_on | on | general32_onehand_A508 |
| 20261007_152121_general32_walking_off | off | general32_walking |
| 20261007_152252_general32_walking_on | on | general32_walking |
| 20261007_165210_general32_door_open_A508_off | off | general32_door_open_A508 |
| 20261007_165319_general32_door_open_A508_on | on | general32_door_open_A508 |
| 20261007_165357_general32_door_open_A508_on | on | general32_door_open_A508 |
| 20261007_165547_general32_sideway_stop_A021_on | on | general32_sideway_stop_A021 |
| 20261007_165829_general32_sideway_stop_A021_on | on | general32_sideway_stop_A021 |
| 20261007_173345_general32_door_open_A508_on | on | general32_door_open_A508 |
| 20261007_173529_general32_door_open_A508_off | off | general32_door_open_A508 |
| 20261007_173630_general32_door_open_A508_on | on | general32_door_open_A508 |
| 20261007_174302_general32_lift_walk_start_A201_on | on | general32_lift_walk_start_A201 |
| 20261007_174456_general32_lift_walk_start_A201_off | off | general32_lift_walk_start_A201 |
| 20261007_174626_general32_high_watering_walk_A501_on | on | general32_high_watering_walk_A501 |
| 20261007_174737_general32_high_watering_walk_A501_off | off | general32_high_watering_walk_A501 |
| 20261007_174901_general32_button_A480_on | on | general32_button_A480 |
| 20261007_175229_general32_button_A480_off | off | general32_button_A480 |
| 20261007_175735_general32_sideway_A021_on | on | general32_sideway_A021 |
| 20261007_175931_general32_sideway_A021_off | off | general32_sideway_A021 |

代表性 unseen on：

- `20261007_174626_general32_high_watering_walk_A501_on`，actor general32/gain1；765ticks。
- `20261007_174302_general32_lift_walk_start_A201_on`，actor general32/gain1；840ticks。

每目录的 q/dq、IMU、action、token、residual、targets_direct、startup_transition、control_loop、run_identity 等仍保留现场；原文件未放进ZIP。

## 离线分析/FK

`/home/user/code/ge/guard_acceptance/hardware_results_20261007/artifacts/`：每 run `_analysis.json`、`_pelvis_wrist_fk.csv` 和空 `controller_dirty_current.patch`。完整FK数据是官方模型 pelvis 相对腕部估计，不是世界位移、录像或外力测量。CSV路径/数据SHA与各run的摘要在实验JSON中。

## 录像/截图

真机视频/截图：unknown / 本次在 code/ge 中未找到可与这些 run 关联的文件；手机未导入，不能补拍摄时刻或施力/松手事件。下面55项均为 MuJoCo prepared runs 与重复仿真归档，只供仿真回看，不能替代真机证据。未扫描训练任务或训练数据。

```text
/home/user/code/ge/guard_acceptance/evidence/reference_next_20261007/20261007_130403_prepared_general32_button_A480_on/simulation/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/evidence/reference_next_20261007/20261007_130403_prepared_general32_button_A480_on/simulation/fullbody_force.avi
/home/user/code/ge/guard_acceptance/evidence/repeat3_20261007/door/on/simulation/fullbody_force.avi
/home/user/code/ge/guard_acceptance/evidence/sideway_20261007/20261007_142238_prepared_general32_sideway_A021_on/simulation/fullbody_force.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162246_prepared_general32_twohand_turn_walk_A508_on/simulation/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162246_prepared_general32_twohand_turn_walk_A508_on/simulation/fullbody_force.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162005_prepared_general32_door_open_A508_on/simulation/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162005_prepared_general32_door_open_A508_on/simulation/fullbody_force.avi
/home/user/code/ge/guard_acceptance/evidence/onehand_20261007/20261007_141226_prepared_general32_onehand_A508_on/simulation/fullbody_force.avi
/home/user/code/ge/guard_acceptance/evidence/loop_upgrade_20261007/simulation/fullbody_force.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_163315_prepared_general32_behind_twohand_pickup_A508_on/simulation/fullbody_force.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_163024_prepared_general32_lift_walk_start_A201_on/simulation/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_163024_prepared_general32_lift_walk_start_A201_on/simulation/fullbody_force.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162126_prepared_general32_behind_twohand_pickup_A508_on/simulation/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162126_prepared_general32_behind_twohand_pickup_A508_on/simulation/fullbody_force.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162702_prepared_general32_wave_walk_A492_dev_on/simulation/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162702_prepared_general32_wave_walk_A492_dev_on/simulation/fullbody_force.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162841_prepared_general32_high_watering_walk_A501_on/simulation/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162841_prepared_general32_high_watering_walk_A501_on/simulation/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_172530_prepared_general32_twohand_turn_walk_A508_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162246_prepared_general32_twohand_turn_walk_A508_on/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162246_prepared_general32_twohand_turn_walk_A508_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_130403_prepared_general32_button_A480_on/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/runs/20261007_130403_prepared_general32_button_A480_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162005_prepared_general32_door_open_A508_on/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162005_prepared_general32_door_open_A508_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_163315_prepared_general32_behind_twohand_pickup_A508_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162535_prepared_general32_onehand_turn_walk_A508_on/simulation/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162535_prepared_general32_onehand_turn_walk_A508_on/simulation/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_142238_prepared_general32_sideway_A021_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_163024_prepared_general32_lift_walk_start_A201_on/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/runs/20261007_163024_prepared_general32_lift_walk_start_A201_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_170851_prepared_general32_twohand_turn_walk_A508_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_170731_prepared_general32_lift_walk_start_A201_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_170058_prepared_general32_wave_walk_A492_dev_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162126_prepared_general32_behind_twohand_pickup_A508_on/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162126_prepared_general32_behind_twohand_pickup_A508_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162702_prepared_general32_wave_walk_A492_dev_on/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162702_prepared_general32_wave_walk_A492_dev_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162841_prepared_general32_high_watering_walk_A501_on/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162841_prepared_general32_high_watering_walk_A501_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_170455_prepared_general32_behind_twohand_pickup_A508_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_175611_prepared_general32_sideway_stop_A021_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162535_prepared_general32_onehand_turn_walk_A508_on/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162535_prepared_general32_onehand_turn_walk_A508_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_141226_prepared_general32_onehand_A508_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162410_prepared_general32_sideway_stop_A021_on/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/runs/20261007_162410_prepared_general32_sideway_stop_A021_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_170623_prepared_general32_high_watering_walk_A501_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_172347_prepared_general32_behind_twohand_pickup_A508_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_170326_prepared_general32_wave_walk_A492_dev_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_172205_prepared_general32_door_open_A508_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/runs/20261007_131610_prepared_general32_button_A480_on/fullbody_force.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162410_prepared_general32_sideway_stop_A021_on/simulation/fullbody_force_until_stop.avi
/home/user/code/ge/guard_acceptance/evidence/dynamic_merge_20261007/runs/20261007_162410_prepared_general32_sideway_stop_A021_on/simulation/fullbody_force.avi
```

真机施力/放手时间、左右手、方向、外力大小未记录。回看这些仿真录像不能给真机补标受力/恢复。


## 本次分阶段响应分析

`/home/user/code/ge/guard_acceptance/hardware_phase_analysis_20261007/artifacts/`：九个重点run `_aligned.npz`、`_detail.json` 与 `phase_analysis_full.json`，完整描述统计与时间对齐数组。Git/ZIP中的 `general32_phase_analysis_20261007.json` 指明路径、SHA和原日志SHA。三个profile共9次：onehand_A508的141649on/150609off/150721on/151808off/151858on，A201的174302on/174456off，A501的174626on/174737off（均20261007）。

数组保留SDK/PR顺序raw/current-new-target/as-of-writer-target/actual q/dq/tau_est/PD代理和phase/timestamp；不含外力或施力/放手标注。其他27次只有phase/raw/residual统计，不声称均做了writer对齐。所有原日志和既有视频位置不变；本次未生成真机录像。
