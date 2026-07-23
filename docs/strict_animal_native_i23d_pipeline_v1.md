# 动物严格版原生 I2I→I23D→动画流程 v1

机器合同：`/data/jzy/code/AVEngine/external/SPEAR/data/controlled_source_attributes_v1/contracts/strict_native_i2i_i23d_animal_v1.json`。

这条路线固定为：属性 JSON 由代码采样并生成完整 prompt；许可证明确的中性姿态几何只作为统一 clay pose guide；未蒸馏 FLUX.2 生成物种与实例外观；Pixal3D 为默认 I2I→3D；随后修复 watertight 拓扑并以 Emission bake 保留原始 PBR Base Color，TokenRig 对生成 mesh 预测目标原生骨架/权重，最后只从兼容运动家族重定向动作。Hunyuan3D 2.x、TRELLIS.2 及其输出继续只作 `technical_spike_only` 对照，不能替代默认路线的证据。

方向不再由头朝向自动估计。源资产只能通过显式的整 90° cardinal yaw 规范成“上轴 +Z、头/前进轴 +X”；动画前审核页必须同时显示 UP 与 FORWARD 箭头。猫狗源本来是 +X；Quaternius Horse 源是 -Y，因此固定加 +90°，不会再用十几度微调补偿歪头。

四足 pose guide 必须为真正正侧面，躯干水平、仅一只眼可见、四腿轮廓分离且四脚同地。远侧前/后肢由骨架拓扑与几何自动识别，只沿前后轴做物种 profile 指定的小幅反向错位；不改变相机 yaw、身体姿态或脚高。马当前使用 0.18，猫狗已验证模板使用 0.30。

目标原生骨架完成后、写入动画前，必须依次执行两个刚体规范化步骤。首先以人工审核的解剖前向把 heading 规范到整 90° cardinal axis；随后取四条语义脚链 head/tail 中更低的端点，拟合一个支撑平面，将完整 mesh+armature 层级刚体旋到世界 +Z 并把最低脚移到地面。不得单独移动脚、压平背线、改变骨骼层级或蒙皮。脚平面残差超过 mesh 对角线的 2%，或倾角超过 30°，必须拒绝而不是强行修正。

形变审计同时保留旧的世界轴 AABB 对角线比例，但自动判定使用旋转不变的 `centroid_bounding_sphere_diameter`。同一资产仅做 pitch/roll/yaw 刚体旋转时，判定尺度不得变化；不得因旋正后 AABB 变小而产生假回归。

每个随机属性最多三个离散值。所有个体是独立采样结果，不记录“从 medium 变成 light”之类的相对编辑历史。公共属性为 size、body_build、life_stage；颜色/纹理由物种 profile 定义，例如 bay horse 使用 light_bay / standard_bay / dark_bay。

必须依次通过 2D、静态 PBR 3D、动画前方向、骨架/权重、动作接触与循环、41 帧形变回读、GLB 导出回读和媒体/场景 QA。声学事件必须与物种匹配；短叫声根据阈值做带最小间隔的事件重复，并把每次事件时间写入 JSON。

当前证据边界：2026-07-23 的 Border Collie target-native 交叉验证获得项目所有者对外观、校平和 Walking 视频的研究用途视觉接受；Idle 自动形变通过，但 Walking 的旋转不变最大局部延伸比例 `0.0884` 仍高于 `0.08`。它可继续做实例属性和 Apartment research canary，但不得写成自动形变通过或正式公开数据注册完成。马仅完成了当前严格版 clay 参考图，尚未进入默认 FLUX/Pixel3D 路线。

边牧的最终研究用途验收记录为 `data/controlled_source_attributes_v1/reviews/animal_border_collie_target_native_final_20260723_v1.json`。记录同时绑定校平前后形变审计和 Idle/Walking 的侧面、正面、背面六个视频。校平使旧 AABB 尺度变化 `-7.334%`，而旋转不变尺度只变化 `+0.0061%`，因此自动判定只使用后者；人工视觉接受与严格 Walking 数值继续分开保存。
