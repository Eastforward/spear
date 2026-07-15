# 动物严格版原生 I2I→I23D→动画流程 v1

机器合同：`/data/jzy/code/AVEngine/external/SPEAR/data/controlled_source_attributes_v1/contracts/strict_native_i2i_i23d_animal_v1.json`。

这条路线固定为：属性 JSON 由代码采样并生成完整 prompt；许可证明确的中性姿态几何只作为统一 clay pose guide；FLUX.2 生成物种与实例外观；Pixal3D 为默认 I2I→3D，TRELLIS.2 为同图对照；随后对生成 mesh 预测匹配的骨架/权重并使用同运动家族的原生动作。Hunyuan3D 2.x 及输出继续只作 `technical_spike_only`。

方向不再由头朝向自动估计。源资产只能通过显式的整 90° cardinal yaw 规范成“上轴 +Z、头/前进轴 +X”；动画前审核页必须同时显示 UP 与 FORWARD 箭头。猫狗源本来是 +X；Quaternius Horse 源是 -Y，因此固定加 +90°，不会再用十几度微调补偿歪头。

四足 pose guide 必须为真正正侧面，躯干水平、仅一只眼可见、四腿轮廓分离且四脚同地。远侧前/后肢由骨架拓扑与几何自动识别，只沿前后轴做物种 profile 指定的小幅反向错位；不改变相机 yaw、身体姿态或脚高。马当前使用 0.18，猫狗已验证模板使用 0.30。

每个随机属性最多三个离散值。所有个体是独立采样结果，不记录“从 medium 变成 light”之类的相对编辑历史。公共属性为 size、body_build、life_stage；颜色/纹理由物种 profile 定义，例如 bay horse 使用 light_bay / standard_bay / dark_bay。

必须依次通过 2D、静态 PBR 3D、动画前方向、骨架/权重、动作接触与循环、41 帧形变回读、GLB 导出回读和媒体/场景 QA。声学事件必须与物种匹配；短叫声根据阈值做带最小间隔的事件重复，并把每次事件时间写入 JSON。

当前证据边界：猫/狗的 200k PBR 权重修复视频被用户评价为“明显好很多”，但尚未获得最终 formal approval；马仅完成了当前严格版 clay 参考图，尚未进入 FLUX/Pixal/TRELLIS。不得把这些状态写成正式数据注册完成。
