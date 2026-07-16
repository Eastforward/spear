# 动物资产无 seed 抽奖工作流

状态：2026-07-16 起的新动物基础资产与实例扩增强制执行。

机器合同：

- `data/controlled_source_attributes_v1/contracts/animal_one_shot_no_seed_lottery_v1.json`
- `data/controlled_source_attributes_v1/contracts/strict_native_i2i_i23d_animal_v2.json`

## 1. 核心边界

FLUX.2 和 Pixal3D 是确定 seed 下可复现的生成模型，但 prompt 本身不能证明任意
输入都必然产生拓扑、四肢和绑定完全正确的模型。因此“稳定”不能定义为失败后
换 seed 直到看起来合格，而要定义为：请求在运行前冻结；每个阶段只执行一次；
所有失败都计入；生产实例不重复生成几何。

seed 在本流程中只是复现标识，不是搜索旋钮。禁止 best-of-N、seed sweep、看到
结果后替换请求、隐藏失败样本或只汇报成功候选。

## 2. 两阶段路线

### 2.1 新物种/新品种基础资产获取

一个 `flux2_pixal3d_animal_v1` profile 只能描述一个冻结的基础资产，所有
`sampled_attribute_domains` 都必须是单值。其执行顺序固定为：

```text
许可证明确的 pose guide + 单值完整属性 JSON
  -> 冻结 request_sha256 / generation_seed / prompt / 模型 revision
  -> 一次 FLUX.2，恰好一张图
  -> 对这一张图做完整 2D hard gate
  -> 一次 Pixal3D，恰好一个 PBR GLB
  -> 静态拓扑/闭合/四肢/落地/方向 gate
  -> 一次匹配骨架和权重
  -> 物种/运动家族对应的 Walk/Idle 或等价动作
  -> 形变、脚接触、循环、GLB 回读和媒体 QA
```

任何一步失败，都保存该请求的输入、输出、日志、hash 和失败原因，并把该实例记为
`rejected`。若失败是系统性的，只能修改 pose guide、prompt 合同、确定性几何步骤
或绑定算法并发布新的 profile revision；不能只换 seed。

### 2.2 已批准基础资产上的生产实例

基础资产通过后，生成一个 `stable_animal_template_v1` profile，冻结 mesh、UV、
PBR、skin、skeleton、FRONT 和动作 hash。普通实例不再调用 FLUX 或 Pixal：

| 属性 | 稳定实现 |
|---|---|
| `size` | 按绝对档位执行 actor/root scale，并回读真实厘米 |
| `coat_tone` / pattern | 语义 mask 上的确定性材质或纹理变换，保留 PBR 细节 |
| `body_build` | 只在预审计语义顶点组做有界、保拓扑形变 |
| `life_stage` | 预审计头/躯干形变和毛色迹象；不改骨架拓扑 |
| 声学属性 | 从物种许可池确定性选择事件类、时长与重复策略 |

因此，颜色、大小、体型和年龄的批量实例不会重新遇到 Pixal 的并腿、腹部缺面、
方向漂移或重新绑定问题。同一骨架/运动家族的锁脚动作可直接继承；只有新的几何
家族才重新走一次基础资产获取。

## 3. prompt 与姿态模板必须同时约束

基础资产 profile 必须显式声明并在正/负 prompt 中保持：

- pose guide、相机方位和视图类型；
- `camera_roll=0`、水平地面与 horizon；
- 投影躯干/脊柱轴笔直且水平；
- 四条完整且相互分离的腿和脚；
- 单条尾巴，与后腿留出背景间隙；
- 封闭腹部；
- 非目标解剖、姿势、相机和背景全部锁定；
- photorealistic/PBR，禁止插画、动画、玩具或 clay 风格泄漏。

三分之四 pose guide 可以增加远侧腿可见性，但其 I2-3D 原始 yaw 本来就不是整
90 度。该角度必须写入 `source_view_contract`，并只在“新基础资产一次性方向门”
中校正躯干轴；不得用头部朝向自动推断。基础资产批准后，所有实例继承同一已
冻结的方向变换，不逐实例人工调整。

## 4. 运行时硬约束

执行器现在会在模型加载前校验：

- policy 文件的路径与 SHA-256；
- 每个 FLUX job 只有一个 consumer、一个 seed、一次 invocation；
- FLUX 返回图像数量必须恰好为一；
- Pixal seed 必须等于冻结 request 的 generation seed；
- Pixal `attempt_ordinal` 必须为 0；
- 动态 GPU claim 对每个 instance 恰好一次；
- 清单明确记录 `seed_retry_allowed=false` 和
  `candidate_ranking_allowed=false`。

新批次缺少这些字段会在推理前失败。旧的已批准只读产物不会被改写；如果旧清单
确实证明单次 FLUX 和单个候选，可作为
`legacy_sealed_manifest_attestation` 继续下游研究，但它明确记录
`profile_qualification_authorized=false`，不能证明整套 profile 没有跨批次挑 seed。

## 5. profile 稳定性验收

在第一次推理前冻结 validation request matrix。每个预先声明的请求都必须进入
分母，要求通过率为 100%；任何失败都会使该 profile revision 不能晋级。测试
矩阵不是生成多个候选后挑赢家，而是验证同一个 pose/prompt/算法合同在声明范围
内是否可靠。

当前浅色比格 v2 是已保存的一次 FLUX、一次 Pixal 旧版证据，可继续完成当前
canary，但它不能反向证明整个 profile 已完成无抽奖资格。下一版 profile
`dog_beagle_open_tricolor_photorealistic_recolor_canary_v3` 已加入单值基础获取
合同；通过后的实例必须迁移到 `stable_animal_template_v1`。
