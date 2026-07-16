# 交接文档：GPURIR 规格房间 + apartment 贴图 + 动物插入（第二轮）

> 本文档自包含。第一轮（`HANDOFF_ANIMALS_APARTMENT.md`）验证了 4 只 Hunyuan3D 动物能进 apartment_0000。本轮建**一个 GPURIR 参数对齐的 shoebox 房间**（默认 5.2×4.4×2.8 m）+ apartment_0000 的 `MI_Floor` / `MI_Walls` 材质 + Y-max 墙上开一个真正的落地窗（4 块 Cube 拼窗洞）+ 从窗户方向斜射的 Directional Light + BP_LightStudio 天空 +  一只 `BP_dog` 放在 GPURIR 声源位置 `(mic + 1.7 m 向 +Y)`，出**一段 360° turntable 视频**。

---

## 一句话现状

Round 1 已验证 4 只 Hunyuan3D 动物能在 apartment_0000 里正确渲染。Round 2 建**参数化 GPURIR shoebox 房间**（`--room-size-m x y z`，默认 `5.2 4.4 2.8`），Cube 拼房间 + apartment_0000 材质 + Y-max 墙落地窗 + 窗户方向的 Directional Light + BP_LightStudio 天空球，把 `BP_dog` 放在 GPURIR 声源位置（`(mic + 1.7 m 向 +Y)`），出**一段 360° 视频**。

---

## 环境（⚠️ 沿用 Round 1）

### Python 环境 —— 必须用 `spear-env`
```bash
/data/jzy/miniconda3/envs/spear-env/bin/python   # Python 3.11，有编译好的 spear_ext
```
**跑任何脚本前先自检**：
```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -c "import spear; print(spear.__can_import_spear_ext__)"
# 必须打印 True
```

### 其它
- UE 5.5: `/data/UE_5.5`
- SpearSim 游戏可执行: `/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh`
- Xvfb: `DISPLAY=:99`（常驻）
- Vulkan ICD: `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`
- **标准前缀**：
  ```bash
  cd /data/jzy/code/SPEAR
  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    /data/jzy/miniconda3/envs/spear-env/bin/python examples/render_in_gpurir_room.py ...
  ```

---

## 本轮 14 项决策（grill 结论）

| # | 决策 |
|---|---|
| Q1 | 房间**参数化** `--room-size-m x y z`，默认 5.2 4.4 2.8（对齐 v77 GPURIR 采样范围 `[4,8]×[4,8]×[2.4,3.5]` 的中位数） |
| Q2 | 材质用 apartment_0000 的 `MI_Floor` + `MI_Walls`；房间要有窗户 + 天空/外景（方案 D） |
| Q3 | 窗户开在 **Y-max** 那面墙上（`y = room_y_m` 的墙） |
| Q4 | 落地窗 **2.0 m × 2.4 m**，x 居中，底边离地 **0.2 m**（所有参数 CLI 化） |
| Q5 | 窗洞用 **4 块 Cube 拼**（左立柱 + 右立柱 + 上过梁 + 下窗台），全部参数从 room/window 尺寸自适应 |
| Q6 | **Directional Light**（斜射从窗户方向）+ **BP_LightStudio**（`/Engine/EngineSky/BP_LightStudio` 自带 skysphere+skylight+fog） |
| Q7 | 只跑 **dog 单只** 一段视频（cat/goose/yak 上一轮已验过） |
| Q8 | dog 放在 **GPURIR 声源位置**：`(mic_x, mic_y + 1.7 m, 0)` = `(2.6, 3.9, 0)` m，dog 在窗户前 0.5 m |
| Q9 | 地面 `z = 0`（GPURIR 契约），floor mesh 顶面对齐；ground-trace 保留作双保险 checklist |
| Q10 | 视频：**radius 200 cm**，1280×720 / 12 fps / 36 帧 / 3 s |
| Q11 | Checklist 全项：Round 1 solo 字段 + 追加 `room_size_m` / `mic_pos_cm` / `source_pos_cm` / `window_bounds_cm` / `directional_light_intensity_lux` / `wall_material` / `floor_material` + `human_review[]` 4 条 |
| Q12 | **新建独立脚本** `examples/render_in_gpurir_room.py`（不改 `render_in_apartment.py`），公共 helper 用 import 复用 |
| Q13 | Game default map 用 **`/Engine/Maps/Entry`**（UE 自带空 map，已验证在 pak 里） |
| Q14 | 纯 helper 全 **TDD**：shoebox layout / window wall / mic+source / room checklist / layout PNG，共 6 个 helper |

---

## 文件地图

| 文件 | 角色 |
|---|---|
| `HANDOFF_GPURIR_ROOM.md` | 本文件，Round 2 spec/handoff |
| `docs/superpowers/plans/2026-07-03-gpurir-room-with-apartment-textures.md` | 9 个 task 的 step-by-step 计划 |
| `examples/render_in_gpurir_room.py` | 新脚本：Cube 拼房间 + 窗户 + 灯光 + dog + 相机 360° |
| `tests/test_render_in_gpurir_room.py` | 新测试文件（不动 Round 1 的 28 个测试） |
| `tmp/asset_meta/dog.json` | read-only |
| `examples/render_in_apartment.py` | read-only + import helpers |
| `HANDOFF_ANIMALS_APARTMENT.md` | read-only 参考 |

---

## 执行顺序

TDD 完成（Tasks 2-7）后一个断点：

1. **Task 1**：写本文件
2. **Tasks 2-6**：TDD 5 组纯 helper（shoebox layout / window wall / mic+source / checklist / layout PNG），每个都是"先失败测试 → 实现 → 通过测试 → 跑整套 44 测试确认不 regress"
3. **Task 7**：集成 `render_gpurir_room` + `configure_gpurir_instance`（覆盖 apartment map，用 `/Engine/Maps/Entry`）+ `spawn_room_piece` / `spawn_directional_light` / `spawn_sky` + CLI
4. **Task 8（断点）**：跑一次 dog 视频，把 mp4 + frame_0000 + layout + checklist 给业主，停下等 approve
5. **Task 9**：wrap-up 汇报 + 请示下一轮方向

---

## Checklist 定义

### Solo 字段（继承自 Round 1 `build_solo_checklist`，自动）
`name / frames / target_cm / scale / radius_cm / ground_z_cm / bounds_bottom_z_cm / lift_applied_cm / penetration_after_lift_cm / clearance_cm / tolerance_cm / ground_ok`

### Room 字段（本轮新增，自动）
| 字段 | 判定 | 来源 |
|---|---|---|
| `room_size_m` | == `[5.2, 4.4, 2.8]`（或用户传入） | `args.room_size_m` |
| `mic_pos_cm` | == `(260, 220, 120)` | `compute_mic_position_cm` |
| `source_pos_cm` | == `(260, 390, 120)` | `compute_source_position_cm` |
| `window_bounds_cm` | `left_x=160, right_x=360, bottom_z=20, top_z=260, y=440` | 派生自 room+window CLI |
| `directional_light_intensity_lux` | == 10.0 | `args.directional_light_intensity_lux` |
| `wall_material` | == `/Game/…/MI_Walls.MI_Walls` | 常量 |
| `floor_material` | == `/Game/…/MI_Floor.MI_Floor` | 常量 |

### `human_review[]` 4 条（业主肉眼判定）
1. 4 面墙都有可见的 apartment 墙贴图（不是灰模）
2. 天花板有可见贴图（默认 fallback 用 MI_Walls）
3. 窗户是**真的洞**——能看到外面天空/光透进来（不是贴图）
4. Directional Light 从窗户方向斜射，能在室内看到明显阴影

---

## 下一轮计划（NOT part of this round）

Round 2 approve 后，候选下一步：
- **Mass render**：`--room-size-m` 参数化 CLI 已就绪，写一个 batch 脚本按 v77 GPURIR 的 `room_range=[4,8], room_z_range=[2.4,3.5]` uniform 采样 N 组尺寸各出一段
- **Puppeteer/骨骼动画**：让 dog 走路（本轮是静止 turntable）
- **多物种同场**：把 4 只动物按 GPURIR 声源坐标放进同一个房间（对齐 v77 multi-source 场景）
