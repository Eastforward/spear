# 交接文档：SPEAR 动物插入 apartment_0000 验证（第二轮）

> 本文档自包含。上一棒交出的是 `HANDOFF_VISUAL_RENDER.md`（Clock 单只 turntable），本轮把 4 只已导入的 Hunyuan3D 动物 (cat/dog/goose/yak) 也丢进 apartment_0000 里验证：4 段单只 + 1 段一字排合影，共 5 段视频。

---

## 一句话现状

上一轮已跑通 apartment_0000 里 Clock 的 turntable（含 ground-trace 防插地、apartment 光照可见、9 个测试全过）。本轮验证 4 只动物 (cat / dog / goose / yak) 在 apartment_0000 的插入效果：**4 段单只 + 1 段一字排合影，共 5 段视频**；每段渲完停下等业主 approve 后再进下一段。GPURIR 规格新房间放到下一轮。

---

## 环境（⚠️ 沿用上一轮的坑）

### Python 环境 —— 必须用 `spear-env`，不是 `thu`
```bash
/data/jzy/miniconda3/envs/spear-env/bin/python   # Python 3.11，有编译好的 spear_ext
```
**为什么**：SPEAR 的 RPC 客户端是 C++ 扩展 `spear_ext`（nanobind 编译），**只在 `spear-env` 里有**。`thu` 环境（Python 3.12）只有纯 Python `spear` 包 → 表面像"RPC 连不上"，其实是客户端根本没创建。

**跑任何脚本前先自检**：
```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -c "import spear; print(spear.__can_import_spear_ext__)"
# 必须打印 True
```

### 其它环境
- **UE 5.5**：`/data/UE_5.5`
- **SpearSim 游戏可执行**：`/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh`
- **Xvfb**（无头渲染）：`DISPLAY=:99`（有一个 `Xvfb :99 -screen 0 1280x720x24` 常驻）
- **Vulkan ICD**：`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`（NVIDIA RTX 4090 D）
- **标准前缀**：
  ```bash
  cd /data/jzy/code/SPEAR
  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    /data/jzy/miniconda3/envs/spear-env/bin/python examples/render_in_apartment.py ...
  ```

---

## 本轮 15 项决策（grill 结论）

| # | 决策 |
|---|---|
| Q1 | **范围** = 动物插入 apartment_0000 验证（GPURIR 房间放下一轮） |
| Q2 | **动物** = cat + dog + goose + yak（4 只，已在 `/Game/MyAssets/Audioset/Blueprints/<name>/BP_<name>` 导入） |
| Q3 | **视频** = 4 段单只 + 1 段合影，共 5 段 mp4 |
| Q4 | **合影布局** = 一字排开 |
| Q5 | **顺序 + 间距** = 从小到大（cat→dog→goose→yak），相邻中心距 = `target_cm + gap_cm` = 80 + 30 = 110 cm |
| Q6 | **相机** = 绕排列中点 360°，半径 = `min(base_r_factor·target + 半跨度, max_radius_cm)` |
| Q7 | **视频规格** = 1280×720 / 12fps / 36 帧 / 3s（沿用上轮） |
| Q8 | **执行** = **每段独立 SpearSim 会话**（上一轮 grill 里我提议单会话连跑，plan 阶段改成分会话——理由：Q9 要求每段停下等业主审核，session 挂着等审核会锁住 GPU/xvfb，分会话反而更稳。若业主要求连跑，改回单 session 只是把 5 次 `render_...` 合并到一个 `main` 里） |
| Q9 | **异常处理** = 每段渲完停下问业主（5 次断点） |
| Q10 | **spawn 位置** = 沿用 Clock 那个已 trace 干净的位置：`spawn_x=-120.0, spawn_y=80.0`，实测 apartment floor z=27.11 cm |
| Q11 | **过关线** = 5 项 checklist 全过（贴图/贴地/穿墙/apartment 光/构图完整），合影 +2 项（4 只都在画面 + 大小合理）；日志先行、视频二次确认 |
| Q12 | **输出** = `/data/jzy/code/SPEAR/tmp/render_animals_apartment/{cat,dog,goose,yak,group}/`，含 `turntable.mp4` + `frame_0000.png` + `checklist.json`；`group` 段追加 `layout.png` |
| Q13 | **测试** = TDD，先扩测试再改代码；纯 helper 单元测试全在 `tests/test_render_in_apartment.py` |
| Q14 | **文档** = 就是本文件，加上 `docs/superpowers/plans/2026-07-03-animals-in-apartment.md` |
| Q15 | **执行方式** = 主 session 直接干，无 worktree（SPEAR 不是 git repo）、无 subagent |

---

## 文件地图

| 文件 | 角色 |
|---|---|
| `HANDOFF_ANIMALS_APARTMENT.md` | 本文件，本轮 spec/handoff |
| `docs/superpowers/plans/2026-07-03-animals-in-apartment.md` | 13 个 task 的 step-by-step 计划 |
| `examples/render_in_apartment.py` | 主脚本：加 `animal_bp_path` / `--animal` / `--mode group` / checklist / layout |
| `tests/test_render_in_apartment.py` | 单元测试：本轮从 9 个扩到 28 个 |
| `tmp/asset_meta/{cat,dog,goose,yak}.json` | 每只动物的 bbox 元数据（read-only） |
| `HANDOFF_VISUAL_RENDER.md` | 上一轮 handoff（read-only 参考） |

---

## 执行顺序

TDD 完成（Tasks 1-7）后，5 个断点顺跑：

1. **断点 1 (Task 8)**：cat 单只 turntable → 出 mp4 + png + checklist → 停下等业主 `next`
2. **断点 2 (Task 9)**：dog → 同上
3. **断点 3 (Task 10)**：goose → 同上
4. **断点 4 (Task 11)**：yak → 同上
5. **断点 5 (Task 12)**：group 一字排合影 → mp4 + png + layout.png + checklist → 停下等业主 `done`

全部 approve 后 **Task 13** 汇报总结并请示是否进 GPURIR 房间那一轮。

---

## checklist 定义

### 单只 checklist (checklist.json) —— `build_solo_checklist` 自动产出
| 字段 | 类型 | 判定 | 来源 |
|---|---|---|---|
| `name` | str | — | `args.name` |
| `frames` | int | == 36 | `args.frames` |
| `target_cm` | float | == 80.0 | `args.target_cm` |
| `scale` | float | 记录 | `compute_asset_fit` |
| `radius_cm` | float | ≤ `max_radius_cm` | `min(r_factor·target, max_radius_cm)` |
| `ground_z_cm` | float | 记录 | `sample_ground_z` LineTrace |
| `bounds_bottom_z_cm` | float | 记录 | `GetActorBounds` |
| `lift_applied_cm` | float | 记录 | `compute_bounds_lift` |
| `penetration_after_lift_cm` | float | \|·\| ≤ `tolerance_cm` | 派生 |
| `clearance_cm` | float | == 0.5 | `args.ground_clearance_cm` |
| `tolerance_cm` | float | == 0.5 | `args.ground_tolerance_cm` |
| `ground_ok` | bool | == True | 派生自 penetration ≤ tolerance |

**人工判定的（视频/PNG 里看）：**
- 贴图正常（不是灰模、无白斑）
- 全程不穿墙不出画
- apartment 窗户光/环境光可见
- 360° 每帧完整动物（不被家具切一半）

### 合影 checklist —— `render_group` 产出
追加：`animals` 列表、`gap_cm`、`per_animal[]`（每只的 x/y/scale/ground_z/bounds_bottom/lift）、`removed_furniture_count`。
**人工判定的追加两项：**
- 4 只都在画面里
- 相对大小合理（yak 最大、cat 最小）

**外加 `layout.png`**：matplotlib 生成的俯视图，标 4 只 spawn 位（橙色圆盘）+ 相机绕圈轨迹（蓝色虚线）+ orbit center。

---

## 下一轮计划（NOT part of this round）

本轮 5 段全部 approve 后：新建 GPURIR shoebox 房间 5.2 m × 4.4 m × 2.8 m，**贴 apartment_0000 材质但不是继续 Cube 缩放**——生成带正确 UV tiling 的墙/地/天花板 mesh，加窗户墙 + 外景 + 室内 area light + 少量 apartment prop，让它有真实尺度和光照，最后把动物放进去。产出新的 handoff + plan + 一段预览渲染。
