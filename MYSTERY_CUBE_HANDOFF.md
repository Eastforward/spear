# 神秘方块 Bug — 交接文档（给新 agent）

> **目标**：找出并修复 SPEAR GPURIR 房间渲染里持续出现的"神秘马赛克方块"。这是数据集引擎的阻断性 bug，**必须根治**，不能绕过。
> 上一棒（Claude）已经花了 ~3 小时、做了 14 轮渲染 + 多种诊断，**未能根治**。本文档把所有硬证据和已排除项写清楚，让新 agent 不重复走错路。
>
> **2026-07-04 更新**：本轮已经解决三层问题：可见 checkerboard 方块、隐藏方块残影阴影、以及后续用户圈出的左后地板黑影。下面保留原始诊断记录，并在第 12 节追加最终根因、修复和推荐排查方法。

---

## 0. 一句话现状

在 SPEAR 的 `/Engine/Maps/Entry` 空地图上，用 Cube mesh 自建一个 5.2×4.4×2.8m 的 GPURIR shoebox 房间并渲 360° 视频时，画面里**始终出现一个灰白 checkerboard 马赛克方块**（WorldGridMaterial fallback 贴图），固定在世界坐标**原点 (0,0,0) 附近**，体积约 1-2m³，有"影子"（depth-write），**换相机角度/换房间布局都不消失**。同一个相机 pipeline 在 `apartment_0000` 已建好的地图里**没有这个方块**。

---

## 1. 环境（必读，坑很多）

### 1.1 Python 环境 —— 必须 `spear-env`，不是 `thu`
```bash
/data/jzy/miniconda3/envs/spear-env/bin/python   # Python 3.11，有编译好的 spear_ext
```
**为什么**：SPEAR 的 RPC 客户端是 C++ 扩展 `spear_ext`，只在 `spear-env` 里有。`thu` 环境只有纯 Python `spear` 包，会静默吞 `ModuleNotFoundError`，表现为"引擎连不上 RPC"。

**自检**：
```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -c "import spear; print(spear.__can_import_spear_ext__)"
# 必须打印 True
```

### 1.2 其它环境
- UE 5.5：`/data/UE_5.5`
- SpearSim 游戏可执行：`/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh`
- Xvfb：`DISPLAY=:99`（`Xvfb :99 -screen 0 1280x720x24` 常驻）
- Vulkan：`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`（RTX 4090）
- 标准前缀：
  ```bash
  cd /data/jzy/code/SPEAR
  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    /data/jzy/miniconda3/envs/spear-env/bin/python <script>.py
  ```

---

## 2. 复现

最小复现脚本（必跑，30 秒出结果）：
```bash
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python /data/jzy/code/SPEAR/examples/render_in_gpurir_room.py \
  --animal dog --run-name repro
```
输出：`/data/jzy/code/SPEAR/tmp/render_gpurir_room/repro/turntable.mp4` + 36 帧 PNG。
**方块在每帧都可见**，在 frame_0009 / frame_0035 最明显。

关键代码：
- 主脚本：`/data/jzy/code/SPEAR/examples/render_in_gpurir_room.py`
- 公共 helper（spawn_camera 等）：`/data/jzy/code/SPEAR/examples/render_in_apartment.py`
- 它加载 `/Engine/Maps/Entry`（常量 `EMPTY_MAP`，第 41 行），spawn 6 面 Cube 墙 + 天花板 + 地板（`compute_shoebox_room_layout`），加 `spawn_sky`（SkyAtmosphere+SkyLight+ExponentialHeightFog）+ `spawn_directional_light` + `spawn_point_light` + 可选 `SphereReflectionCapture` + 相机 + 动物（BP_dog 等）。

---

## 3. 方块的硬特征（已用实验确认）

| 特征 | 确认方式 | 含义 |
|---|---|---|
| 有 geometry | 用户说"有影子" | 不是纯 HUD overlay |
| **贴图 = WorldGridMaterial 灰白 checkerboard** | 用户多次描述"马赛克深浅不一小方块" | UE 默认 fallback material |
| **固定在世界原点 (0,0,0) 附近** | 俯视图（`/tmp/diag_topdown3/topdown.png`）显示方块在房间**西南角**（左下角），不随房间布局移动 | 不跟房间/相机走 |
| **不是被光源照亮的** | 用户说"完全无光下方块不在，全黑" | **纠正**：方块**被光照亮**（不是自发光） |
| 无 collision | 150 条 LineTraceSingleByProfile（BlockAll + Visibility）从相机向方块区域射，**全部命中已知墙/地板，没有命中未知物体**（`/data/jzy/code/SPEAR/tools/raycast_cube.py`） | raycast 命中不到 |
| 不在 actor 列表 | `find_actors_by_class('AActor')` 枚举 30 个 actor，**没有一个 bounds 跟方块匹配** | 不是 AActor 子类 |
| 不在 component 列表 | `get_components_by_class_as_dict(actor, 'UPrimitiveComponent')` 枚举所有 actor 的组件，**只有 6 面墙的 StaticMeshComponent + DefaultPhysicsVolume** | 不是 actor 拥有的 UPrimitiveComponent |

---

## 4. 已尝试且**无效**的修复（不要重复）

### 4.1 destroy 各类 actor —— 全部无效
逐类 destroy，方块都在：
- `AStaticMeshActor`、`ASkeletalMeshActor`、`ABrush`、`ADecalActor`
- `APlayerStart`、`ADefaultPawn`、`ASpectatorPawn`
- `AInstancedFoliageActor`
- `AGameplayDebuggerCategoryReplicator`、`AGameplayDebuggerPlayerManager`
- `AHUD`（destroy 了 2 个 AHUD）

注意：`find_actors_by_class('AChaosDebugDrawActor')` 会 KeyError（类名不对）。

### 4.2 ShowFlagSettings（UPROPERTY）—— 无效
在 `spawn_camera`（`render_in_apartment.py:253-271`）里：
```python
comp.set_property_value(property_name="ShowFlagSettings", property_value=[
    {"ShowFlagName": "BSP", "Enabled": False},
    {"ShowFlagName": "BSPTriangles", "Enabled": False},
])
```
**设置确实生效了**（读回 `comp.get_property_value('ShowFlagSettings')` 确认 BSP=False），但方块还在。

### 4.3 console command —— 无效
```python
game.unreal_service.execute_console_command(command='r.ShowFlags.BSP 0')
game.unreal_service.execute_console_command(command='r.ShowFlags.StaticMeshes 0')
game.unreal_service.execute_console_command(command='r.ShowFlags.Brush 0')
```
俯视图测试（`/tmp/diag_topdown4/`）：3 张图方块都在。
- `00_baseline.png` / `01_no_static_meshes.png` / `02_no_bsp.png` 方块都在
- **重要**：`01_no_static_meshes`（关 StaticMeshes）时方块还在，说明**方块不是 StaticMesh**——排除了 BSP！因为 UE5 里 BSP 也渲染为 static mesh。

### 4.4 换 material —— 无效
- 把所有墙的 material 从 `M_Basic_Wall` 换成 `MI_Floor`（apartment 木地板）/ `M_Basic_Floor`：方块都在
- 单独换 wall_x1：方块还在

### 4.5 bisect 逐项删除 geometry —— 方块跟 geometry 无关
脚本：`/data/jzy/code/SPEAR/tools/bisect_cube.py`
保留光源，逐项删 geometry：
- A_all（全配置）→ B_no_ceiling → C_no_walls_y → D_no_walls_x → E_floor_only → **F_no_floor（啥也不 spawn，只有 Entry map + 光源 + 相机）**
**结果：F 配置（完全不 spawn 任何 Cube）方块还在**（`/tmp/diag_bisect/F_no_floor.png`）。
**这证明方块不是我 spawn 的任何东西，是 Entry map 或 SPEAR 自动 spawn 的**。

### 4.6 换 map —— 卡死
把 `EMPTY_MAP` 从 `/Engine/Maps/Entry` 换成 `/Engine/Maps/Templates/Minimal_Default`：SpearSim 进程启动后**死锁**（21 分钟 CPU 3%，stdout 0 bytes，无输出目录）。已 revert 回 Entry。
（`/Engine/Maps/Templates/Minimal_Default_BuiltData.uasset` 确认在 pak 里。）

---

## 5. 关键诊断产物（看图说话）

| 路径 | 内容 | 关键观察 |
|---|---|---|
| `/tmp/diag_topdown3/topdown.png` | 无天花板俯视图，相机 (260,220,500) 朝下 | **方块在房间西南角（左下）**，体积比斜视角看到的大 → 固定在世界原点附近 |
| `/tmp/diag_topdown4/02_no_bsp.png` | 关 BSP show flag 的俯视图 | 方块还在 → 不是 BSP |
| `/tmp/diag_bisect/F_no_floor.png` | 完全不 spawn 任何 geometry，只有 Entry+光源+相机 | 方块还在 → 是 Entry/SPEAR 自带，不是我的 Cube |
| `/tmp/diag_raycast/frame_baseline.png` | 斜视角 baseline（方块最明显） | 方块在画面右侧 |
| `/tmp/diag_one_marker/contact.png` | 3 张图各 spawn 1 个棕色 marker（原点/中心/东墙） | 用户判断"方块离 mic 位置(260,220) 最近"，但俯视图推翻了这个（方块在原点） |
| `/data/jzy/code/SPEAR/tmp/render_gpurir_room/dog_cat_v16_bsp_off/` | 完整 36 帧视频，方块在 frame_0000/0009/0035 最明显 | 当前"最新"版本，ShowFlagSettings 已设但无效 |

---

## 6. 已确认的事实（不要再查）

1. **Entry.umap 里有 BSP 数据**：`/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Saved/Cooked/Linux/Engine/Content/Maps/Entry.umap` strings 含 `Model2`、`ModelComponent`、`Default__Model`、`Default__ModelComponent`、`BodySetup`、`CollisionCapsule`、`LevelBoundsLocation=(83.33, 39.51, 104.00)`、`LevelBoundsExtent=(211.33, 167.51, 232.00)`。但**关 BSP show flag 无效**，所以即使方块在原点，也不是 BSP（或 BSP show flag 关不掉它）。

2. **SPEAR 自动 spawn 的 actor 清单**（Entry map 加载后，`find_actors_by_class('AActor')` 枚举，30 个）：
   - `ChaosDebugDrawActor`、`GameplayDebuggerCategoryReplicator`（bounds ext=1000000 全世界）、`GameplayDebuggerPlayerManager`、`PlayerCameraManager` ×2、`HUD`、`ParticleEventManager`、`__SP_STABLE_NAME_MANAGER__`、`ExponentialHeightFog`、`GameNetworkManager`、`SkyLight`、`WorldInfo`、`SkyAtmosphere`、`GameSession`、`GameStateBase`、`PlayerState` ×2、`AbstractNavData-Default`、`DirectionalLight`、`PointLight`、`PlayerStart0`（loc=(55,79,208), bounds ext=(40,40,40)，有 `CollisionCapsule`）、`SpGameMode`、`InstancedFoliageActor`、`SpPlayerController`、`SpDebugCameraHUD`、`DefaultPhysicsVolume`、`SpDebugCameraController`、`__SP_DEFAULT_PAWN__ SpSpectatorPawn`（loc=(55,79,208), bounds ext=(35,35,35)，有 `CollisionComponent0`）
   - **没有一个的 bounds 跟方块匹配**（除了 PlayerStart/SpectatorPawn 在 (55,79)，但俯视图显示方块在 (0,0)）

3. **apartment_0000 没有方块**：用同一个 `spawn_camera`、同样的 BP_CameraSensor，方块不出现。apartment 是已建好的 map，不 spawn Cube。

4. **方块大小变化**：斜视角下方块看起来约 1m³；俯视看"相当大"（2m+）。**说明方块可能是不规则形状**，从不同角度看大小不同。

---

## 7. 还没查的方向（建议优先级）

### 7.1 ⭐ 方块可能是 BSP，但 show flag 关不掉（最高优先级）
- UE5 里 BSP 通过 `UModelComponent` 渲染，**挂在 `ULevel::ModelComponents`**，不属于任何 actor —— 这解释了为什么 actor/component 枚举都找不到它
- `r.ShowFlags.BSP 0` 在 game runtime 可能被 `FORCEINLINE` 或 build 配置忽略
- **真正能关 BSP 的方式**：
  - a. `UKismetSystemLibrary::ExecuteConsoleCommand` 用 `show BSP`（不带 r.ShowFlags 前缀）
  - b. 直接遍历 `ULevel::ModelComponents`，对每个 `UModelComponent` 调 `SetVisibility(false)` —— 需要一个 SPEAR C++ RPC 暴露这个，或用 `find_objects_by_class('UModelComponent')`（如果 SPEAR 支持）
  - c. 在 scene capture 上设 `ShowFlags.SetDisabled(FEngineShowFlags::EShowFlag::BSP)`（C++）
- **测试方法**：spawn 后遍历 `game.unreal_service.find_objects_by_class(...)` 找 `UModelComponent`，看是否存在并隐藏

### 7.2 ⭐ 用 `obj list` console command 列出原点附近所有 UObject
- `obj list class=ModelComponent` 或 `obj list class=UModelComponent` 看有没有实例
- `obj list` 全量太慢，加 `class=` 过滤

### 7.3 方块可能是 SPEAR 的某个 debug draw（`ENABLE_DRAW_DEBUG`）
- `SpDebugCameraHUD.cpp` 有 `#if ENABLE_DRAW_DEBUG` 块，但只画文字
- 搜 `DrawDebugSolidBox` / `DrawDebugMesh` 在整个 SPEAR C++ 源码 —— **已搜过，没有**（`grep -rn "DrawDebugSolidBox\|DrawDebugMesh\|DrawDebugSphere" /data/jzy/code/SPEAR/cpp/` 无结果）
- 但 `DrawDebugBox`（线框）也没搜到——所以不是 DrawDebug

### 7.4 方块可能是 NavMesh 可视化
- actor 列表里有 `AbstractNavData-Default`
- NavMesh 在 game 里默认不可见，但某些 cvar 开了会显示绿色/彩色多面体
- 试 `show Navigation 0` 或 `r.ShowFlags.NavigationMesh 0`

### 7.5 方块可能是 `ChaosDebugDrawActor`（物理 debug）
- actor 列表第一个就是它
- Chaos 物理引擎的 debug draw，在某些 build 里默认开
- 试 `p.Chaos.DebugDraw 0` 或类似 cvar

### 7.6 终极方案：不用 Entry map
- 在 SpearSim project 里用 UE Editor 新建一个**真正空的 map**（不基于 Entry，没有任何 BSP/actor）
- 保存为 `/Game/Maps/gpurir_empty.umap`
- 重 cook pak（`run_uat.py`，~10 分钟，参考 `/data/jzy/code/SPEAR/tools/run_uat.py`，命令在 `HANDOFF_ANIMALS_APARTMENT.md` 第 2 节）
- 改 `EMPTY_MAP = "/Game/Maps/gpurir_empty"`
- 这能根治（apartment 就是用预建 map，没方块）

### 7.7 用 `ShowOnlyActors` 反向证明
- 设 `comp.PrimitiveRenderMode = PRM_UseShowOnlyList` + `ShowOnlyActors = [我们 spawn 的 6 面墙]`
- 如果方块消失 → 它不是 actor（是 level/BSP/viewer 级）
- 如果还在 → 它 somehow 在 show-only list 里（不太可能）
- 上一棒试过但脚本崩了（`ShowOnlyActors` 需要 actor object list，API 用法待查 `unreal_service.py`）

---

## 8. SPEAR RPC API 速查（用错的都会翻车）

```python
# 正确的 UFUNCTION 调用（kwarg）
comp.SetIntensity(NewIntensity=2200.0)
comp.Initialize()
cam.K2_SetActorLocationAndRotation(NewLocation={...}, NewRotation={...}, bSweep=False, bTeleport=True)

# 正确的 UPROPERTY 读写（用 set/get_property_value）
comp.set_property_value(property_name='ShowFlagSettings', property_value=[...])
val = comp.get_property_value(property_name='ShowFlagSettings')

# 错误：直接赋值
comp.ShowFlagSettings = [...]  # 报 'UnrealObject' object is not callable 或不生效

# 列 actor
actors = game.unreal_service.find_actors_by_class(uclass='AStaticMeshActor')
# 列 actor 的组件（返回 dict）
prims = game.unreal_service.get_components_by_class_as_dict(
    actor=a, uclass='UPrimitiveComponent',
    include_actor_stable_name=True, include_actor_unreal_name=True)

# console command
game.unreal_service.execute_console_command(command='r.ShowFlags.BSP 0')

# spawn（注意 spawn_parameters 要 AlwaysSpawn，否则会被 collision 挡）
game.unreal_service.spawn_actor(
    uclass='AStaticMeshActor',
    location={'X':..,'Y':..,'Z':..},
    spawn_parameters={'SpawnCollisionHandlingOverride': 'AlwaysSpawn'})
```

**坑**：
- `find_actors_by_class('AChaosDebugDrawActor')` → KeyError（类名不对，跳过这类时要 try/except）
- `K2_GetActorLocation(as_dict=True)` 返回的 dict key 是**大写** `X/Y/Z`（不是小写）—— 之前脚本因此崩
- GetStaticMesh() 不能直接 `smc.GetStaticMesh()`，要查正确的 proxy API（上一棒没搞定，用 `get_components_by_class_as_dict` 拿到 component handle 后再调）

---

## 9. 用户的关键反馈（不可忽略）

1. 方块是"马赛克贴图，深浅不一的小方块" = WorldGridMaterial
2. 方块**有影子**（depth-write）= 有 geometry
3. 方块**完全无光时消失**（全黑）= 被光照亮，不是自发光
4. **俯视图方块在房间西南角（左下），体积比斜视角大** = 固定在世界原点 (0,0,0) 附近，不随房间/相机移动
5. 方块"似乎没上色" = 用 fallback material
6. 用户多次纠正 Claude 的肉眼判断 —— **新 agent 应该让用户看图，不要自己下结论**

---

## 10. 用户的工作风格（重要）

- 用户**懂技术**（自己跑命令、读图、提假设），不接受糊弄
- 用户**下班了**可能不在，新 agent 应该**自主推进**，把诊断产物路径列清楚让用户回来看
- 用户要**根治**，不要 workaround（"相机避开方块位置"这种方案被否决）
- 之前 Claude 的失败：**3 次肉眼误判**（说"方块消失"但实际还在）—— 新 agent 必须用**代码证据**，不要靠"我看图觉得消失了"

---

## 11. 推荐的第一步

1. **先跑最小复现**（第 2 节命令），确认方块在你的环境里也复现
2. **跑 7.1 的诊断**：spawn 房间后，用 `find_objects_by_class` 或 `obj list` console command 找 `UModelComponent`，如果找到 → 隐藏它 → 渲一帧看方块消失没
3. 如果 7.1 无效，按 7.2~7.5 逐个试 console command
4. 如果全无效，走 7.6（新建空 map + 重 cook）—— **保证根治**

---

## 12. 2026-07-04 最终解决记录和可复用排查方法

### 12.1 问题 A：可见 checkerboard 方块

**根因**：`/Engine/Maps/Entry` 里残留的 BSP/Brush geometry 被挂在 `ULevel::ModelComponents` 上。它不是 actor，也不是任何 actor 拥有的 `UPrimitiveComponent`，所以 `find_actors_by_class` / destroy actor / 枚举 actor components 都找不到它。

**关键证据**：
- 方块固定在世界原点附近，不随房间或相机移动。
- 完全不 spawn 自己的房间 geometry，方块仍然存在。
- 直接写 `ShowFlagSettings` UPROPERTY 无效；必须调用 scene capture component 的 UFUNCTION setter。

**修复**：在 `examples/render_in_apartment.py:spawn_camera()` 中调用：

```python
comp.SetShowFlagSettings(
    InShowFlagSettings=[
        {"ShowFlagName": "BSP", "Enabled": False},
        {"ShowFlagName": "BSPTriangles", "Enabled": False},
    ],
)
```

注意：用 `SetShowFlagSettings(...)`，不要直接 `set_property_value("ShowFlagSettings", ...)`。直接写属性不会触发 UE 内部 `UpdateShowFlags()`。

**验证**：

```bash
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python tools/verify_entry_bsp_hidden.py
```

通过时应看到 `VERIFY_ENTRY_BSP max_gray_component_area=1`。

### 12.2 问题 B：方块不可见后仍留下残影阴影

**根因**：scene capture show flag 只隐藏 BSP 主渲染 pass；`ULevel::ModelComponents` 仍然参与 shadow pass。

**关键证据**：
- 关点光源阴影不能消掉残影。
- 对 `ULevel::ModelComponents` 调 `SetCastShadow(False)` 后，残影 ROI 分数从约 `34.1` 降到约 `22.3`。
- `ULevel.ModelComponents` 数量为 `4`，与 Entry map 残留 BSP 相符。

**修复**：在 `examples/render_in_apartment.py` 加 `disable_level_model_component_shadows(game)`，遍历 `world.PersistentLevel.ModelComponents`，对每个 `UModelComponent`：

```python
model_component.SetCastShadow(NewCastShadow=False)
model_component.SetCastHiddenShadow(NewCastHiddenShadow=False)
model_component.bCastDynamicShadow = False
model_component.bCastStaticShadow = False
```

并在 `spawn_camera()` 初始化 scene capture 后、设置 BSP show flags 前调用。

**验证**：

```bash
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python tools/verify_entry_bsp_shadow_hidden.py
```

通过时应看到 `VERIFY_ENTRY_BSP_SHADOW score` 小于 `26.0`。当前通过值约 `22.342`。

### 12.3 问题 C：用户红框圈出的左后地板黑影和错误修复产生的光柱

**2026-07-04 纠正**：上一版“只关闭 `wall_y1_top` 投影”的结论已撤回。它确实会提亮红框区域，但同时让阳光穿过本应挡光的窗口上沿墙，产生一根不物理的光柱。用户指出“正常光影被搞坏了”后，已把默认值恢复为 `wall_y1_top` 正常投影。

**当前证据结论**：
- 黑色不是新的 mesh，也不是 Entry BSP 残留；它是 GPURIR 木地板材质在 DirectionalLight 阴影区被压暗后的表现。
- 该阴影区的遮挡几何主要是 `wall_y1_top`（窗口上沿到天花板之间那块后墙/lintel）。
- 如果让 `wall_y1_top` 不 cast shadow，反向光线从地板看过去会从 `first_hit=wall_y1_top` 变成 `first_hit=open_sky/window`，所以地板上会冒出新光柱。
- 因此不能再用“关墙体投影”作为修复；如果要弱化黑块，应从材质、曝光/环境光、太阳角度或房间几何参数处理。

**关键证据**：
- 新诊断脚本：`tools/diag_gpurir_light_path_sources.py`
- 证据拼图：`/tmp/diag_gpurir_light_path_sources/contact.png`
- 侧剖光线路径图：`/tmp/diag_gpurir_light_path_sources/light_path_yz.png`
- 运行日志：`/tmp/diag_gpurir_light_path_sources/run.log`

模式对比：

| 模式 | `floor_artifact_mean` | `sun_patch_mean` | 结论 |
|---|---:|---:|---|
| `physical_floor_texture` | `69.445` | `198.421` | 原木地板 + 正常阴影时黑块明显 |
| `physical_plain_floor` | `189.393` | `236.314` | 换纯色地板后黑纹理基本消失，说明黑色强度来自地板材质 |
| `directional_no_shadow` | `154.362` | `180.103` | 关 DirectionalLight shadow 后黑块消失，说明形状来自阴影 |
| `top_wall_no_shadow` | `132.344` | `198.667` | 会产生新光柱，已否定 |

反向追光关键输出：

```text
TRACE pixel=(535,185) world=(143.8,150.3,0.5) top_wall_casts_shadow=True first_hit=wall_y1_top
TRACE pixel=(535,185) world=(143.8,150.3,0.5) top_wall_casts_shadow=False first_hit=open_sky/window
TRACE pixel=(685,360) world=(260.0,249.9,0.5) top_wall_casts_shadow=True first_hit=open_sky/window
```

当前代码保留了诊断开关，但默认恢复物理投影：

```python
def piece_casts_shadow(
    name,
    *,
    ceiling_casts_shadow=True,
    window_top_wall_casts_shadow=True,
    window_wall_casts_shadow=True,
):
    if name == "ceiling":
        return bool(ceiling_casts_shadow)
    elif name == "wall_y1_top":
        return bool(window_wall_casts_shadow and window_top_wall_casts_shadow)
    elif name.startswith("wall_y1_"):
        return bool(window_wall_casts_shadow)
    else:
        return True
```

`spawn_room_piece(...)` 仍支持 `cast_shadow` 参数，便于做消融实验：

```python
smc.SetCastShadow(NewCastShadow=bool(cast_shadow))
smc.SetCastHiddenShadow(NewCastHiddenShadow=False)
smc.bCastDynamicShadow = False
smc.bCastStaticShadow = False
```

如需复现错误光柱，可显式加：

```bash
--no-window-top-wall-casts-shadow
```

上一版错误视频不要作为最终结果使用：

```text
/data/jzy/code/SPEAR/tmp/render_gpurir_room/codex_window_top_shadow_fixed/turntable.mp4
```

**2026-07-04 调参实验**：按用户要求，从“地板材质、曝光/环境光、太阳角度、窗口/房间几何”四类方向试过，不关闭任何墙体投影。

关键脚本和图：
- 调参矩阵脚本：`tools/diag_gpurir_tuning_matrix.py`
- 第一轮拼图：`/tmp/diag_gpurir_tuning_matrix/contact_topdown.png`、`/tmp/diag_gpurir_tuning_matrix/contact_perspective.png`
- 第二轮拼图：`/tmp/diag_gpurir_tuning_matrix_round2/contact_topdown.png`、`/tmp/diag_gpurir_tuning_matrix_round2/contact_perspective.png`
- 三联候选对比：`/tmp/gpurir_tuning_candidates_frame0017.png`

核心数值：

| 模式 | `floor_artifact_mean` | `sun_patch_mean` | 评价 |
|---|---:|---:|---|
| baseline | `69.379` | `198.400` | 原始物理投影，暗块最重 |
| `plain_floor_debug` | `189.389` | `236.314` | 证明木地板材质贡献很大；不建议作为最终视觉 |
| `ceiling_light_10000` | `123.169` | `210.639` | 单调环境光有效，但偏平 |
| `sun_pitch_m30` | `84.165` | `167.816` | 单调太阳角度有限 |
| `light6000_pitch_m30` | `115.458` | `192.574` | 推荐折中：暗块变浅、层次还在 |
| `light10000_pitch_m30` | `132.014` | `201.969` | 数值最好，但画面更平 |
| `window_w180*` 系列 | `73.151`~`110.891` | `199.380`~`207.576` | 改窗口几何收益不如光照，且改变光斑形状 |

已生成候选视频：

```text
/data/jzy/code/SPEAR/tmp/render_gpurir_room/codex_tune_light6000_pitch_m30/turntable.mp4
/data/jzy/code/SPEAR/tmp/render_gpurir_room/codex_tune_light10000_pitch_m30/turntable.mp4
```

当前推荐先用 6000 lm + sun pitch -30：

```bash
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python examples/render_in_gpurir_room.py \
  --animal dog \
  --run-name codex_tune_light6000_pitch_m30 \
  --ceiling-light-lumens 6000 \
  --directional-light-pitch-deg -30
```

### 12.4 推荐的排查方法：不要靠肉眼，靠可复现证据

这次最值得复用的方法是：

1. **固定复现条件**：固定 map、相机、光源、frame index、随机输入和输出目录。
2. **先做空间定位**：凡是“影子/方块/奇怪区域”，先做俯视图或正交图。透视相机会误导判断。
3. **把用户圈出的区域变成 ROI**：用像素坐标框住问题区，再选一个参考区；输出 `mean/max/count`，不要只说“看起来消失了”。
4. **一次只改一个变量**：例如 `plain_floor_material`、`directional_no_shadow`、`wall_y1_top_no_shadow`、`ceiling_no_shadow`。谁一关 ROI 就跳变，谁就是来源。
5. **把诊断变成验证脚本**：先让脚本在旧代码上失败（RED），再做最小修复，最后让同一脚本通过（GREEN）。
6. **旧测试失败时先判断测试前提是否变了**：本轮曾经误把整面窗墙投影关掉，导致旧 Entry BSP 阴影 ROI 被新的光照构图污染；解决方式是让旧测试显式传 `cast_shadow=True`，继续隔离它本来要测的 Entry BSP 残影。
7. **每个结论都留路径**：输出原图、ROI 图、对比拼图、最终视频路径。下一个人不需要相信 agent 的描述，直接看图和跑脚本。

### 12.5 当前建议的一键验证顺序

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m py_compile \
  examples/render_in_apartment.py \
  examples/render_in_gpurir_room.py \
  tools/verify_entry_bsp_hidden.py \
  tools/verify_entry_bsp_shadow_hidden.py \
  tools/diag_window_shadow_pieces.py \
  tools/diag_user_floor_shadow_topdown.py \
  tools/diag_gpurir_light_path_sources.py

DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python tools/verify_entry_bsp_hidden.py

DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python tools/verify_entry_bsp_shadow_hidden.py

DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python tools/diag_gpurir_light_path_sources.py

DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python examples/render_in_gpurir_room.py \
  --animal dog --run-name codex_physical_shadows_restored
```

目前这条链路的核心结论：方块主 pass 隐藏、Entry BSP 残影阴影关闭；红框地板黑块已定位为“木地板材质 + `wall_y1_top` directional shadow”的组合，不应通过关闭 `wall_y1_top` 投影修复。
