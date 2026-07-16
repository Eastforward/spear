# 交接文档：SPEAR 真实房间 + 物体渲染（给能看图的 GPT）

> 这份文档自包含，目标读者：一个**能看图、能读写代码**的 GPT，接手后要让"SPEAR 渲染出真实公寓质感 + Hunyuan3D 物体放进去 + 物体能动起来"这条路走通。
> 上一棒（Claude）已经把管线跑通，但**视觉效果不达标**。本文档讲清楚：背景、现状、卡点根因、可选方案、以及现成可粘贴的 prompt。

---

## 0. 一句话现状

管线已经能跑通渲染（之前的阻塞是**用错了 Python 环境**，已修复）。现在卡在**视觉质量**：自己用 Cube 搭的房间贴公寓材质后，墙地平得像灰墙，很难看。业主要求：**就用 SPEAR 自带那个有窗户、有家具的公寓场景（apartment_0000）的质感**，那个本来就好看。

---

## 1. 项目背景（为什么做这件事）

- **下游**：`/data/jzy/code/Spatial/v77_4ch_S2L` —— 一个 4 通道声源定位模型（audio spatial localization）。以后要**加视频模态**：让模型看到场景里的物体（同时听到 gpuRIR 仿真的声音）。
- **当前阶段**：**视觉可行性验证**（不是出训练数据）。要验证三件事：
  1. 房间能渲染得**真实**（公寓质感）
  2. Hunyuan3D 生成的物体能**合理放进去**（尺寸/位置/朝向对）
  3. 物体能**动起来**（骨骼动画，例如人形 Manny 走路）
- **帧数/分辨率暂时不重要**，先把"真实 + 合理"做出来。

### 1.1 物体来源
- Hunyuan3D-2.1 生成的 `.glb` → 导入 UE 成 StaticMesh + Blueprint
- 已有 100+ 个资产 meta：`/data/jzy/code/SPEAR/tmp/asset_meta/*.json`（alarm_clock, banjo, cello, piano, dog, cat, helicopter... 每个 JSON 记录 `ext`/`bmin_z`/`height`）
- 已导入的 BP：`/Game/MyAssets/Blueprints/BP_Clock.BP_Clock_C` 等

---

## 2. 环境（⚠️ 最关键的坑，务必用对）

### 2.1 Python 环境 —— 必须用 `spear-env`，不是 `thu`
```bash
/data/jzy/miniconda3/envs/spear-env/bin/python   # Python 3.11，有编译好的 spear_ext
```
**为什么**：SPEAR 的 RPC 客户端是 C++ 扩展 `spear_ext`（nanobind 编译），**只在 `spear-env` 里有**（`/data/jzy/miniconda3/envs/spear-env/lib/python3.11/site-packages/spear_ext/spear_ext.cpython-311-x86_64-linux-gnu.so`）。
`thu` 环境（Python 3.12）只有纯 Python 的 `spear` 包，没有 `spear_ext`，而 `spear/instance.py` 的连接逻辑用**裸 `except:`** 把 `ModuleNotFoundError` 静默吞掉 → 表面像"引擎连不上 RPC"，其实是客户端根本没创建出来。引擎每次都正常启动。

**跑任何脚本前先自检**：
```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -c "import spear; print(spear.__can_import_spear_ext__)"
# 必须打印 True
```

### 2.2 其它环境
- **UE 5.5**：`/data/UE_5.5`
- **SpearSim 游戏可执行**：`/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh`
- **Xvfb**（无头渲染）：`DISPLAY=:99`（已有一个 `Xvfb :99 -screen 0 1280x720x24` 在跑）
- **Vulkan ICD**：`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`（NVIDIA RTX 4090 D）
- **GPU**：4 张 4090，GPU 0 空闲，GPU 1 可能被别人占着（共享机器）
- **运行游戏会话脚本的标准前缀**：
  ```bash
  cd /data/jzy/code/SPEAR
  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    /data/jzy/miniconda3/envs/spear-env/bin/python examples/<script>.py
  ```
- **运行编辑器 commandlet**（改资产用）：
  ```bash
  cd /data/jzy/code/SPEAR
  /data/jzy/miniconda3/envs/spear-env/bin/python tools/run_editor_script.py \
    --unreal-engine-dir /data/UE_5.5 \
    --script "$(pwd)/examples/<script>.py" \
    --launch-mode full --render-offscreen
  ```

---

## 3. SPEAR 编程模型（最少必要知识）

- `spear.Instance(config)` → 连一个 UE 应用。`instance.get_game()` 拿 game world 服务。
- 所有 UE 操作包在 `with instance.begin_frame():` ... `with instance.end_frame():` 配对里（同一个 UE 帧的开头/结尾执行）。
- UFUNCTION 用关键字调用（精确 UE 大小写）：`actor.SetActorScale3D(NewScale3D={...})`
- UPROPERTY 读用 `.get()`，写用直接赋值。
- `instance.step(num_frames=N)` 推进 N 帧（让 auto-exposure/TAA 沉淀）。
- 相机：spawn `BP_CameraSensor`，取 `DefaultSceneRoot.final_tone_curve_hdr_` 组件（`USpSceneCaptureComponent2D`），`comp.read_pixels()` 读帧。
- 详见 `/data/jzy/code/SPEAR/docs/agents.md`、`agents.mcp_usage.md`。

---

## 4. 文件地图

### 4.1 关键脚本（都在 `/data/jzy/code/SPEAR/examples/`）
| 脚本 | 作用 | 状态 |
|---|---|---|
| `my_import_asset.py` | `.glb` → StaticMesh + BP + 写 `tmp/asset_meta/<NAME>.json` | ✅ 能用（编辑器 commandlet） |
| `my_build_room.py` | 用 BasicShapes Cube 搭 5.2×4.4×2.8m 房间 + 点灯，墙贴 `MI_Walls`、地贴 `MI_Floor` | ⚠️ 能跑但**视觉差**（见第 5 节） |
| `preview_room.py` | 空房间 6 角度快照（墙 N/S/E/W + 天花板 + 地板） | ✅ 能用 |
| `render_asset_in_room.py` | 单物体 turntable：spawn 房间+物体+相机，36 帧 → mp4 | ✅ 能用（在 Cube 房间里） |
| `inspect_apartment_windows.py` | 扫描 apartment_0000 里窗户朝向（用 EditorLevelLibrary，已修过一次） | 未验证跑通 |

### 4.2 产物目录（`/data/jzy/code/SPEAR/tmp/`）
- `asset_meta/*.json` —— 100+ 物体的 bbox 元数据
- `preview_room/*.png` —— 空房间 6 角度（**当前的难看版本**，可拿来对比）
- `render_Clock/` —— 落地钟 turntable（`frame_0000.png` … `frame_0035.png` + `turntable.mp4`）**当前的难看版本**
- `apartment_layout.png` —— 一个没用的俯瞰（相机在 (0,0,800) 朝下，拍到的是公寓外的草地，不是室内）

### 4.3 UE 内资产路径
- **自建房间 BP**：`/Game/MyAssets/Room/BP_Room{Floor,Ceiling,WallXP,WallXN,WallYP,WallYN}` + `BP_RoomLight`
- **导入物体 BP**：`/Game/MyAssets/Blueprints/BP_Clock.BP_Clock_C` 等
- **🎯 那个好看的公寓场景**：`/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000`
  - 墙材质：`.../apartment_0000/Materials/Material_Instances/MI_Walls`
  - 地材质：`.../MI_Floor`
  - **HDRI 光照**：`.../MI_HDRI_Projection`（基于图像的光照，关键！）
  - **灯光管理**：`.../Debug/BP_LightManager`
  - 家具材质：`MI_Sofa`、`MI_Carpet`、`MI_Cabinet_Vase_Mirror`、`MI_LivingRoom_Table` 等（约 50 个）
  - 主材质：`.../Materials/Materials/M_Master`
  - 窗户相关：`MI_Casement`、`MI_Casement_Glass`、`MI_Vinyl_Frame`、`MI_Glass`

---

## 5. 卡点：为什么 Cube 房间 + 公寓材质 = 难看

业主原话："**一开始那个还有窗户的公寓房间，有很多家具，那个材质就很好，就拿那个的材质**"。

**关键认知（上一棒踩的坑）**：apartment_0000 好看 **≠ MI_Walls 这个材质文件本身好**。它好看是因为三件事**一起**对：
1. **正确 UV**：MI_Walls 的贴图坐标是按公寓墙体 mesh 的 UV 展开调的。把 MI_Walls 抠出来贴到 BasicShapes Cube（每面只有 0-1 UV）上，一张本该铺 1m² 的贴图被拉到 5m² 一整面墙 → 纹理细节全糊掉 → 看着像纯灰墙。
2. **HDRI 光照**：公寓用 `MI_HDRI_Projection`（基于图像的全局光照）+ 天光（config 里 `FORCE_SKYLIGHT_UPDATE: true`）。自建 Cube 房间只有**一个 800 流明点灯**，光平、无层次。
3. **LightManager**：公寓有 `BP_LightManager` 统一管灯。

所以"拿那个材质"的正确实现**不是把 MI_Walls 贴到 Cube 上**，而是下面第 6 节的方案 A（直接在那个场景里渲染）。

### 5.1 还有个独立的小问题：物体尺寸
- Hunyuan3D 的 mesh bbox 不准。例：`Clock.json` 的 `ext=199cm`（看着像落地钟），但上一棒按"桌面钟 35cm"缩，缩成个小圆顶。
- 房间 5.2×4.4m、相机 R 必须 ≤200cm（否则穿墙），大物体（落地钟/钢琴）没法退远拍全景 → 构图偏小。
- 这个问题在换到 apartment_0000 后**依然存在**（apartment 也不大），需要单独处理 per-asset TARGET。

---

## 6. 候选方案（业主倾向 A，但没拍板）

### 方案 A：直接在 apartment_0000 场景里渲染（业主原话最接近）
- `MAP = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"`
- 墙地窗本来就是对的材质 + 对的 UV + HDRI 光照，质感直接到位。
- **要解决的子问题**：
  - 家具遮挡：要么 `destroy_actor` 删家具、要么在空地放物体（apartment 有客厅/厨房空地）
  - apartment 偏暗：可能要调 `BP_LightManager` 或加灯
  - 物体放哪：需要一个俯瞰扫描找空地（现有 `scan_apartment.py` 相机位置错了，拍到草地，要修相机坐标到室内）
- **推荐**：最接近业主诉求，质感有保障。

### 方案 B：保留 Cube 房间，但修 UV + 加光照
- 复制 `MI_Walls` 到 `/Game/MyAssets/Room/MI_RoomWall`，改其 `UV Tiling`（或 `Texture Coordinate` 的 scale）让贴图每 ~1m 重复一次
- 加 HDRI/天光：spawn 一个 `BP_SkyLight` 或挂 `MI_HDRI_Projection`
- 工作量中等，效果不如 A 真实（毕竟是假房间）

### 方案 C：接受现状 —— 业主已否决（"太难看了"）

---

## 7. 现成 prompt（直接粘贴给能看图的 GPT）

> 下面每个 prompt 假设：你会把对应的图片/视频**作为附件**一起发给 GPT。文档（本文件）也一起发。

### Prompt 1 — 先建立"目标质感"参照（诊断前置）
```
我在用 Unreal Engine 5.5 + SPEAR 库做物体 360° 展示视频。我有一个自带的公寓场景 apartment_0000，
里面的墙/地材质和光照是我想要的目标质感。请看附件 apartment_interior_reference.png（apartment_0000 室内一帧）
和 cube_room_ugly.png（我自己用 Cube 搭的房间贴同款材质的结果）。

请对比这两张图，用 JSON 列出：
1) good_look_factors: 公寓那张"好看"具体好在哪里（光照/纹理/反射/景深…逐条）
2) ugly_look_causes: Cube 那张"难看"的每一条原因，并标注对应到 good_look_factors 里缺了哪个
3) is_uv_stretch_the_main_cause: 你判断 UV 拉伸是不是材质变平的主因（true/false + 理由）
4) is_lighting_the_main_cause: 光照（缺 HDRI/天光）是不是主因（true/false + 理由）
5) recommended_approach: A（在 apartment 场景里渲染）/ B（修 Cube 的 UV+光照）/ 其它，给一句话理由
```
**附件**：
- `cube_room_ugly.png` → 用 `/data/jzy/code/SPEAR/tmp/preview_room/wallN.png` 或 `floor.png`
- `apartment_interior_reference.png` → **目前没有！需要先在 apartment_0000 里渲染一张室内帧**（见 Prompt 1.5）

### Prompt 1.5 — 生成 apartment 室内参照帧（需要执行）
```
请在 SPEAR 里渲染一张 apartment_0000 的室内参照帧，用来当"目标质感"基准。
要求：
- 地图：/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000
- 相机放在室内客厅区域（apartment 的世界原点在室外，相机要平移到室内；可先 spawn 在 (0,0,180) 朝 +X 或 -X 看，不行就多试几个坐标，比如 (300, 0, 180)、(-300,0,180)、(0,300,180)）
- 分辨率 1280x720，渲染前 step 30 帧让 auto-exposure 沉淀
- 保存到 /data/jzy/code/SPEAR/tmp/apartment_reference/frame_0000.png
运行前缀：cd /data/jzy/code/SPEAR && DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json /data/jzy/miniconda3/envs/spear-env/bin/python <你的脚本>
可以参考 examples/scan_apartment.py（但它的相机坐标 (0,0,800) 朝下是错的，拍到草地，要改成室内人眼高度平视）。
```

### Prompt 2 — 决定走方案 A 还是 B（决策）
```
基于 Prompt 1 的诊断，请在以下两个方案里选一个并给出实施步骤：

方案 A：直接在 apartment_0000 场景里渲染物体 turntable。
  - 地图改为 /Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000
  - 处理家具遮挡（destroy_actor 删家具，或在空地放物体）
  - 处理 apartment 偏暗
方案 B：保留 /Game/MyAssets/Room/ 的 Cube 房间，复制 MI_Walls/MI_Floor 到 /Game/MyAssets/Room/ 下，
  改 UV Tiling 让贴图每 1m 重复，并 spawn 天光/HDRI。

约束：
- 必须用 /data/jzy/miniconda3/envs/spear-env/bin/python（见 HANDOFF 文档第 2 节，用错环境会静默失败）
- 物体尺寸：Hunyuan3D mesh bbox 不准（如 Clock ext=199cm 像落地钟），需要合理 TARGET
- 相机半径 R 必须 ≤200cm（房间几何约束，否则穿墙）

请输出：选定方案 + 分步实施代码（基于 examples/render_asset_in_room.py 改）。
```

### Prompt 3 — 实施方案 A（在公寓里渲染，删家具）
```
目标：写一个 examples/render_in_apartment.py，在 apartment_0000 场景里渲染 BP_Clock 的 360° turntable。

要求：
1. MAP = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
2. spawn BP_Clock 在公寓室内一块空地上（先渲染一张俯瞰找空地，或直接试客厅坐标）
3. 删除/隐藏会遮挡相机的家具：用 game.unreal_service 找所有 StaticMeshActor，
   按 actor 名字过滤家具类（Meshes/05_chair、06_sofa、07_table 等，见 v77 的 semantic 分类），
   destroy_actor 删掉，或移到地下。
   注意：apartment 的家具是 Level Sequence / Level Instance 加载的，可能要用 get_all_actors_of_class 遍历。
4. 相机绕物体转盘，R≤200cm，36 帧，输出 mp4
5. 用 /data/jzy/miniconda3/envs/spear-env/bin/python 跑

参考脚本：examples/render_asset_in_room.py（Cube 房间版，结构可复用，把 spawn 房间那段换成"加载 apartment 地图 + 删家具"）。
参考分类：/data/jzy/code/Spatial/v77_4ch_S2L/ 下有语义分类文档。
请直接写完整脚本，并在文档里标注每处不确定的 API 调用（我好去查 SPEAR docs/agents.mcp_usage.md）。
```

### Prompt 4 — 物体尺寸自动校准（独立子任务，A/B 都需要）
```
问题：Hunyuan3D 的 mesh bbox 不准（离群点撑大），导致 TARGET/ext 缩放后物体尺寸不对。
Clock 的 ext=199cm，但实际可见几何可能只有几十 cm。

请写一个编辑器 commandlet 脚本 examples/recompute_bounds.py：
- 遍历 /Game/MyAssets/Meshes/ 下所有 StaticMesh
- 用 SM 的真实顶点 bounds（排除离群点，比如用 99 分位数，或直接 GetBoundingBox()）
- 重算 ext/bmin_z/height，覆盖写到 /data/jzy/code/SPEAR/tmp/asset_meta/<NAME>.json
- 同时给每个资产一个建议的 realistic TARGET（按类别：钟~30cm 桌面钟 / ~180cm 落地钟；狗~80cm；钢琴~150cm；吉他~110cm…），
  存到 meta 里多一个字段 "suggested_target_cm"

运行：cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python tools/run_editor_script.py --unreal-engine-dir /data/UE_5.5 --script "$(pwd)/examples/recompute_bounds.py" --launch-mode full --render-offscreen
参考现有导入脚本：examples/my_import_asset.py（里面有用 smc.get_local_bounds() 的例子）。
```

---

## 8. 验收标准（业主心里"过"的样子）

- 房间墙地有**可见的真实纹理**（木地板有板缝、墙有质感），不是平灰
- 光照有层次（HDRI/天光带来的明暗 + 软阴影）
- 物体在画面里占 ≥20%、能认出是什么、站在地面上不悬空/穿模
- 360° 转盘每一帧物体都完整可见、不被墙/家具遮挡
- （后续）人形 Manny 能在房间里原地走路（骨骼动画播放）

---

## 9. 已知不要踩的坑（前两棒血泪）

1. **环境**：用 `spear-env`，别用 `thu`。`import spear; spear.__can_import_spear_ext__` 必须 True。
2. **脚本路径**：编辑器 commandlet 的 `--script` 要用**绝对路径** `$(pwd)/examples/x.py`，相对路径会被解析到 UE 二进制目录然后静默找不到。
3. **裸 except**：`spear/instance.py` 连接失败用裸 `except:` 吞异常，"连不上 RPC"几乎总是客户端侧问题（环境/模块），不是引擎。
4. **相机穿墙**：自建房间 5.2×4.4m，相机半径 R 必须 ≤200cm（Y 墙在 ±220），`R = min(R_FACTOR*TARGET, 200)`。
5. **不要碰** `/Game/SPEAR/Scenes/apartment_0000/` 下的原装资产（只读引用）。
6. **僵尸进程**：跑挂了的 SpearSim/UnrealEditor 要按 PID kill（别用 `pkill -f`，共享机器会误杀别人）。端口默认 30000。
7. **MI_Walls/MI_Floor 不能直接贴 Cube**：UV 会废。要么用公寓场景，要么复制后改 UV Tiling。

---

## 10. 一句话给 GPT 的开场

> "看完 HANDOFF_VISUAL_RENDER.md。业主要在 SPEAR 里复刻 apartment_0000 那种带窗户、有真实光照的公寓质感，把 Hunyuan3D 物体放进去做 360° 展示。现在管线能跑但视觉差（Cube 房间 + 抠出来的材质 = 平灰墙）。先用 Prompt 1 诊断（我会给你 Cube 房间和公寓的对比图），再走方案 A。务必用 spear-env。"
