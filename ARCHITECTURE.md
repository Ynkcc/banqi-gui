# ARCHITECTURE — banqi-tauri（Tauri 桌面端）

> **仓库规划**：本目录计划拆分为独立仓库 `banqi-tauri`（已是独立 crate / workspace member）。
> 拆分前随主仓库维护；本文与主仓库 `docs/ARCHITECTURE.md` §5 对应，以本文为准。

## 1. 定位与结构

Tauri 2 桌面 GUI：独立 crate `banqi-tauri`（path 依赖 `banqi-core` + `banqi-engine`——**后续切 git 依赖**；领域核心在 `banqi-core`，策略/torch/onnx/NNUE 推理在 `banqi-engine`）。

| 条目 | 说明 |
|---|---|
| `Cargo.toml` / `build.rs`（GTK/WebKit 预检 + `tauri_build::build()`）/ `tauri.conf.json` / `icons/` | crate 配套 |
| `src/main.rs` | 入口，持 `AppState`（环境 + 各引擎） |
| `frontend/` | Vite + Vue 3 + TypeScript 工程（源码 `src/`，构建产物 `dist/`；Tauri 加载 `frontendDist=./frontend/dist`，dev 经 devUrl http://localhost:5173） |

## 2. 前端

- `src/api/types.ts` + `src/api/client.ts`：command 返回类型定义与 `invoke` 封装（`api.*`）；
- `src/composables/`：`useGame`（对局状态/走子/高亮/日志/机器人回合）、`useSettings`（变体/对手/模型列表与参数、后端能力探测）、`useMctsTree`（搜索树懒加载缓存 + 布局）、`useToast`、`useLogs`；
- `src/components/`：`App`（三栏布局 + 抽屉）、`BoardGrid`（受控棋盘渲染，接收 `board`/`variant`/高亮/标记 props，供主棋盘与搜索树节点局面共用）、`BoardView`（`useGame` 数据的薄包装）、`ControlPanel`、`StatusPanel`、`LogPanel`、`PieceTray`、`BitboardPanel`、`MctsTreePanel`、`ToastHost`；`src/domain/pieces.ts` 棋子/位板常量。

## 3. `#[tauri::command]` 列表

- 对局：`reset_game`、`step_game`、`bot_move`、`get_game_state`、`get_move_action`、`get_opponent_type`
- 能力探测：`get_capabilities`（返回编译期 feature 启用的推理后端 `torch`/`onnx`；前端据此隐藏 MctsDL/MctsOnnx 对手与对应的 `.pt`/`.onnx` 模型项）
- 模型：`list_models`（返回固定模型目录 `~/banqi-models` 绝对路径 + 其中递归收集的 `.pt`/`.onnx`/`.nnue`）、`open_models_dir`（文件管理器中打开该目录，Rust 侧经 `tauri-plugin-opener` 调用，无需前端 capability 配置）、`load_model`（`.onnx` 加载后用当前局面做一次探针推理，模型输入形状与当前变体不符即报错，避免搜索期静默退化为均匀策略）
- 引擎参数：`set_minimax_depth`、`set_mcts_iterations`、`set_engine_budget`、`set_heuristic_sims`、`set_nnue_depth`、`set_nnue_budget`
- MCTS 树可视化（懒加载）：`mcts_get_root`、`mcts_get_children`、`mcts_get_node_detail`、`mcts_get_node_board`（返回某节点保存的完整局面 + 到达该节点的着法所在格，机会节点取其机会结果子节点的动作）、`mcts_search`。MctsDL/MctsOnnx 落子后整棵 `MctsArena<DarkChessEnv>` 常驻 `AppState.mcts_tree`，前端按需逐节点拉取子边渲染（抽屉内左右分栏：左侧 SVG 树 + 主变例路径高亮，右侧节点检视区展示只读棋盘与搜索统计，机会节点 outcome 亦懒展开）

## 4. 构建/开发

- Rust：主仓库内 `cargo build -p banqi-tauri`（本目录内直接 `cargo build` 亦可；torch/onnx 对手需 `-p banqi-tauri --features torch,onnx`）；
- 前端（`frontend/` 内）：`npm run dev`（Vite，端口 5173）、`npm run build`（vue-tsc + vite build → `dist/`）。

## 5. 变更记录

- 2026-09-11：从主仓库 `docs/ARCHITECTURE.md` §5 拆出，作为未来独立仓库的架构文档。
- 2026-09-11：依赖切换：`banqi_4x8`（主仓库 path）→ `banqi-core` + `banqi-engine`（各自 path；feature `torch`/`onnx` 透传 `banqi-engine`）；`main.rs` 导入路径同步改写（core→`banqi_core::core`，engine/inference/nnue→`banqi_engine`）。
- 2026-09-13：`load_model`（`.onnx`）新增变体匹配校验：加载后以当前局面做一次 1-batch 探针推理，形状不符直接返回错误（含当前观测维度），修复「模型与变体不匹配时 ONNX 推理失败被吞掉、对手静默变成均匀随机策略」的问题。
- 2026-09-13：模型路径固定化：模型搜索/存放统一到 `~/banqi-models`（`AppState.models_dir`，`setup` 时创建，不再扫描相对 CWD 的 `python/outputs`、`outputs`）；`list_models` 返回 `{ dir, models }`；新增 `open_models_dir` 命令与 `tauri-plugin-opener` 依赖；前端 ControlPanel 新增「模型目录」卡片（展示绝对路径 + 打开按钮）。
- 2026-09-13：新增 `get_capabilities` 命令；前端按后端已编译 feature 过滤对手与模型列表（未启用 torch 即不显示 MctsDL 与 `.pt` 模型）。棋盘改为由 `--cell`（容器可用宽高与行列数取 min）推导尺寸，格子恒为正方形且不再溢出容器（修复 4x2 需滚动、棋子变形、托盘/血量卡片被棋盘压住）。
- 2026-09-14：变体词表统一到 `banqi-core` 的 `Variant`（"4x8" / "4x4" / "4x2"）：`extract_game_state` 直接输出 `env.config.variant.as_str()`（不再按行列反推），`reset_game` 经 `Variant::from_str` 解析；前端 `api/types.ts` / `ControlPanel` / `useSettings` / `useGame` / `StatusPanel` / `domain/pieces.ts` 同步由 `"dark"` / `"mini"` 改为 `"4x8"` / `"4x2"`。
- 2026-09-14：搜索树面板可读性改造：①新增 `mcts_get_node_board` 命令（复用 `extract_game_state`，配合 `banqi-core` 新增的 `get_last_action()` 给出到达着法所在格）；②棋盘渲染抽为受控组件 `BoardGrid`（`BoardView` 改为薄包装），供主棋盘与节点局面共用；③`MctsTreePanel` 由单列改为左右分栏（抽屉宽度 `min(720px,85vw)` → `min(1240px,94vw)`），点击节点在右侧检视区展示只读棋盘 + N/Q/Q_hp/prior/logit/V 先验等统计，移除原 hover tooltip；④新增从根沿最大 N 的主变例（PV）路径高亮（蓝色连线/描边，根实际选择仍为红色）。
