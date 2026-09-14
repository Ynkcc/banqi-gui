//! banqi-gui：Tauri 桌面端入口。
//!
//! 文件划分：
//!   - state.rs       应用状态（`AppState`）、共享 DTO、锁辅助与状态序列化
//!   - game_cmds.rs   对局命令（重置 / 走子 / AI 行动 / 状态查询）
//!   - engine_cmds.rs 引擎参数命令（MCTS 规模 / 各对手预算）
//!   - model_cmds.rs  模型能力探测、列表与加载
//!   - mcts_view.rs   MCTS 搜索树可视化命令

mod engine_cmds;
mod game_cmds;
mod mcts_view;
mod model_cmds;
mod state;

use banqi_core::core::env::DarkChessEnv;
use std::sync::Mutex;
use tauri::Manager;

use engine_cmds::*;
use game_cmds::*;
use mcts_view::*;
use model_cmds::*;
use state::{AppState, OpponentType};

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // 固定模型目录：~/banqi-models，不存在则创建，用户可直接把模型复制进来
            let dir = models_dir(&app.path().home_dir()?);
            if let Err(e) = std::fs::create_dir_all(&dir) {
                eprintln!("创建模型目录失败 {}: {e}", dir.display());
            }
            // 初始化游戏环境和状态
            let env = DarkChessEnv::new();
            app.manage(AppState {
                game: Mutex::new(env),
                opponent_type: Mutex::new(OpponentType::PvP),
                engine_budget: Mutex::new(300_000),
                mcts_num_simulations: Mutex::new(200),
                #[cfg(feature = "torch")]
                model: Mutex::new(None),
                #[cfg(feature = "torch")]
                mcts_policy: Mutex::new(None),
                #[cfg(feature = "onnx")]
                onnx_model: Mutex::new(None),
                #[cfg(feature = "onnx")]
                onnx_policy: Mutex::new(None),
                nnue_evaluator: Mutex::new(None),
                nnue_depth: Mutex::new(8),
                nnue_budget: Mutex::new(200_000),
                mcts_tree: Mutex::new(None),
                models_dir: dir,
            });
            Ok(())
        })
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            reset_game,
            step_game,
            bot_move,
            get_game_state,
            get_opponent_type,
            get_move_action,
            get_capabilities,
            list_models,
            open_models_dir,
            load_model,
            set_mcts_iterations,
            set_engine_budget,
            set_nnue_depth,
            set_nnue_budget,
            mcts_get_root,
            mcts_get_children,
            mcts_get_node_detail,
            mcts_get_node_board,
            mcts_search
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() {
    run();
}
