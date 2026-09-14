//! 应用状态、共享 DTO 与状态序列化辅助。
//!
//! **锁顺序约定**：同一命令内若需同时持有两把锁，按下列顺序获取，避免交叉嵌套：
//! `game` → `opponent_type` → `mcts_num_simulations` → `engine_budget` →
//! `nnue_depth`/`nnue_budget` → `nnue_evaluator` → `model`/`mcts_policy` →
//! `onnx_model`/`onnx_policy` → `mcts_tree`。
//! 仅取单个值时优先用临时 guard（`*lock(&state.x)`），不跨语句持有。

use banqi_core::core::env::*;
use banqi_core::core::mcts::MctsArena;
use banqi_engine::nnue::NnueEvaluator;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

#[cfg(feature = "onnx")]
use banqi_engine::inference::onnx::{OnnxMctsPolicy, OnnxModel};
#[cfg(feature = "torch")]
use banqi_engine::engine::{MctsDlPolicy, ModelWrapper};

/// 获取互斥锁；若此前有线程 panic 导致锁中毒，恢复内部数据继续。
///
/// 直接 `.lock().unwrap()` 会让任意一次命令内的 panic 永久污染该锁，
/// 之后所有命令在同一字段上连锁 panic（整个后端不可用）。此处显式恢复。
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// 游戏状态的可序列化版本
#[derive(Debug, Clone, Serialize)]
pub struct GameState {
    pub board: Vec<String>,
    pub current_player: String,
    pub move_counter: usize,
    pub total_step_counter: usize,
    pub dead_red: Vec<String>,
    pub dead_black: Vec<String>,
    pub hidden_red: Vec<String>,
    pub hidden_black: Vec<String>,
    pub action_masks: Vec<i32>,
    pub reveal_probabilities: Vec<f32>,
    pub bitboards: HashMap<String, Vec<bool>>,
    pub hp_red: i32,   // 红方血量
    pub hp_black: i32, // 黑方血量
    pub variant: String, // "4x8" = 标准暗棋, "4x2" = 迷你暗棋, "4x4" = 4x4 暗棋
}

#[derive(Debug, Clone, Serialize)]
pub struct StepResult {
    pub state: GameState,
    pub terminated: bool,
    pub truncated: bool,
    pub winner: Option<i32>,
}

// 对手类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OpponentType {
    PvP,         // 本地双人
    Random,      // 随机对手
    RevealFirst, // 优先翻棋
    Engine,      // 纯计算强引擎（αβ + Star1 + 置换表 + 迭代加深，节点预算可控；需 NNUE）
    MctsDL,      // MCTS + 深度学习（TorchScript，需 torch feature）
    MctsOnnx,    // MCTS + ONNX 深度学习（需 onnx feature，无需 libtorch）
    Nnue,        // Expectimax + NNUE 叶评估（加载 .nnue，无需 torch/onnx feature）
}

// 应用状态：包含游戏环境和当前对手设置
pub struct AppState {
    pub game: Mutex<DarkChessEnv>,
    pub opponent_type: Mutex<OpponentType>,
    // 强引擎节点预算
    pub engine_budget: Mutex<u64>,
    // MCTS 配置
    pub mcts_num_simulations: Mutex<usize>,
    // 已加载的模型（可在选择 MctsDL 时构建策略）
    #[cfg(feature = "torch")]
    pub model: Mutex<Option<Arc<ModelWrapper>>>,
    // MCTS+DL 策略（基于 DarkChessEnv，4x8/4x4/4x2 共用）
    #[cfg(feature = "torch")]
    pub mcts_policy: Mutex<Option<MctsDlPolicy<DarkChessEnv>>>,
    // 已加载的 ONNX 模型（MctsOnnx 对手，无需 libtorch）
    #[cfg(feature = "onnx")]
    pub onnx_model: Mutex<Option<Arc<OnnxModel>>>,
    #[cfg(feature = "onnx")]
    pub onnx_policy: Mutex<Option<OnnxMctsPolicy<DarkChessEnv>>>,
    // NNUE 对手：已加载的 .nnue 评估器 + 搜索参数
    pub nnue_evaluator: Mutex<Option<Arc<NnueEvaluator>>>,
    pub nnue_depth: Mutex<i32>,
    pub nnue_budget: Mutex<u64>,
    // 最近一次 Gumbel MCTS 搜索树（供前端按需浏览；非 MCTS 对手时为 None）
    pub mcts_tree: Mutex<Option<MctsTreeHandle>>,
    // 固定的模型目录（~/banqi-models）：模型搜索与存放的唯一位置
    pub models_dir: std::path::PathBuf,
}

/// 常驻后端的 MCTS 树：arena + 根索引 + 本次搜索选中的动作。
pub struct MctsTreeHandle {
    pub arena: MctsArena<DarkChessEnv>,
    pub root_idx: usize,
    pub chosen_action: usize,
}

pub fn player_label(p: Player) -> String {
    match p {
        Player::Red => "Red".to_string(),
        Player::Black => "Black".to_string(),
    }
}

// 辅助函数：获取棋子短名称
pub fn get_piece_short_name(piece: &Piece) -> String {
    let p_char = match piece.player {
        Player::Red => "R",
        Player::Black => "B",
    };
    let t_char = match piece.piece_type {
        PieceType::General => "Gen",
        PieceType::Advisor => "Adv",
        PieceType::Soldier => "Sol",
        PieceType::Cannon => "Can",
        PieceType::Horse => "Hor",
        PieceType::Chariot => "Car",
        PieceType::Elephant => "Ele",
    };
    format!("{}_{}", p_char, t_char)
}

// 辅助函数：从游戏环境中提取状态
pub fn extract_game_state(env: &DarkChessEnv) -> GameState {
    let board_slots = env.get_board_slots();
    let board: Vec<String> = board_slots
        .iter()
        .map(|slot| match slot {
            Slot::Empty => "Empty".to_string(),
            Slot::Hidden => "Hidden".to_string(),
            Slot::Revealed(piece) => get_piece_short_name(piece),
        })
        .collect();

    let current_player = match env.get_current_player() {
        Player::Red => "Red".to_string(),
        Player::Black => "Black".to_string(),
    };

    let dead_red: Vec<String> = env
        .get_dead_pieces(Player::Red)
        .iter()
        .map(|pt| format!("{:?}", pt))
        .collect();

    let dead_black: Vec<String> = env
        .get_dead_pieces(Player::Black)
        .iter()
        .map(|pt| format!("{:?}", pt))
        .collect();

    let hidden_red: Vec<String> = env
        .get_hidden_pieces(Player::Red)
        .iter()
        .map(|pt| format!("{:?}", pt))
        .collect();

    let hidden_black: Vec<String> = env
        .get_hidden_pieces(Player::Black)
        .iter()
        .map(|pt| format!("{:?}", pt))
        .collect();

    let action_masks = env.action_masks();
    let reveal_probabilities = project_reveal_probabilities(env.get_reveal_probabilities());
    let bitboards = env.get_bitboards();
    let hp_red = env.get_hp(Player::Red);
    let hp_black = env.get_hp(Player::Black);
    // 变体标识由 core 的 Variant 单一真源给出（"4x8"/"4x4"/"4x2"），避免按行列反推。
    let variant = env.config.variant.as_str().to_string();

    GameState {
        board,
        current_player,
        move_counter: env.get_move_counter(),
        total_step_counter: env.get_total_steps(),
        dead_red,
        dead_black,
        hidden_red,
        hidden_black,
        action_masks,
        reveal_probabilities,
        bitboards,
        hp_red,
        hp_black,
        variant,
    }
}

fn project_reveal_probabilities(raw: &[f32]) -> Vec<f32> {
    // 直接返回所有14个概率（红方7种棋子 + 黑方7种棋子）
    // 顺序: R_Sol, R_Can, R_Hor, R_Car, R_Ele, R_Adv, R_Gen, B_Sol, B_Can, B_Hor, B_Car, B_Ele, B_Adv, B_Gen
    raw.to_vec()
}
