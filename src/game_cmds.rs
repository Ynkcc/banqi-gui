//! 对局命令：重置 / 走子 / AI 行动 / 状态查询。

use crate::state::{
    extract_game_state, lock, AppState, GameState, MctsTreeHandle, OpponentType, StepResult,
};
use banqi_core::core::env::*;
use banqi_engine::engine::{Policy, RandomPolicy, RevealFirstPolicy};
use tauri::State;

#[cfg(any(feature = "torch", feature = "onnx"))]
use crate::mcts_view::mcts_search_blocking;
#[cfg(feature = "onnx")]
use banqi_engine::inference::onnx::OnnxMctsPolicy;
#[cfg(feature = "torch")]
use banqi_engine::engine::MctsDlPolicy;

// Tauri 命令：重置游戏
#[tauri::command]
pub(crate) fn reset_game(
    opponent: Option<String>,
    variant: Option<String>,
    state: State<AppState>,
) -> GameState {
    let mut game = lock(&state.game);
    let mut opp_type_lock = lock(&state.opponent_type);

    // 设置对手类型
    *opp_type_lock = match opponent.as_deref() {
        Some("Random") => OpponentType::Random,
        Some("RevealFirst") => OpponentType::RevealFirst,
        Some("Engine") => OpponentType::Engine,
        Some("MctsDL") => OpponentType::MctsDL,
        Some("MctsOnnx") => OpponentType::MctsOnnx,
        Some("Nnue") => OpponentType::Nnue,
        _ => OpponentType::PvP,
    };

    // 按变体重建环境：4x2 迷你暗棋（8 格 / 40 动作空间），
    // 4x4 暗棋（16 格 / 112 动作空间），其余 = 4x8 标准暗棋（32 格 / 352 动作空间）。
    // 变体字符串由 core 的 Variant 单一真源定义；构造器内部已 reset。
    *game = match variant.as_deref().and_then(Variant::from_str) {
        Some(Variant::DarkChess4x2) => DarkChessEnv::new_mini(),
        Some(Variant::DarkChess4x4) => DarkChessEnv::new_4x4(),
        _ => DarkChessEnv::new(),
    };

    // Nnue 对手：变体切换后校验特征维度，不匹配则清空（bot_move 会给出明确报错）
    if *opp_type_lock == OpponentType::Nnue {
        let mut eval_lock = lock(&state.nnue_evaluator);
        let dim_ok = eval_lock
            .as_ref()
            .map(|e| e.feature_dim == game.config.nnue_feature_dim())
            .unwrap_or(false);
        if !dim_ok {
            *eval_lock = None;
        }
    }

    // 重开后旧搜索树失效
    *lock(&state.mcts_tree) = None;

    // 若选择 MctsDL / MctsOnnx 且已有对应模型，创建策略实例
    #[cfg(feature = "torch")]
    if *opp_type_lock == OpponentType::MctsDL {
        let model_opt = lock(&state.model).clone();
        let mut policy_lock = lock(&state.mcts_policy);
        if let Some(model) = model_opt {
            let sims = *lock(&state.mcts_num_simulations);
            *policy_lock = Some(MctsDlPolicy::new(model, &*game, sims));
        } else {
            *policy_lock = None; // 未加载模型，策略不可用
        }
    } else {
        // 非 MctsDL 模式清空策略
        *lock(&state.mcts_policy) = None;
    }
    #[cfg(feature = "onnx")]
    if *opp_type_lock == OpponentType::MctsOnnx {
        let model_opt = lock(&state.onnx_model).clone();
        let mut policy_lock = lock(&state.onnx_policy);
        if let Some(model) = model_opt {
            let sims = *lock(&state.mcts_num_simulations);
            *policy_lock = Some(OnnxMctsPolicy::new(model, &*game, sims));
        } else {
            *policy_lock = None; // 未加载 ONNX 模型，策略不可用
        }
    } else {
        // 非 MctsOnnx 模式清空 ONNX 策略
        *lock(&state.onnx_policy) = None;
    }

    extract_game_state(&game)
}

// Tauri 命令：执行动作
#[tauri::command]
pub(crate) fn step_game(action: usize, state: State<AppState>) -> Result<StepResult, String> {
    let mut game = lock(&state.game);

    match game.step(action, None) {
        Ok((_reward, terminated, truncated, winner)) => {
            let state_data = extract_game_state(&game);
            Ok(StepResult {
                state: state_data,
                terminated,
                truncated,
                winner,
            })
        }
        Err(e) => Err(e.to_string()),
    }
}

// Tauri 命令：执行 AI 动作
#[tauri::command]
pub(crate) async fn bot_move(state: State<'_, AppState>) -> Result<StepResult, String> {
    let opp_type = *lock(&state.opponent_type);

    // 如果处于 PvP，提示前端无需调用 AI
    if opp_type == OpponentType::PvP {
        return Err("当前为本地双人模式，无需 AI 行动".to_string());
    }

    // 调用策略模块选择动作。
    // Engine / Nnue / MctsDL / MctsOnnx 为计算密集搜索，放后台线程执行避免阻塞 UI；
    // 其余策略廉价，直接同步执行。
    // MctsDL / MctsOnnx 搜索后保留整棵树到 mcts_tree 供前端可视化；其他对手清空旧树。
    let snapshot = *lock(&state.game);
    #[cfg_attr(not(any(feature = "torch", feature = "onnx")), allow(unused_mut))]
    let mut new_tree: Option<MctsTreeHandle> = None;
    let chosen_action = match opp_type {
        OpponentType::Engine => {
            let budget = *lock(&state.engine_budget);
            let cfg = banqi_core::core::expectimax::SearchConfig {
                node_budget: budget,
                ..Default::default()
            };
            tauri::async_runtime::spawn_blocking(move || {
                banqi_core::core::expectimax::search(&snapshot, &cfg).map(|r| r.action)
            })
            .await
            .map_err(|e| format!("引擎搜索线程错误: {e}"))?
        }
        OpponentType::Nnue => {
            let evaluator = lock(&state.nnue_evaluator)
                .clone()
                .ok_or("未加载 .nnue 模型，无法执行 NNUE 策略")?;
            let dim_ok = evaluator.feature_dim == snapshot.config.nnue_feature_dim();
            if !dim_ok {
                return Err(format!(
                    ".nnue 特征维度({})与当前变体({})不匹配，请切换变体或重新加载模型",
                    evaluator.feature_dim,
                    snapshot.config.nnue_feature_dim()
                ));
            }
            let depth = *lock(&state.nnue_depth);
            let budget = *lock(&state.nnue_budget);
            tauri::async_runtime::spawn_blocking(move || {
                let cfg = banqi_core::core::expectimax::SearchConfig {
                    node_budget: budget,
                    max_depth: depth,
                    nnue_evaluator: Some(evaluator),
                    ..Default::default()
                };
                banqi_core::core::expectimax::search(&snapshot, &cfg).map(|r| r.action)
            })
            .await
            .map_err(|e| format!("NNUE Expectimax 搜索线程错误: {e}"))?
        }
        #[cfg(feature = "torch")]
        OpponentType::MctsDL => {
            let model = lock(&state.model)
                .clone()
                .ok_or("未加载模型，无法执行 MCTS+DL 策略")?;
            let sims = *lock(&state.mcts_num_simulations);
            let handle = tauri::async_runtime::spawn_blocking(move || {
                mcts_search_blocking(OpponentType::MctsDL, &snapshot, sims, Some(model), None)
            })
            .await
            .map_err(|e| format!("MCTS 搜索线程错误: {e}"))??;
            let action = handle.chosen_action;
            new_tree = Some(handle);
            Some(action)
        }
        #[cfg(not(feature = "torch"))]
        OpponentType::MctsDL => return Err("MctsDL 需要启用 torch 特性".into()),
        #[cfg(feature = "onnx")]
        OpponentType::MctsOnnx => {
            let model = lock(&state.onnx_model)
                .clone()
                .ok_or("未加载 ONNX 模型，无法执行 MCTS+ONNX 策略")?;
            let sims = *lock(&state.mcts_num_simulations);
            let handle = tauri::async_runtime::spawn_blocking(move || {
                mcts_search_blocking(OpponentType::MctsOnnx, &snapshot, sims, None, Some(model))
            })
            .await
            .map_err(|e| format!("MCTS 搜索线程错误: {e}"))??;
            let action = handle.chosen_action;
            new_tree = Some(handle);
            Some(action)
        }
        #[cfg(not(feature = "onnx"))]
        OpponentType::MctsOnnx => return Err("MctsOnnx 需要启用 onnx 特性".into()),
        _ => {
            let game = lock(&state.game);
            match opp_type {
                OpponentType::RevealFirst => RevealFirstPolicy::choose_action(&game),
                OpponentType::Random => RandomPolicy::choose_action(&game),
                OpponentType::PvP => None, // 已在上面返回 Err，这里兜底
                OpponentType::Engine => None,
                OpponentType::Nnue => None,
                _ => None,
            }
        }
    }
    .ok_or_else(|| "AI 无棋可走".to_string())?;

    *lock(&state.mcts_tree) = new_tree;

    let mut game = lock(&state.game);
    match game.step(chosen_action, None) {
        Ok((_reward, terminated, truncated, winner)) => {
            let state_data = extract_game_state(&game);
            Ok(StepResult {
                state: state_data,
                terminated,
                truncated,
                winner,
            })
        }
        Err(e) => Err(e.to_string()),
    }
}

// Tauri 命令：获取当前状态
#[tauri::command]
pub(crate) fn get_game_state(state: State<AppState>) -> GameState {
    let game = lock(&state.game);
    extract_game_state(&game)
}

// Tauri 命令：获取对手类型
#[tauri::command]
pub(crate) fn get_opponent_type(state: State<AppState>) -> OpponentType {
    *lock(&state.opponent_type)
}

// Tauri 命令：获取移动动作编号
#[tauri::command]
pub(crate) fn get_move_action(from_sq: usize, to_sq: usize, state: State<AppState>) -> Option<usize> {
    let game = lock(&state.game);
    game.get_action_for_coords(&[from_sq, to_sq])
}
