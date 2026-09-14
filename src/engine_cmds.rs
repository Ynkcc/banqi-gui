//! 引擎参数命令：MCTS 搜索规模与各对手的搜索预算。

use crate::state::{lock, AppState};
use tauri::State;

/// 设置 MCTS 每步搜索次数
#[tauri::command]
pub(crate) fn set_mcts_iterations(iters: usize, state: State<AppState>) -> Result<usize, String> {
    if iters == 0 {
        return Err("搜索次数必须大于 0".into());
    }
    let mut sims = lock(&state.mcts_num_simulations);
    *sims = iters;

    #[cfg(feature = "torch")]
    if let Some(policy) = lock(&state.mcts_policy).as_mut() {
        policy.set_iterations(iters);
    }
    #[cfg(feature = "onnx")]
    if let Some(policy) = lock(&state.onnx_policy).as_mut() {
        policy.set_iterations(iters);
    }

    Ok(*sims)
}

/// 设置纯计算强引擎（Engine）的节点预算
#[tauri::command]
pub(crate) fn set_engine_budget(budget: u64, state: State<AppState>) -> Result<u64, String> {
    if budget == 0 {
        return Err("节点预算必须大于 0".into());
    }
    let mut b = lock(&state.engine_budget);
    *b = budget;
    Ok(*b)
}

/// 设置 NNUE 对手（Expectimax）的搜索深度
#[tauri::command]
pub(crate) fn set_nnue_depth(depth: i32, state: State<AppState>) -> Result<i32, String> {
    if depth <= 0 {
        return Err("搜索深度必须大于 0".into());
    }
    let mut d = lock(&state.nnue_depth);
    *d = depth;
    Ok(*d)
}

/// 设置 NNUE 对手（Expectimax）的节点预算
#[tauri::command]
pub(crate) fn set_nnue_budget(budget: u64, state: State<AppState>) -> Result<u64, String> {
    if budget == 0 {
        return Err("节点预算必须大于 0".into());
    }
    let mut b = lock(&state.nnue_budget);
    *b = budget;
    Ok(*b)
}
