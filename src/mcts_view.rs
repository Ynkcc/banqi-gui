//! MCTS 搜索树可视化：树节点 DTO、只读查询命令与手动搜索。

use crate::state::{
    extract_game_state, lock, player_label, AppState, GameState, MctsTreeHandle, OpponentType,
};
use banqi_core::core::env::DarkChessEnv;
use banqi_core::core::mcts::{GumbelConfig, MctsArena};
use serde::Serialize;
use tauri::State;

#[cfg(any(feature = "torch", feature = "onnx"))]
use banqi_core::core::mcts::GumbelMCTS;
#[cfg(any(feature = "torch", feature = "onnx"))]
use std::sync::Arc;

/// 统一的"边"摘要：普通节点取 children，机会节点取 possible_states。
#[derive(Debug, Clone, Serialize)]
pub struct MctsEdge {
    child_id: usize,
    /// 普通边 = 动作索引；机会边 = outcome_id
    action: usize,
    prior: f32,
    logit: f32,
    n: u32,
    q: f32,
    health_q: f32,
    is_chance: bool,
    chance_prob: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct MctsNodeSummary {
    id: usize,
    n: u32,
    q: f32,
    health_q: f32,
    player: String,
    is_chance: bool,
    is_terminal: bool,
    is_expanded: bool,
    edge_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MctsRootInfo {
    root: MctsNodeSummary,
    chosen_action: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MctsNodeDetail {
    id: usize,
    prior: f32,
    logit: f32,
    n: u32,
    q: f32,
    health_q: f32,
    initial_value: f32,
    player: String,
    is_chance: bool,
    is_terminal: bool,
    is_expanded: bool,
    child_count: usize,
    outcome_count: usize,
}

/// 搜索树某节点的完整局面（信息状态视角：暗子仍以 Hidden 表示，不泄露真实身份）。
#[derive(Debug, Clone, Serialize)]
pub struct MctsNodeBoard {
    state: GameState,
    /// 产生该局面的动作涉及格：翻棋/吃暗子 1 格，移动/炮击 2 格（from, to）；无则为空。
    move_coords: Vec<usize>,
}

/// 收集某节点的全部子边（普通 + 机会），按 N 降序、再按先验降序。
fn mcts_edge_list(arena: &MctsArena<DarkChessEnv>, node_id: usize) -> Vec<MctsEdge> {
    let node = arena.get(node_id);
    let mut edges: Vec<MctsEdge> = node
        .children
        .iter()
        .map(|(action, idx)| {
            let c = arena.get(*idx);
            MctsEdge {
                child_id: *idx,
                action: *action,
                prior: c.prior,
                logit: c.logit,
                n: c.visit_count,
                q: c.q_value(),
                health_q: c.q_health_value(),
                is_chance: false,
                chance_prob: 1.0,
            }
        })
        .collect();
    edges.extend(node.possible_states.iter().map(|(outcome_id, prob, idx)| {
        let c = arena.get(*idx);
        MctsEdge {
            child_id: *idx,
            action: *outcome_id,
            prior: c.prior,
            logit: c.logit,
            n: c.visit_count,
            q: c.q_value(),
            health_q: c.q_health_value(),
            is_chance: true,
            chance_prob: *prob,
        }
    }));
    edges.sort_by(|a, b| {
        b.n.cmp(&a.n)
            .then(b.prior.partial_cmp(&a.prior).unwrap_or(std::cmp::Ordering::Equal))
    });
    edges
}

fn mcts_node_summary(arena: &MctsArena<DarkChessEnv>, node_id: usize) -> MctsNodeSummary {
    let node = arena.get(node_id);
    let edge_count = node.children.len() + node.possible_states.len();
    MctsNodeSummary {
        id: node_id,
        n: node.visit_count,
        q: node.q_value(),
        health_q: node.q_health_value(),
        player: player_label(node.player),
        is_chance: node.is_chance_node,
        is_terminal: node.is_terminal,
        is_expanded: node.is_expanded,
        edge_count,
    }
}

fn mcts_root_info(handle: &MctsTreeHandle) -> MctsRootInfo {
    MctsRootInfo {
        root: mcts_node_summary(&handle.arena, handle.root_idx),
        chosen_action: handle.chosen_action,
    }
}

/// 阻塞执行一次 Gumbel MCTS 搜索并保留树。仅支持 MctsDL / MctsOnnx。
#[allow(unused_variables)]
pub(crate) fn mcts_search_blocking(
    opp: OpponentType,
    env: &DarkChessEnv,
    sims: usize,
    torch_model: TorchModelOpt,
    onnx_model: OnnxModelOpt,
) -> Result<MctsTreeHandle, String> {
    let config = GumbelConfig::with_search_scale(sims.max(1), 16);
    match opp {
        #[cfg(feature = "torch")]
        OpponentType::MctsDL => {
            let model = torch_model.ok_or("未加载 .pt 模型，无法执行 MCTS+DL 搜索")?;
            let evaluator = banqi_engine::engine::TchEvaluator::<DarkChessEnv>::new(model);
            let mut mcts = GumbelMCTS::new(env, &evaluator, config);
            let result = mcts
                .run()
                .map_err(|e| e.to_string())?
                .ok_or_else(|| "MCTS 搜索无合法动作".to_string())?;
            Ok(MctsTreeHandle {
                root_idx: mcts.root_idx,
                arena: mcts.arena,
                chosen_action: result.action,
            })
        }
        #[cfg(not(feature = "torch"))]
        OpponentType::MctsDL => Err("MctsDL 需要启用 torch 特性".into()),
        #[cfg(feature = "onnx")]
        OpponentType::MctsOnnx => {
            let model = onnx_model.ok_or("未加载 .onnx 模型，无法执行 MCTS+ONNX 搜索")?;
            let evaluator = banqi_engine::inference::onnx::OnnxEvaluator::<DarkChessEnv>::new(model);
            let mut mcts = GumbelMCTS::new(env, &evaluator, config);
            let result = mcts
                .run()
                .map_err(|e| e.to_string())?
                .ok_or_else(|| "MCTS 搜索无合法动作".to_string())?;
            Ok(MctsTreeHandle {
                root_idx: mcts.root_idx,
                arena: mcts.arena,
                chosen_action: result.action,
            })
        }
        #[cfg(not(feature = "onnx"))]
        OpponentType::MctsOnnx => Err("MctsOnnx 需要启用 onnx 特性".into()),
        _ => Err("MCTS 树可视化仅支持 MctsDL / MctsOnnx 对手".into()),
    }
}

#[cfg(feature = "torch")]
type TorchModelOpt = Option<Arc<banqi_engine::engine::ModelWrapper>>;
#[cfg(not(feature = "torch"))]
type TorchModelOpt = Option<()>;

#[cfg(feature = "onnx")]
type OnnxModelOpt = Option<Arc<banqi_engine::inference::onnx::OnnxModel>>;
#[cfg(not(feature = "onnx"))]
type OnnxModelOpt = Option<()>;

/// 获取当前 MCTS 树根（前端打开面板时调用）
#[tauri::command]
pub(crate) fn mcts_get_root(state: State<AppState>) -> Result<MctsRootInfo, String> {
    let guard = lock(&state.mcts_tree);
    let tree = guard
        .as_ref()
        .ok_or("暂无 MCTS 搜索树：请先让 MctsDL / MctsOnnx 对手走一步，或在搜索树面板手动搜索")?;
    Ok(mcts_root_info(tree))
}

/// 按需读取某节点的全部子边（懒展开）
#[tauri::command]
pub(crate) fn mcts_get_children(
    node_id: usize,
    state: State<AppState>,
) -> Result<Vec<MctsEdge>, String> {
    let guard = lock(&state.mcts_tree);
    let tree = guard.as_ref().ok_or("暂无 MCTS 搜索树")?;
    Ok(mcts_edge_list(&tree.arena, node_id))
}

/// 读取某节点完整统计（tooltip 用）
#[tauri::command]
pub(crate) fn mcts_get_node_detail(
    node_id: usize,
    state: State<AppState>,
) -> Result<MctsNodeDetail, String> {
    let guard = lock(&state.mcts_tree);
    let tree = guard.as_ref().ok_or("暂无 MCTS 搜索树")?;
    let node = tree.arena.get(node_id);
    Ok(MctsNodeDetail {
        id: node_id,
        prior: node.prior,
        logit: node.logit,
        n: node.visit_count,
        q: node.q_value(),
        health_q: node.q_health_value(),
        initial_value: node.initial_value,
        player: player_label(node.player),
        is_chance: node.is_chance_node,
        is_terminal: node.is_terminal,
        is_expanded: node.is_expanded,
        child_count: node.children.len(),
        outcome_count: node.possible_states.len(),
    })
}

/// 读取某节点保存的完整局面（含到达该节点的着法所在格）。
#[tauri::command]
pub(crate) fn mcts_get_node_board(
    node_id: usize,
    state: State<AppState>,
) -> Result<MctsNodeBoard, String> {
    let guard = lock(&state.mcts_tree);
    let tree = guard.as_ref().ok_or("暂无 MCTS 搜索树")?;
    let node = tree.arena.get(node_id);
    let env = node
        .env
        .as_ref()
        .ok_or_else(|| format!("节点 {node_id} 未保存局面"))?;
    // 机会节点保存的是「翻棋前」局面，其 last_action 仍指向父节点的着法；
    // 故改从任一机会结果子节点取回真正的翻棋动作。
    let action = if node.is_chance_node {
        node.possible_states
            .first()
            .and_then(|(_, _, idx)| tree.arena.get(*idx).env.as_ref())
            .and_then(|e| e.get_last_action())
            .or_else(|| env.get_last_action())
    } else {
        env.get_last_action()
    };
    let move_coords = action
        .and_then(|a| env.get_coords_for_action(a))
        .unwrap_or_default();
    Ok(MctsNodeBoard {
        state: extract_game_state(env),
        move_coords,
    })
}

/// 在当前局面上手动触发一次 MCTS 搜索并替换浏览树
#[tauri::command]
pub(crate) async fn mcts_search(state: State<'_, AppState>) -> Result<MctsRootInfo, String> {
    let opp = *lock(&state.opponent_type);
    #[cfg(feature = "torch")]
    let torch_model = lock(&state.model).clone();
    #[cfg(not(feature = "torch"))]
    let torch_model = None;
    #[cfg(feature = "onnx")]
    let onnx_model = lock(&state.onnx_model).clone();
    #[cfg(not(feature = "onnx"))]
    let onnx_model = None;
    let snapshot = *lock(&state.game);
    let sims = *lock(&state.mcts_num_simulations);

    let handle = tauri::async_runtime::spawn_blocking(move || {
        mcts_search_blocking(opp, &snapshot, sims, torch_model, onnx_model)
    })
    .await
    .map_err(|e| format!("MCTS 搜索线程错误: {e}"))??;

    let info = mcts_root_info(&handle);
    *lock(&state.mcts_tree) = Some(handle);
    Ok(info)
}
