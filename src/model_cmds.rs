//! 模型相关命令：能力探测、模型列表、目录打开与模型加载。

use crate::state::{lock, AppState};
use banqi_engine::nnue::NnueEvaluator;
use serde::Serialize;
use std::sync::Arc;
use tauri::State;

#[cfg(any(feature = "torch", feature = "onnx"))]
use crate::state::OpponentType;
#[cfg(feature = "onnx")]
use banqi_core::core::env::GameEnv;
#[cfg(feature = "onnx")]
use banqi_engine::inference::onnx::{OnnxMctsPolicy, OnnxModel};
#[cfg(feature = "torch")]
use banqi_engine::engine::{MctsDlPolicy, ModelWrapper};

/// 编译期启用的推理后端：前端据此隐藏无法使用的对手与模型类型。
#[derive(Debug, Clone, Serialize)]
pub struct Capabilities {
    torch: bool,
    onnx: bool,
}

#[tauri::command]
pub(crate) fn get_capabilities() -> Capabilities {
    Capabilities {
        torch: cfg!(feature = "torch"),
        onnx: cfg!(feature = "onnx"),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    name: String,
    path: String,
}

/// 递归收集目录下的 .pt / .onnx / .nnue 模型。
fn collect_models(dir: &std::path::Path, depth: usize, out: &mut Vec<ModelEntry>) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let path = e.path();
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            collect_models(&path, depth + 1, out);
        } else if ft.is_file() {
            let is_model = path
                .extension()
                .map(|x| x == "pt" || x == "onnx" || x == "nnue")
                .unwrap_or(false);
            if is_model {
                let name = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                out.push(ModelEntry {
                    name,
                    path: path.to_string_lossy().to_string(),
                });
            }
        }
    }
}

/// 固定模型目录：~/banqi-models。模型的搜索与存放都以它为准（用户直接复制文件进来即可）。
pub(crate) fn models_dir(home: &std::path::Path) -> std::path::PathBuf {
    home.join("banqi-models")
}

/// 模型列表：附带固定目录的绝对路径，供前端展示与提示。
#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    dir: String,
    models: Vec<ModelEntry>,
}

/// 列出固定模型目录（含子目录）下的 .pt / .onnx / .nnue 模型
#[tauri::command]
pub(crate) fn list_models(state: State<AppState>) -> ModelList {
    let mut models = Vec::new();
    collect_models(&state.models_dir, 0, &mut models);
    models.sort_by(|a, b| a.path.cmp(&b.path));
    ModelList {
        dir: state.models_dir.to_string_lossy().to_string(),
        models,
    }
}

/// 在系统文件管理器中打开固定模型目录（便于把模型复制进来）
#[tauri::command]
pub(crate) fn open_models_dir(app: tauri::AppHandle, state: State<AppState>) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let dir = state.models_dir.to_string_lossy().to_string();
    app.opener()
        .open_path(dir.clone(), None::<&str>)
        .map_err(|e| format!("打开模型目录失败 {dir}: {e}"))
}

/// 载入模型：按扩展名分派（.pt → TorchScript（需 torch feature），.onnx → ONNX，
/// .nnue → NNUE 叶评估（无需 feature））。
#[cfg(any(feature = "torch", feature = "onnx"))]
#[tauri::command]
pub(crate) fn load_model(path: String, state: State<AppState>) -> Result<String, String> {
    let ext = std::path::Path::new(&path)
        .extension()
        .map(|x| x.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "onnx" => load_onnx_model_impl(&path, &state),
        "nnue" => load_nnue_model_impl(&path, &state),
        _ => load_torch_model_impl(&path, &state),
    }
}

/// 加载 .nnue 评估器并按当前环境校验特征维度。
fn load_nnue_model_impl(path: &str, state: &State<AppState>) -> Result<String, String> {
    let evaluator = NnueEvaluator::load_from_file(path)
        .map_err(|e| format!("NNUE 模型加载失败: {e}"))?;
    let expected = lock(&state.game).config.nnue_feature_dim();
    evaluator
        .validate_feature_dim(expected)
        .map_err(|e| format!("{e}（当前变体维度 {expected}，请先选择匹配变体再加载）"))?;
    *lock(&state.nnue_evaluator) = Some(Arc::new(evaluator));
    Ok(format!("NNUE 模型已加载: {}", path))
}

#[cfg(feature = "onnx")]
fn load_onnx_model_impl(path: &str, state: &State<AppState>) -> Result<String, String> {
    let model = OnnxModel::new(path, "auto").map_err(|e| format!("ONNX 模型加载失败: {e}"))?;
    let arc_model = Arc::new(model);

    // 用当前局面做一次探针推理：模型输入形状与当前变体不符时立即失败，
    // 否则搜索阶段只会打印日志并退化成均匀策略（对手变成随机走子）。
    {
        let game = lock(&state.game);
        let obs = game.get_resnet_state();
        let (ch, rows, cols) = (
            obs.board.shape()[0],
            obs.board.shape()[1],
            obs.board.shape()[2],
        );
        let scalars = obs.scalars.len();
        let mut board_data = Vec::new();
        let mut scalars_data = Vec::new();
        game.encode_resnet_features_flat_into(&mut board_data, &mut scalars_data);
        arc_model
            .run(&board_data, &scalars_data, 1, ch, rows, cols, scalars)
            .map_err(|e| {
                format!("模型与当前变体不匹配（当前观测 {ch}x{rows}x{cols} / 标量 {scalars}）: {e}")
            })?;
    }

    *lock(&state.onnx_model) = Some(arc_model.clone());
    // 若当前为 MctsOnnx 且已有游戏，尝试重建策略
    if *lock(&state.opponent_type) == OpponentType::MctsOnnx {
        let sims = *lock(&state.mcts_num_simulations);
        let game = lock(&state.game);
        *lock(&state.onnx_policy) = Some(OnnxMctsPolicy::new(arc_model, &game, sims));
    }
    Ok(format!("ONNX 模型已加载: {}", path))
}

#[cfg(not(feature = "onnx"))]
#[allow(dead_code)]
fn load_onnx_model_impl(path: &str, _state: &State<AppState>) -> Result<String, String> {
    Err(format!("需要启用 onnx 特性才能加载 ONNX 模型（{path}）"))
}

#[cfg(feature = "torch")]
fn load_torch_model_impl(path: &str, state: &State<AppState>) -> Result<String, String> {
    let wrapper = ModelWrapper::load_from_file(path)?;
    let arc_wrapper = Arc::new(wrapper);
    *lock(&state.model) = Some(arc_wrapper.clone());
    // 若当前为 MctsDL 且已有游戏，尝试重建策略
    if *lock(&state.opponent_type) == OpponentType::MctsDL {
        let sims = *lock(&state.mcts_num_simulations);
        let game = lock(&state.game);
        *lock(&state.mcts_policy) = Some(MctsDlPolicy::new(arc_wrapper, &game, sims));
    }
    Ok(format!("模型已加载: {}", path))
}

#[cfg(not(feature = "torch"))]
#[allow(dead_code)]
fn load_torch_model_impl(path: &str, _state: &State<AppState>) -> Result<String, String> {
    Err(format!("需要启用 torch 特性才能加载 TorchScript 模型（{path}）"))
}

/// 无 torch / onnx 特性时的占位函数（.nnue 不依赖 feature，仍可加载）
#[cfg(not(any(feature = "torch", feature = "onnx")))]
#[tauri::command]
pub(crate) fn load_model(path: String, state: State<AppState>) -> Result<String, String> {
    let is_nnue = std::path::Path::new(&path)
        .extension()
        .map(|x| x == "nnue")
        .unwrap_or(false);
    if is_nnue {
        load_nnue_model_impl(&path, &state)
    } else {
        Err("需要启用 torch 或 onnx 特性才能加载该模型".into())
    }
}
