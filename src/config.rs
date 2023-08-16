use crate::weak_classifier::WeakClass;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Defines what the config file should contain.
pub struct Config {
    pub tracking_buffer_size: usize,
    pub tracking_score_threshold: f64,
    pub length_weight_in_tracking_score: f64,
    pub moving_threshold_speed_km: f64,
    pub easier_tracking: bool,
    pub weak_class_conf_vec: Option<Vec<WeakClass>>,
}
