use crate::weak_classifier::WeakClass;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Defines what the config file should contain.
pub struct Config {
    /// The maximum delay time for a track to be missing.
    pub tracking_buffer_sec: f64,
    /// Tracking score threshold to match a track.
    pub tracking_score_threshold: f64,
    /// The weight in tracked length factor when conducting matching.
    pub length_weight_in_tracking_score: f64,
    /// The threshold in kilometers per hour to consider a object as moving.
    pub moving_threshold_speed_km: f64,
    /// Use larger bbox to compute IoU when conducting matching.
    pub easier_tracking: bool,
    /// The config for weak classifier.
    pub weak_class_conf_vec: Option<Vec<WeakClass>>,
}
