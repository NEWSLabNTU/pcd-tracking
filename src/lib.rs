pub mod config;
mod iou_matrix;
mod kalman;
pub mod object;
mod track_attr_updater;
mod track_attributes;
pub mod tracker;
pub mod weak_classifier;

pub use tracker::*;
