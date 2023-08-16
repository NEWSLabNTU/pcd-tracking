use crate::{
    iou_matrix::IouMatrixBuilder, track_attr_updater::TrackAttributeUpdater,
    track_attributes::TrackAttributes, weak_classifier::WeakClassifier,
};

use std::{collections::HashMap, ops::RangeFrom};

pub struct Tracker {
    tracking_buffer_size: usize,
    track_attr_hashmap: HashMap<usize, TrackAttributes>,
    expired_track_attr_hashmap: HashMap<usize, TrackAttributes>,
    unique_id_iter: RangeFrom<usize>,
    weak_class_processor: Option<WeakClassifier>,
    track_attr_updater: TrackAttributeUpdater,
    iou_matrix_builder: IouMatrixBuilder,
    easier_tracking: bool,
}
