// use super::TrackingObject;
use crate::{
    config::Config,
    object::TrackingObject,
    track_attributes::{TrackAttrMap, TrackAttributes},
};
use itertools::{Either, Itertools, MinMaxResult};
use noisy_float::types::{r64, R64};
use priority_matrix::PriorityMatrix;
use std::{collections::HashSet, hash::Hash};

#[derive(Debug, Clone)]
pub struct IouMatrix {
    matrix: PriorityMatrix<Index, Track, Iou>,
}

impl IouMatrix {
    pub fn remove_row_and_column(&mut self, pivot_row: &Index, pivot_col: &Track) {
        self.matrix.remove_row_and_column(&pivot_row, &pivot_col)
    }

    pub fn max(&self) -> Option<Element> {
        let entry = self.matrix.peek()?;
        Some(Element {
            box_idx: *entry.row,
            box_track_id: *entry.column,
            iou: *entry.weight,
        })
    }

    pub fn max_from_row(&self, row: &Index) -> Option<Element> {
        let entry = self.matrix.peek_from_row(row)?;
        Some(Element {
            box_idx: *entry.row,
            box_track_id: *entry.column,
            iou: *entry.weight,
        })
    }

    pub fn len(&self) -> usize {
        self.matrix.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Index(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Track(pub usize);

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Iou(pub R64);

pub struct IouMatrixBuilder {
    tracking_score_threshold: f64,
    length_weight_in_tracking_score: f64,
}

impl IouMatrixBuilder {
    pub fn new(config: &Config) -> Self {
        let Config {
            tracking_score_threshold,
            length_weight_in_tracking_score,
            ..
        } = *config;

        Self {
            tracking_score_threshold,
            length_weight_in_tracking_score,
        }
    }

    pub fn build_with_track_attr_map(
        &self,
        objects: &Vec<TrackingObject>,
        track_attr_hashmap: &TrackAttrMap,
        use_larger_bbox: bool,
    ) -> IouMatrix {
        // Split objects into two parts
        // - existing_track_ids: all track IDs of input objects
        // - unlabeled_objs: objects without track IDs along with their position indices
        let (used_track_ids, unlabeled_objs): (HashSet<Track>, Vec<(Index, &TrackingObject)>) =
            objects
                .iter()
                .enumerate()
                .partition_map(|(obj_idx, obj)| match obj.track_id {
                    Some(track_id) => Either::Left(Track(track_id)),
                    None => Either::Right((Index(obj_idx), obj)),
                });

        // Find tracks that are not associated with any objects.
        let unused_tracks: Vec<(&usize, &TrackAttributes)> = track_attr_hashmap
            .iter()
            .filter(|&(&prev_track_id, _)| !used_track_ids.contains(&Track(prev_track_id)))
            .collect();

        let matrix: PriorityMatrix<Index, Track, Iou> = unlabeled_objs
            .into_iter()
            .filter_map(|(curr_idx, curr_obj)| {
                let curr_bbox = &curr_obj.bbox;

                // Find tracks that overlaps on the curr_obj
                let candidate_pairs: Vec<Entry> = unused_tracks
                    .iter()
                    .filter_map(|&(prev_track_id, track_attr)| {
                        let predicted_track_bboxes =
                            track_attr.predict_track_bboxes(curr_obj.timestamp, 10);

                        let max_iou = predicted_track_bboxes
                            .into_iter()
                            // Compute IoU
                            .filter_map(|predicted_bbox| {
                                curr_bbox.iou_with(&predicted_bbox, use_larger_bbox)
                            })
                            // Keep boxes with IoU above a threshold
                            .filter(|&iou| iou > self.tracking_score_threshold)
                            .map(r64)
                            // Find the maximum IoU
                            .max()?;

                        // Make elements
                        Some(Entry {
                            obj_idx: curr_idx,
                            track_id: Track(*prev_track_id),
                            iou: Iou(max_iou),
                        })
                    })
                    .collect();

                // Get min and max track lengths
                let (min_length, max_length) = {
                    let minmax = candidate_pairs
                        .iter()
                        .map(|ele| track_attr_hashmap[&ele.track_id.0].objects.len())
                        .minmax();
                    match minmax {
                        MinMaxResult::NoElements => return None,
                        MinMaxResult::OneElement(val) => (val, val),
                        MinMaxResult::MinMax(min, max) => (min, max),
                    }
                };

                let weight = self.length_weight_in_tracking_score;
                let max_len_diff = max_length - min_length;

                // Re-compute IoU scores for each pair using track lengths
                let candidate_pairs = candidate_pairs
                    .into_iter()
                    .map(move |entry| {
                        let track_length = track_attr_hashmap[&entry.track_id.0].objects.len();
                        let len_diff = track_length - min_length;
                        let length_score = (len_diff + 1) as f64 / (max_len_diff + 1) as f64;
                        let new_iou = entry.iou.0 * (1.0 - weight) + length_score * weight;

                        Entry {
                            iou: Iou(new_iou),
                            ..entry
                        }
                    })
                    .filter(|entry| entry.iou.0 > self.tracking_score_threshold)
                    .map(|entry| {
                        let Entry {
                            obj_idx,
                            track_id,
                            iou,
                        } = entry;
                        (obj_idx, track_id, iou)
                    });

                Some(candidate_pairs)
            })
            .flatten()
            .collect();

        IouMatrix { matrix }
    }
}

#[derive(Clone, Debug)]
pub struct Element {
    pub box_idx: Index,
    pub box_track_id: Track,
    pub iou: Iou,
}

struct Entry {
    pub obj_idx: Index,
    pub track_id: Track,
    pub iou: Iou,
}
