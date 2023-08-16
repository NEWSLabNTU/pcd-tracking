use crate::{
    iou_matrix::{self, Element, IouMatrixBuilder},
    object::{BBox3D, OutputObject, TrackingObject},
    track_attr_updater::TrackAttributeUpdater,
    track_attributes::{TrackAttrMap, TrackAttributes},
    weak_classifier::{HistoricWeakClassUpdater, WeakClassifier},
};
use nalgebra::UnitQuaternion;
use noisy_float::prelude::r64;
use std::{collections::HashMap, mem, ops::RangeFrom};

pub struct Tracker {
    tracking_buffer_size: usize,
    track_attr_hashmap: HashMap<usize, TrackAttributes>,
    expired_track_attr_hashmap: HashMap<usize, TrackAttributes>,
    unique_id_iter: RangeFrom<usize>,
    weak_class_processor: Option<WeakClassProcessor>,
    track_attr_updater: TrackAttributeUpdater,
    iou_matrix_builder: IouMatrixBuilder,
    easier_tracking: bool,
}

pub struct WeakClassProcessor {
    weak_classifier: WeakClassifier,
    historic_weak_class_updater: HistoricWeakClassUpdater,
}

impl Tracker {
    pub fn track_objects_in_one_frame(
        mut self,
        bboxes: Vec<BBox3D>,
        frame_id: usize,
        timestamp: u64,
        is_last_frame: bool,
    ) -> Vec<OutputObject> {
        let objects: Vec<_> = bboxes
            .into_iter()
            .map(|bbox| TrackingObject {
                bbox,
                timestamp,
                frame_id,
                track_id: None,
                weak_class: None,
                best_match_track_id: None,
            })
            .collect();

        let objects = {
            let mut objects = objects;

            self.pair_up_tracks_and_objs(&mut objects, false);
            if self.easier_tracking {
                // Perform more aggresive tracking algorithm here
                self.pair_up_tracks_and_objs(&mut objects, true);
            }

            // Assign unique track id for each object that doesn't have a track id
            objects
                .iter_mut()
                .filter(|obj| obj.track_id.is_none())
                .zip(&mut self.unique_id_iter)
                .for_each(|(obj, new_track_id)| {
                    obj.track_id = Some(new_track_id);
                });
            objects
        };

        // Refine bboxes
        let objects = {
            let mut objects = objects;
            objects.iter_mut().for_each(|obj| {
                refine_bbox_heading(obj, &mut self.track_attr_hashmap);
            });
            objects
        };

        // Annotate object classes using weak class processor
        let objects = if self.weak_class_processor.is_some() {
            Self::annotate_weak_classes(
                &mut self.track_attr_hashmap,
                self.weak_class_processor.as_ref().unwrap(),
                objects,
            )
        } else {
            objects
        };

        self.track_attr_updater.update_track_attribute_map(
            &mut self.track_attr_hashmap,
            &objects,
            &self.expired_track_attr_hashmap,
        );

        // Remove expired track attributes
        let keys: Vec<usize> = self.track_attr_hashmap.keys().cloned().collect();
        for key in keys {
            let last_frame_id = {
                let track_attr = &self.track_attr_hashmap[&key];
                track_attr.objects.back().unwrap().frame_id
            };

            if frame_id - last_frame_id >= self.tracking_buffer_size || is_last_frame {
                let track_attr = self.track_attr_hashmap.remove(&key).unwrap();
                self.expired_track_attr_hashmap.insert(key, track_attr);
            }
        }

        // Clean expired track attribute hashmaps
        self.clean_expired_track_attributes(frame_id, is_last_frame);

        let output_objects = objects
            .into_iter()
            .map(|object| OutputObject {
                bbox: object.bbox,
                timestamp: object.timestamp,
                weak_class: object.weak_class,
                track_id: object.track_id,
            })
            .collect();
        output_objects
    }

    fn pair_up_tracks_and_objs(&self, objects: &mut Vec<TrackingObject>, use_larger_bbox: bool) {
        let mut iou_matrix = self.iou_matrix_builder.build_with_track_attr_map(
            objects,
            &self.track_attr_hashmap,
            use_larger_bbox, // don't enlarge bboxes
        );
        // Fill in best_match_track_id in each object
        objects
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, obj)| {
                let max_pair = iou_matrix.max_from_row(&iou_matrix::Index(idx))?;
                Some((obj, max_pair))
            })
            .for_each(|(obj, max_pair)| {
                let best_match_track = max_pair.box_track_id.0;
                obj.best_match_track_id = Some(best_match_track);
            });

        // Choose the max value in this matrix as track_id to be assigned
        while let Some(max_elem) = iou_matrix.max() {
            let Element {
                box_idx,
                box_track_id,
                ..
            } = max_elem;

            iou_matrix.remove_row_and_column(&box_idx, &box_track_id);
            objects[box_idx.0].track_id = Some(box_track_id.0);
        }
    }

    fn annotate_weak_classes(
        track_attr_hashmap: &mut TrackAttrMap,
        weak_class_processor: &WeakClassProcessor,
        mut objects: Vec<TrackingObject>,
    ) -> Vec<TrackingObject> {
        objects.iter_mut().for_each(|object| {
            let weak_class = weak_class_processor
                .weak_classifier
                .classify_one(&object.bbox);
            if let Some(weak_class) = weak_class {
                object.weak_class = Some(weak_class.class.clone());
            }
        });

        let objects = weak_class_processor
            .historic_weak_class_updater
            .update_class_from_historic_classes(objects, track_attr_hashmap);

        objects
    }

    fn clean_expired_track_attributes(&mut self, frame_id: usize, clear_all: bool) {
        if clear_all {
            self.expired_track_attr_hashmap = HashMap::new();
            return;
        }

        let keys_to_remove: Vec<usize> = self
            .expired_track_attr_hashmap
            .iter()
            .filter(|(_, track_attr)| {
                let last_frame_id = track_attr.objects.back().unwrap().frame_id;
                frame_id - last_frame_id >= 3000
            })
            .map(|(key, _)| *key)
            .collect();

        for key in keys_to_remove {
            self.expired_track_attr_hashmap.remove(&key).unwrap();
        }
    }
}

fn refine_bbox_heading(obj: &mut TrackingObject, track_attr_map: &mut TrackAttrMap) {
    let bbox = &mut obj.bbox;
    let track_id = obj
        .best_match_track_id
        .unwrap_or_else(|| obj.track_id.unwrap());
    let Some(track_attr) = track_attr_map.get_mut(&track_id) else {
        obj.change_bbox_heading_to_long_axis();
        return;
    };
    let Some(move_dir) = track_attr.facing_direction else {
        obj.change_bbox_heading_to_long_axis();
        return;
    };

    // Change BBox heading to moving direction without changing BBox appearance
    let bbox_dirs = [
        bbox.direction_front(),
        bbox.direction_back(),
        bbox.direction_left(),
        bbox.direction_right(),
    ];
    let min_dir = bbox_dirs
        .iter()
        .min_by_key(|dir| r64(move_dir.angle(dir)))
        .unwrap();
    let min_extent = min_dir.norm() * 2.0;

    if (min_extent - bbox.size_x).abs() > (min_extent - bbox.size_y).abs() {
        mem::swap(&mut bbox.size_x, &mut bbox.size_y);
    }

    let yaw = min_dir.y.atan2(min_dir.x);
    let rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, yaw);

    bbox.rotation = rotation;
}
