use crate::kalman::{KalmanFilter, KalmanState};
use crate::object::{BBox3D, TrackingObject};
use itertools::Itertools;
use nalgebra::{matrix, Vector2, Vector3};
use noisy_float::prelude::r64;
use std::{
    collections::{HashMap, VecDeque},
    iter,
};

pub type TrackAttrMap = HashMap<usize, TrackAttributes>;

#[derive(Clone, Debug)]
pub struct TrackAttributes {
    pub track_id: usize,
    pub bbox_list: Vec<BBox3D>,
    pub historic_class_names: Vec<String>,
    pub objects: VecDeque<TrackingObject>,
    pub avg_size: Vector3<f64>,
    pub kalman_filter: KalmanFilter,
    pub kalman_states: Vec<KalmanState>,
    pub facing_direction: Option<Vector2<f64>>,
}

impl TrackAttributes {
    pub fn new(
        track_id: usize,
        object: &TrackingObject,
        track_attr_map: &TrackAttrMap,
        expired_track_attr_map: &TrackAttrMap,
    ) -> TrackAttributes {
        let mut historic_class_names = vec![];
        if let Some(class) = &object.weak_class {
            historic_class_names.push(class.clone())
        }

        let facing_direction =
            guess_facing_direction_from_other_tracks(object, expired_track_attr_map);
        let kalman_filter = predict_kalman_filter_from_other_tracks(object, track_attr_map);

        TrackAttributes {
            track_id,
            bbox_list: vec![object.bbox.clone()],
            historic_class_names,
            objects: [object.clone()].into_iter().collect(),
            avg_size: Vector3::from([object.bbox.size_x, object.bbox.size_y, object.bbox.size_z]),
            kalman_filter: kalman_filter.clone(),
            kalman_states: vec![kalman_filter.state],
            facing_direction,
        }
    }
}

impl TrackAttributes {
    // Return the speed of the object predicted by kalman-filter
    pub fn current_speed(&self) -> f64 {
        let curr_state = self.kalman_states.last().unwrap();
        let speed = Vector2::from([curr_state.vel_x, curr_state.vel_y]).norm() as f64;
        speed
    }
    // Return the moving direction of the object predicted by kalman-filter
    pub fn current_direction(&self) -> Vector2<f64> {
        let curr_state = self.kalman_states.last().unwrap();
        let direction =
            Vector2::from([curr_state.vel_x as f64, curr_state.vel_y as f64]).normalize();
        direction
    }

    pub fn predict_track_bboxes(&self, curr_time: u64, num_bbox: usize) -> Vec<BBox3D> {
        let bboxes = self
            .objects
            .iter()
            .rev()
            .take(num_bbox)
            .map(|obj| self.predicted_bbox(curr_time, obj))
            .collect();
        bboxes
    }

    pub fn predicted_bbox(&self, curr_time: u64, ref_obj: &TrackingObject) -> BBox3D {
        let init_state = KalmanState {
            pos_x: ref_obj.bbox.center_x as f32,
            pos_y: ref_obj.bbox.center_y as f32,
            ..self.kalman_filter.state.clone()
        };
        let time_diff = (curr_time - ref_obj.timestamp_ns) as f32;
        let time_diff = time_diff / 1_000_000_000.0;
        let f = matrix![1., time_diff, 0.5 * time_diff.powf(2.), 0., 0., 0.;
                        0., 1., time_diff, 0., 0., 0.;
                        0., 0., 1., 0., 0., 0.;
                        0., 0., 0., 1., time_diff, 0.5 * time_diff.powf(2.);
                        0., 0., 0., 0., 1., time_diff;
                        0., 0., 0., 0., 0., 1.];
        let predicted_state = KalmanState::from_vec(&(f * init_state.to_vec()));
        let avg_size = self.avg_size;
        let avg_rotation = ref_obj.bbox.rotation.clone();
        BBox3D {
            center_x: predicted_state.pos_x as f64,
            center_y: predicted_state.pos_y as f64,
            center_z: ref_obj.bbox.center_z,
            size_x: avg_size.x,
            size_y: avg_size.y,
            size_z: avg_size.z,
            rotation: avg_rotation,
        }
    }
}

fn extend_objects(objects: &mut VecDeque<TrackingObject>, num_extended_objects: usize) {
    let first_obj = objects[0].clone();
    let extend_dir = objects
        .iter()
        .find_map(|ref_obj| {
            let displace_3d: Vector3<_> = first_obj.bbox.center() - ref_obj.bbox.center();
            let is_far_enough = displace_3d.norm() > 7.0;
            is_far_enough.then(|| {
                let displace_2d = Vector2::new(displace_3d.x, displace_3d.y);
                displace_2d.normalize()
            })
        })
        .unwrap_or_else(|| Vector2::from([0., 0.]));

    let extend_vectors = {
        let first = extend_dir.clone();
        iter::successors(Some(first), |prev| Some(prev + extend_dir)).take(num_extended_objects)
    };

    let new_objects = extend_vectors.map(|extend_vec| {
        let mut obj = first_obj.clone();
        obj.bbox.center_x = first_obj.bbox.center_x + extend_vec.x;
        obj.bbox.center_y = first_obj.bbox.center_y + extend_vec.y;
        obj
    });

    for obj in new_objects {
        objects.push_front(obj);
    }
}

fn guess_facing_direction_from_other_tracks(
    object: &TrackingObject,
    track_attr_map: &TrackAttrMap,
) -> Option<Vector2<f64>> {
    let mut track_attr_map = track_attr_map.clone();

    // Find tracks traversed through ego object
    let track_to_closest_index: Vec<(usize, usize)> = track_attr_map
        .iter_mut()
        .filter(|(_, track_attr)| track_attr.objects.len() >= 10)
        .filter_map(|(&track_id, track_attr)| {
            // Compute minimum distance to track
            extend_objects(&mut track_attr.objects, 10);

            let (min_index, min_distance) = track_attr
                .objects
                .iter()
                .enumerate()
                .map(|(index, obj)| {
                    let distance = (object.bbox.center() - obj.bbox.center()).norm();
                    (index, r64(distance))
                })
                .min_by_key(|&(_, distance)| distance)?;

            let is_close_enough = min_distance <= 1.0;
            is_close_enough.then(|| (track_id, min_index))
        })
        .collect();

    if track_to_closest_index.is_empty() {
        return None;
    }

    let (track_count, sum_displace) = track_to_closest_index
        .iter()
        .filter_map(|&(track_id, obj_idx)| {
            let track_attr = &track_attr_map[&track_id];
            let closest_object = &track_attr.objects[obj_idx];
            let ref_point = closest_object.bbox.center();
            let mut measured_objects = track_attr
                .objects
                .iter()
                .take(track_attr.objects.len())
                .skip(obj_idx);

            let displace_3d = measured_objects.find_map(|obj| {
                let target_point = obj.bbox.center();
                let displace_3d = target_point - ref_point;
                let is_far_enough = displace_3d.norm() > 7.0;
                is_far_enough.then(|| displace_3d)
            })?;
            let displace_2d = Vector2::new(displace_3d.x, displace_3d.y);

            Some(displace_2d)
        })
        .fold((0, Vector2::zeros()), |(count, sum), displace| {
            (count + 1, sum + displace)
        });

    if track_count == 0 {
        return None;
    }

    let avg_dir = (sum_displace / (track_count as f64)).normalize();
    Some(avg_dir)
}

fn predict_kalman_filter_from_other_tracks(
    object: &TrackingObject,
    track_attr_map: &TrackAttrMap,
) -> KalmanFilter {
    let mut track_attr_map = track_attr_map.clone();

    // Find tracks traversed through ego object
    let track_to_closest_index: Vec<(usize, usize)> = track_attr_map
        .iter_mut()
        .filter(|(_, track_attr)| track_attr.objects.len() >= 10)
        .filter_map(|(&track_id, track_attr)| {
            extend_objects(&mut track_attr.objects, 10);

            // Compute minimum distance to track
            let (min_index, min_distance) = track_attr
                .objects
                .iter()
                .enumerate()
                .map(|(index, obj)| {
                    let distance = (object.bbox.center() - obj.bbox.center()).norm();
                    (index, r64(distance))
                })
                .min_by_key(|&(_, distance)| distance)?;

            let is_close_enough = min_distance <= 1.0;
            is_close_enough.then(|| (track_id, min_index))
        })
        .collect();

    if track_to_closest_index.is_empty() {
        return KalmanFilter::with_state(KalmanState {
            pos_x: object.bbox.center_x as f32,
            vel_x: 0.0,
            acc_x: 0.0,
            pos_y: object.bbox.center_y as f32,
            vel_y: 0.0,
            acc_y: 0.0,
        });
    }

    let mut avg_state = KalmanState::default();
    for &(track_id, obj_idx) in &track_to_closest_index {
        let track_attr = &track_attr_map[&track_id];
        let measured_objects: Vec<&TrackingObject> = track_attr
            .objects
            .iter()
            .take(track_attr.objects.len())
            .skip(obj_idx)
            .collect();
        let last_object = measured_objects.last().unwrap();

        let init_kf = KalmanFilter::with_state(KalmanState {
            pos_x: last_object.bbox.center_x as f32,
            vel_x: 0.0,
            acc_x: 0.0,
            pos_y: last_object.bbox.center_y as f32,
            vel_y: 0.0,
            acc_y: 0.0,
        });
        let kf =
            measured_objects
                .iter()
                .rev()
                .tuple_windows()
                .fold(init_kf, |mut kf, (obj1, obj2)| {
                    let time_diff = obj1.timestamp_ns - obj2.timestamp_ns;
                    let measurement = matrix!(obj2.bbox.center_x as f32;obj2.bbox.center_y as f32);
                    kf.update(time_diff, measurement);
                    kf
                });

        avg_state = avg_state + kf.state;
    }
    avg_state = avg_state / (track_to_closest_index.len() as f32);
    avg_state = avg_state * -1.;
    avg_state.pos_x = object.bbox.center_x as f32;
    avg_state.pos_y = object.bbox.center_y as f32;
    KalmanFilter::with_state(avg_state)
}
