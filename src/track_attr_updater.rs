use crate::config::Config;
use crate::{
    object::{BBox3D, TrackingObject},
    track_attributes::{TrackAttrMap, TrackAttributes},
};
use nalgebra as na;
use nalgebra::matrix;

pub struct TrackAttributeUpdater {
    average_weight_for_current_bbox_size: f64,
    moving_threshold_speed_mps: f64,
}

impl TrackAttributeUpdater {
    pub fn new(config: &Config) -> Self {
        let Config {
            moving_threshold_speed_km,
            ..
        } = *config;

        Self {
            average_weight_for_current_bbox_size: 0.1,
            moving_threshold_speed_mps: moving_threshold_speed_km * 1000.0 / 3600.0,
        }
    }

    /// Update track attribute map.
    pub fn update_track_attribute_map(
        &self,
        track_attr_map: &mut TrackAttrMap,
        objects: &Vec<TrackingObject>,
        expired_track_attr_map: &TrackAttrMap,
    ) {
        objects.iter().for_each(|obj| {
            let track_id = obj.track_id.unwrap();

            if let Some(attr) = track_attr_map.get_mut(&track_id) {
                self.update_track_attributes(attr, obj.bbox.clone(), obj.clone());
            } else {
                track_attr_map.insert(
                    track_id,
                    TrackAttributes::new(track_id, &obj, track_attr_map, expired_track_attr_map),
                );
            }
        });
    }

    fn update_track_attributes(
        &self,
        attrs: &mut TrackAttributes,
        new_bbox: BBox3D,
        new_object: TrackingObject,
    ) {
        let Self {
            average_weight_for_current_bbox_size,
            moving_threshold_speed_mps,
        } = *self;

        // Update Kalman filter
        let time_diff = new_object.timestamp_ns - attrs.objects.back().unwrap().timestamp_ns;
        let measurement = matrix!(new_bbox.center_x as f32;
                                  new_bbox.center_y as f32);
        attrs.kalman_filter.update(time_diff, measurement);
        attrs.kalman_states.push(attrs.kalman_filter.state.clone());

        // Update average box size
        attrs.avg_size = {
            let weight = average_weight_for_current_bbox_size;
            let new_size = na::Vector3::new(new_bbox.size_x, new_bbox.size_y, new_bbox.size_z);
            weight * new_size + (1.0 - weight) * attrs.avg_size
        };

        // Update historic contexts
        attrs.bbox_list.push(new_bbox);

        // Push an object to the track
        attrs.objects.push_back(new_object);

        // Update the facing direction if moving fast enough
        if attrs.current_speed() > moving_threshold_speed_mps {
            let move_dir = attrs.current_direction();
            attrs.facing_direction = Some(move_dir);
        }
    }
}
