use crate::{
    object::{BBox3D, TrackingObject},
    track_attributes::TrackAttrMap,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A structure that defines the weak class with size of the bounding
/// box.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeakClass {
    /// The name of the class
    pub class: String,
    /// The minimum length (x-axis) of the box to be considered as this class.
    pub min_length: f64,
    /// The minimum width (y-axis) of the box to be considered as this class.
    pub min_width: f64,
    /// The minimum height (z-axis) of the box to be considered as this class.
    pub min_height: f64,
    /// The maximum length (x-axis) of the box to be considered as this class.
    pub max_length: f64,
    /// The maximum width (y-axis) of the box to be considered as this class.
    pub max_width: f64,
    /// The maximum height (z-axis) of the box to be considered as this class.
    pub max_height: f64,
}

#[derive(Clone, Debug)]
pub struct WeakClassifier {
    weak_classes: Vec<WeakClass>,
}

impl WeakClassifier {
    pub fn new(weak_classes: Vec<WeakClass>) -> Self {
        Self { weak_classes }
    }

    pub fn classify_one(&self, bbox: &BBox3D) -> Option<&WeakClass> {
        self.weak_classes.iter().find(|weak_class| {
            let bbox_length = bbox.size_x;
            let bbox_width = bbox.size_y;
            bbox_length > weak_class.min_length
                && bbox_width > weak_class.min_width
                && bbox.size_z > weak_class.min_height
                && bbox_length <= weak_class.max_length
                && bbox_width <= weak_class.max_width
                && bbox.size_z <= weak_class.max_height
        })
    }

    pub fn update_bbox_classes(&self, objects: &mut [TrackingObject]) {
        objects.iter_mut().for_each(|object| {
            let weak_class = self.classify_one(&object.bbox);
            if let Some(weak_class) = weak_class {
                object.weak_class = Some(weak_class.class.clone());
            }
        });
    }
}

pub fn update_class_from_historic_classes(
    mut objects: Vec<TrackingObject>,
    track_attr_map: &mut TrackAttrMap,
) -> Vec<TrackingObject> {
    let track_ids: Vec<usize> = track_attr_map.keys().cloned().collect();
    objects
        .iter_mut()
        .filter(|obj| track_ids.contains(&&obj.track_id.unwrap()))
        .for_each(|obj| {
            let track_id = obj.track_id.unwrap();

            // Find corresponding track attribute
            let track_attr = track_attr_map.get_mut(&track_id).unwrap();

            // Assign track class from the object class
            if let Some(class) = &obj.weak_class {
                track_attr.historic_class_names.push(class.clone());
            }

            // Choose the most common class in last 10 recent object classes.
            let max_class = find_common_class(&track_attr.historic_class_names);

            // Update class name if most common class is available
            if let Some(max_class) = max_class {
                obj.weak_class = Some(max_class.to_string());
            }
        });
    return objects;
}

fn find_common_class(historic_classes: &[String]) -> Option<&str> {
    // Get last 10 classes in history
    let last_10_classes = historic_classes.iter().rev().take(10).map(|s| s.as_str());

    // Initialize per-class counts
    let mut counts: HashMap<&str, usize> = HashMap::new();

    // Increase count for each recent class.
    // Return the class with the greatest count.
    last_10_classes.max_by_key(|class| {
        let count: &mut usize = counts.entry(class).or_insert(0);
        *count += 1;
        *count
    })
}
