use crate::object::{BBox3D, TrackingObject};
use serde::{Deserialize, Serialize};

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

/// This struct classifies a [BBoxPvrcnn] bounding box with the
/// provided sizes in the in
/// [WeakClassConfig](crate::box_enhance::WeakClassConfig).
///
/// # Input
///
/// * `bbox`: The bounding box to be classified
/// * `config`: The config file that contains the sizes to determine the class.
///
/// # Return
///
/// An option of [WeakClass]. If the size does not match any of the
/// provided sizes in the config file, `None` will be returned.
///
/// # Example usage
///
/// ```rust
/// let bbox = protos::BBoxPvrcnn{
///     x: 0,
///     y: 0,
///     z: 0,
///     dx: 4,
///     dy: 2,
///     dz: 2,
///     heading: 0,
///     ...// omitted
/// };
/// let config: config::Config = Json5Path::open_and_take("./example_config.json5")?;
/// let classifier = WeakClassifier::new(config.weak_class_conf_vec.as_ref().unwrap());
/// let weak_class = classifier.classify_one(&bbox);
/// let correct_weak_class = WeakClass{
///     class: "Car",
///     ... // omitted
/// };
/// asssert!(weak_class == Some(correct_weak_class))
/// ```
///
/// With the `./example_config.json5` as:
///
/// ```json5
/// "weak_class_conf_vec": [{
///     "class": "Scooter",
///     "class_id": 0,
///     "min_length": 1.0,
///     "min_width": 0.5,
///     "min_height": 0.0,
///     "max_length": 4.0,
///     "max_width": 1.0,
///     "max_height": 2.0,
///     },
///     {
///     "class": "Car",
///     "class_id": 1,
///     "min_length": 2.0,
///     "min_width": 0.5,
///     "min_height": 0.0,
///     "max_length": 5.5,
///     "max_width": 3.0,
///     "max_height": 2.0,
///     },
///     {
///     "class": "Truck",
///     "class_id": 2,
///     "min_length": 2.0,
///     "min_width": 1.0,
///     "min_height": 1.3,
///     "max_length": 10.0,
///     "max_width": 3.5,
///     "max_height": 3.0,
///     },
///     {
///     "class": "Other",
///     "class_id": 3,
///     "min_length": 2.0,
///     "min_width": 1.0,
///     "min_height": 1.3,
///     "max_length": 30.0,
///     "max_width": 30.0,
///     "max_height": 30.0,
/// }],
/// ```
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
