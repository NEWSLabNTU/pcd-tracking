use geo::{prelude::*, LineString, Polygon};
use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector2};

#[derive(Clone, Debug)]
pub struct BBox3D {
    pub center_x: f64,
    pub center_y: f64,
    pub center_z: f64,
    pub size_x: f64,
    pub size_y: f64,
    pub size_z: f64,
    pub rotation: UnitQuaternion<f64>,
}

#[derive(Clone, Debug)]
pub struct OutputObject {
    pub bbox: BBox3D,
    pub timestamp_ns: u64,
    pub track_id: usize,
    pub weak_class: Option<String>,
}

#[derive(Clone, Debug)]
pub struct TrackingObject {
    pub bbox: BBox3D,
    pub timestamp_ns: u64,
    pub weak_class: Option<String>,
    pub track_id: Option<usize>,
    pub best_match_track_id: Option<usize>,
}

impl TrackingObject {
    pub fn change_bbox_heading_to_long_axis(&mut self) {
        let bbox = &mut self.bbox;
        if bbox.size_x < bbox.size_y {
            let dir_left = bbox.direction_left();
            let yaw = dir_left.y.atan2(dir_left.x);
            let na_rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, yaw);
            bbox.rotation = na_rotation;
            let tmp = bbox.size_x;
            bbox.size_x = bbox.size_y;
            bbox.size_y = tmp;
        }
    }
}

impl BBox3D {
    pub fn center(&self) -> Point3<f64> {
        let Self {
            center_x,
            center_y,
            center_z,
            ..
        } = *self;
        Point3::new(center_x, center_y, center_z)
    }

    pub fn vertex(&self, x_choice: bool, y_choice: bool, z_choice: bool) -> Point3<f64> {
        let point = {
            let x = self.size_x / 2.0 * if x_choice { 1.0 } else { -1.0 };
            let y = self.size_y / 2.0 * if y_choice { 1.0 } else { -1.0 };
            let z = self.size_z / 2.0 * if z_choice { 1.0 } else { -1.0 };
            Point3::new(x, y, z)
        };
        self.pose() * point
    }

    pub fn vertices(&self) -> Vec<Point3<f64>> {
        (0b000..=0b111)
            .map(|mask| self.vertex(mask & 0b001 != 0, mask & 0b010 != 0, mask & 0b100 != 0))
            .collect()
    }

    pub fn pose(&self) -> Isometry3<f64> {
        let rotation = self.rotation;
        let translation = Translation3::new(self.center_x, self.center_y, self.center_z);
        Isometry3::from_parts(translation, rotation)
    }

    pub fn direction_front(&self) -> Vector2<f64> {
        let to = (self.pose() * Point3::new(self.size_x / 2.0, 0.0, 0.0)).xy();
        let dir = Vector2::new(to.x - self.center_x, to.y - self.center_y);
        dir
    }

    pub fn direction_back(&self) -> Vector2<f64> {
        let to = (self.pose() * Point3::new(-self.size_x / 2.0, 0.0, 0.0)).xy();
        let dir = Vector2::new(to.x - self.center_x, to.y - self.center_y);
        dir
    }

    pub fn direction_left(&self) -> Vector2<f64> {
        let to = (self.pose() * Point3::new(0.0, self.size_y / 2.0, 0.0)).xy();
        let dir = Vector2::new(to.x - self.center_x, to.y - self.center_y);
        dir
    }

    pub fn direction_right(&self) -> Vector2<f64> {
        let to = (self.pose() * Point3::new(0.0, -self.size_y / 2.0, 0.0)).xy();
        let dir = Vector2::new(to.x - self.center_x, to.y - self.center_y);
        dir
    }

    pub fn transform_to_2d_polygon_big(&self) -> Polygon<f64> {
        let new_bbox = BBox3D {
            size_x: self.size_x * 1.5,
            size_y: self.size_y * 1.5,
            rotation: self.rotation.clone(),
            ..*self
        };
        let ordered_points: Vec<(f64, f64)> = vec![
            new_bbox.vertex(false, false, false).xy(),
            new_bbox.vertex(true, false, false).xy(),
            new_bbox.vertex(true, true, false).xy(),
            new_bbox.vertex(false, true, false).xy(),
            new_bbox.vertex(false, false, false).xy(),
        ]
        .iter()
        .map(|point_na| (point_na[0], point_na[1]))
        .collect();
        Polygon::new(LineString::from(ordered_points), vec![])
    }

    pub fn transform_to_2d_polygon(&self) -> Polygon<f64> {
        let ordered_points: Vec<(f64, f64)> = [
            self.vertex(false, false, false).xy(),
            self.vertex(true, false, false).xy(),
            self.vertex(true, true, false).xy(),
            self.vertex(false, true, false).xy(),
            self.vertex(false, false, false).xy(),
        ]
        .iter()
        .map(|point_na| (point_na[0], point_na[1]))
        .collect();
        Polygon::new(LineString::from(ordered_points), vec![])
    }

    pub fn intersection(&self, other: &BBox3D, use_larger_bbox: bool) -> Option<f64> {
        use std::panic;

        let (poly_1, poly2) = if use_larger_bbox {
            (
                self.transform_to_2d_polygon_big(),
                other.transform_to_2d_polygon_big(),
            )
        } else {
            (
                self.transform_to_2d_polygon(),
                other.transform_to_2d_polygon(),
            )
        };
        let result = panic::catch_unwind(|| poly_1.intersection(&poly2));
        if result.is_err() {
            return None;
        }
        return Some(result.unwrap().unsigned_area());
    }

    pub fn area(&self) -> f64 {
        self.size_x * self.size_y
    }

    pub fn iou_with(&self, other: &Self, use_larger_bbox: bool) -> Option<f64> {
        let intersec = self.intersection(&other, use_larger_bbox)?;
        let union = self.area() + other.area() - intersec;
        Some(intersec / union)
    }
}
