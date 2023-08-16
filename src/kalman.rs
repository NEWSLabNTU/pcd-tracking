use nalgebra::{matrix, Matrix2, Matrix2x6, Matrix6, Vector2, Vector6};
use std::ops::{Add, Div, Mul};

#[derive(Clone, Debug)]
pub struct KalmanFilter {
    pub state: KalmanState,
    // /// State transition matrix
    /// Covariance matrix of estimation error
    pub p: Matrix6<f32>,
    /// Covariance matrix of measurement error
    pub r: Matrix2<f32>,
    /// Covariance matrix of process noise
    pub q: Matrix6<f32>,
    /// Observation Matrix
    pub h: Matrix2x6<f32>,
}

impl KalmanFilter {
    pub fn new() -> Self {
        Self::with_state(KalmanState::default())
    }

    pub fn with_state(state: KalmanState) -> Self {
        // Higher p -> Higher confidence on measurement
        let p = Matrix6::identity() * 100.;
        let q_weight = 0.1;
        let q = Matrix6::identity() * q_weight;
        // Higher r_weight -> Lower confidence on measurement
        let r_weight = 50.0;
        let r = Matrix2::identity() * r_weight;
        let h = matrix![1., 0., 0., 0., 0., 0.;
                        0., 0., 0., 1., 0., 0.];
        Self { state, p, r, q, h }
    }

    pub fn update(&mut self, delta_time: u64, measurement: Vector2<f32>) {
        let delta_time = delta_time as f32;
        let delta_time = delta_time / 1_000_000_000.0;

        let f = matrix![1., delta_time, 0.5 * delta_time.powf(2.), 0., 0., 0.;
                        0., 1., delta_time, 0., 0., 0.;
                        0., 0., 1., 0., 0., 0.;
                        0., 0., 0., 1., delta_time, 0.5 * delta_time.powf(2.);
                        0., 0., 0., 0., 1., delta_time;
                        0., 0., 0., 0., 0., 1.];
        // Predict state
        self.state = KalmanState::from_vec(&(f * self.state.to_vec()));
        self.p = f * self.p * f.transpose() + self.q;

        // Update state
        let k = self.p
            * self.h.transpose()
            * (self.h * self.p * self.h.transpose() + self.r)
                .try_inverse()
                .unwrap();

        let state_mat = &self.state.to_vec();
        let i = Matrix6::identity();
        self.state = KalmanState::from_vec(&(state_mat + k * (measurement - self.h * state_mat)));
        self.p =
            (i - k * self.h) * self.p * (i - k * self.h).transpose() + k * self.r * k.transpose();
    }

    // Return the predicted state based on current states and passed time
    pub fn predict(&self, delta_time: u64, ref_state: &KalmanState) -> KalmanState {
        let delta_time = delta_time as f32;
        let delta_time = delta_time / 1_000_000_000.0;
        let f = matrix![1., delta_time, 0.5 * delta_time.powf(2.), 0., 0., 0.;
                        0., 1., delta_time, 0., 0., 0.;
                        0., 0., 1., 0., 0., 0.;
                        0., 0., 0., 1., delta_time, 0.5 * delta_time.powf(2.);
                        0., 0., 0., 0., 1., delta_time;
                        0., 0., 0., 0., 0., 1.];
        KalmanState::from_vec(&(f * ref_state.to_vec()))
    }
}
#[derive(Debug, Clone)]
pub struct KalmanState {
    pub pos_x: f32,
    pub vel_x: f32,
    pub acc_x: f32,
    pub pos_y: f32,
    pub vel_y: f32,
    pub acc_y: f32,
}

impl KalmanState {
    pub fn to_vec(&self) -> Vector6<f32> {
        let Self {
            pos_x,
            vel_x,
            acc_x,
            pos_y,
            vel_y,
            acc_y,
        } = *self;
        matrix![pos_x; vel_x; acc_x; pos_y; vel_y; acc_y]
    }

    pub fn from_vec(from: &Vector6<f32>) -> Self {
        KalmanState {
            pos_x: from[(0, 0)],
            vel_x: from[(1, 0)],
            acc_x: from[(2, 0)],
            pos_y: from[(3, 0)],
            vel_y: from[(4, 0)],
            acc_y: from[(5, 0)],
        }
    }
}

impl Default for KalmanState {
    fn default() -> Self {
        Self {
            pos_x: 0.0,
            vel_x: 0.0,
            acc_x: 0.0,
            pos_y: 0.0,
            vel_y: 0.0,
            acc_y: 0.0,
        }
    }
}

impl Add for KalmanState {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            pos_x: self.pos_x + other.pos_x,
            vel_x: self.vel_x + other.vel_x,
            acc_x: self.acc_x + other.acc_x,
            pos_y: self.pos_y + other.pos_y,
            vel_y: self.vel_y + other.vel_y,
            acc_y: self.acc_y + other.acc_y,
        }
    }
}
impl Div<f32> for KalmanState {
    // The division of rational numbers is a closed operation.
    type Output = Self;

    fn div(self, divisor: f32) -> Self {
        Self {
            pos_x: self.pos_x / divisor,
            vel_x: self.vel_x / divisor,
            acc_x: self.acc_x / divisor,
            pos_y: self.pos_y / divisor,
            vel_y: self.vel_y / divisor,
            acc_y: self.acc_y / divisor,
        }
    }
}
impl Mul<f32> for KalmanState {
    // The division of rational numbers is a closed operation.
    type Output = Self;

    fn mul(self, multilier: f32) -> Self {
        Self {
            pos_x: self.pos_x * multilier,
            vel_x: self.vel_x * multilier,
            acc_x: self.acc_x * multilier,
            pos_y: self.pos_y * multilier,
            vel_y: self.vel_y * multilier,
            acc_y: self.acc_y * multilier,
        }
    }
}
