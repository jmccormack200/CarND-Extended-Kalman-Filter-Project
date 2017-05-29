#include "kalman_filter.h"
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
  long x_size = x_.size();
  I_ = MatrixXd::Identity(x_size, x_size);
}

void KalmanFilter::Predict() {
  // Equation (11)
  x_ = F_ * x_;

  //Equation (12)
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  MatrixXd H_trans = H_.transpose();

  // Equation (13)
  VectorXd y = z - H_ * x_;

  // Equation(14)
  MatrixXd S = H_ * P_ * H_trans + R_;

  // Equation (15)
  MatrixXd K = P_ * H_trans * S.inverse();

  // Equation (16)
  x_ = x_ + (K * y);

  // Equation (17)
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Equation (18)
  float p_x = x_[0];
  float p_y = x_[1];
  float v_x = x_[2];
  float v_y = x_[3];

  // Calculate Ro (5.14, step #1)
  float p_x2 = p_x * p_x;
  float p_y2 = p_y * p_y;

  float ro = sqrt(p_x2 + p_y2);
  // Calculate phi (5.14 step #2)
  float phi = atan2(p_y, p_x);

  // calculate Ro dot, watch out for divie by zero (5.14 step #3)
  float ro_dot;
  if (ro < 0.00001) {
    ro_dot = 0;
  } else {
    ro_dot = (p_x * v_x + p_y * v_y) / ro;
  }

  VectorXd h = VectorXd(3);
  h(0) = ro;
  h(1) = phi;
  h(2) = ro_dot;

  // for Radar: y=z−h(x').
  VectorXd y = z - h;

  // Normailizing Angles: In C++, atan2() returns values between -pi and pi.
  // When calculating phi in y = z - h(x) for radar measurements, the resulting
  // angle phi in the y vector should be adjusted so that it is between
  // -pi and pi. when working in radians, you can add 2π or subtract 2π
  // until the angle is within the desired range.
  while( y(1) > M_PI) {
    y(1) -= 2 * M_PI;
  }

  // Now calculate S, K, x', and P
  // Use Hj instead of H
  Tools tools;

  MatrixXd Hj = tools.CalculateJacobian(x_);
  MatrixXd Hj_trans = Hj.transpose();

  MatrixXd S = Hj * P_ * Hj_trans + R_;
  MatrixXd K = P_ * Hj_trans * S.inverse();

  x_ = x_ + (K * y);

  P_ = (I_ - K * Hj) * P_;
}
