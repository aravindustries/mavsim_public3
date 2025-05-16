import numpy as np
import models.model_coef_solution as TF
import parameters.aerosonde_parameters_max as MAV

gravity = MAV.gravity
Va0 = TF.Va_trim
rho = 1.293
sigma = 0

#---------- ROLL LOOP -------------
a_phi1 = TF.a_phi1
a_phi2 = TF.a_phi2
delta_a_max = np.radians(35)  # reduce to prevent excessive roll rate
phi_max = np.radians(25)

#---------- ROLL LOOP -------------
zeta_roll = 0.75  # Reduce damping slightly for smoother response
wn_roll = np.sqrt(abs(a_phi2) * delta_a_max / phi_max) / 0.8  # Slightly lower bandwidth for less aggressive roll corrections
roll_kp = delta_a_max / phi_max * np.sign(a_phi2)
roll_kd = (2 * zeta_roll * wn_roll - a_phi1) / a_phi2

#---------- COURSE LOOP -------------
zeta_course = 1.0  # Keep the damping at 1 for reasonable stability
wn_course = wn_roll / 6  # Slow down the bandwidth for smoother course correction
course_kp = 2 * zeta_course * wn_course * (Va0 / gravity) * 1.0  # Decrease to reduce course overcorrection
course_ki = wn_course**2 * (Va0 / gravity) * 1.0  # Decrease for smoother steady-state correction

#---------- YAW DAMPER -------------
yaw_damper_p_wo = 4.0  # Increased for better yaw stabilization
yaw_damper_kr = 2.0  # Increased yaw damper gain for faster correction

#---------- PITCH LOOP -------------
a_theta1 = TF.a_theta1
a_theta2 = TF.a_theta2
a_theta3 = -TF.a_theta3

zeta_pitch = 0.55
wn_pitch = np.sqrt(abs(a_theta2) + abs(a_theta3)) / 2  # slower response
pitch_kp = (wn_pitch**2 - a_theta2) / a_theta3
pitch_kd = (2 * zeta_pitch * wn_pitch - a_theta1) / a_theta3
K_theta_DC = a_theta3 / (wn_pitch**2 + a_theta2)

#---------- ALTITUDE LOOP -------------
zeta_altitude = 0.9
wn_altitude = wn_pitch / 8  # slower outer loop
altitude_kp = 2 * zeta_altitude * wn_altitude / (K_theta_DC * Va0)
altitude_ki = wn_altitude**2 / (K_theta_DC * Va0)
altitude_zone = 5

#---------- AIRSPEED THROTTLE LOOP -------------
a_V1 = TF.a_V1
a_V2 = TF.a_V2

#---------- AIRSPEED THROTTLE LOOP -------------
zeta_airspeed_throttle = 1.0  # Keep a reasonable damping
wn_airspeed_throttle = 0.75  # Slower throttle response to smooth out the oscillations
airspeed_throttle_kp = (2 * zeta_airspeed_throttle * wn_airspeed_throttle - a_V1) / a_V2
airspeed_throttle_ki = wn_airspeed_throttle**2 / a_V2