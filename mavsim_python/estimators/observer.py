"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
        3/4/2024 - RWB
"""
import numpy as np
import parameters.control_parameters as CTRL
import parameters.sensor_parameters as SENSOR
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors
from estimators.filters import AlphaFilter, ExtendedKalmanFilterContinuousDiscrete

class Observer:
    def __init__(self, ts: float, initial_measurements: MsgSensors=MsgSensors()):
        self.Ts = ts  # sample rate of observer
        # initialized estimated state message
        self.estimated_state = MsgState()

        ##### TODO #####
        self.lpf_gyro_x = AlphaFilter(alpha=0.3, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=0.2, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=0.5, y0=initial_measurements.gyro_z)
        self.lpf_accel_x = AlphaFilter(alpha=0.5, y0=initial_measurements.accel_x)
        self.lpf_accel_y = AlphaFilter(alpha=0.3, y0=initial_measurements.accel_y)
        self.lpf_accel_z = AlphaFilter(alpha=0.4, y0=initial_measurements.accel_z)
        # use alpha filters to low pass filter absolute and differential pressure
        self.lpf_abs = AlphaFilter(alpha=0.7, y0=initial_measurements.abs_pressure)
        self.lpf_diff = AlphaFilter(alpha=0.3, y0=initial_measurements.diff_pressure)
        # ekf for phi and theta
        self.attitude_ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f_attitude, 
            Q=10*np.diag([
                (1e-3)**2, # phi 
                (1e-3)**2, # theta
                ]), 
            P0= np.diag([
                (0.*np.pi/180.)**2, # phi
                (0.*np.pi/180.)**2, # theta
                ]), 
            xhat0=np.array([
                [0.*np.pi/180.], # phi 
                [0.*np.pi/180.], # theta
                ]), 
            Qu=np.diag([
                SENSOR.gyro_sigma**2, 
                SENSOR.gyro_sigma**2, 
                SENSOR.gyro_sigma**2, 
                SENSOR.abs_pres_sigma]), 
            Ts=ts,
            N=5
            )
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f_smooth, 
            Q=1000*np.diag([
                (0.0003)**2,  # pn
                (0.0003)**2,  # pe
                (0.003)**2,  # Vg
                (np.radians(0.001))**2,  # chi
                (0.0003)**2,  # wn
                (0.003)**2,  # we
                (0.003*np.radians(2.0))**2,  # psi
            ]), 
 
            P0=np.diag([
                (0.002)**2,  # pn (10m initial uncertainty)
                (0.003)**2,  # pe (10m initial uncertainty)
                (0.004)**2,  # Vg (5 m/s initial uncertainty)
                (np.radians(0.001))**2,  # chi (10-degree uncertainty)
                (0.002)**2,  # wn (2 m/s initial wind uncertainty)
                (0.002)**2,  # we (2 m/s initial wind uncertainty)
                (np.radians(0.002))**2,  # psi (10-degree uncertainty)
            ]), 
 
            xhat0=np.array([
                [0.0], # pn 
                [0.0], # pe 
                [0.0], # Vg 
                [0.0], # chi
                [0.0], # wn 
                [0.0], # we 
                [0.0], # psi
                ]), 
            Qu=np.diag([
                SENSOR.gyro_sigma**2, 
                SENSOR.gyro_sigma**2, 
                SENSOR.abs_pres_sigma,
                np.radians(3), # guess for noise on roll
                np.radians(3), # guess for noise on pitch
                ]),
            Ts=ts,
            N=10
            )
        self.R_accel = np.diag([
                SENSOR.accel_sigma**2, 
                SENSOR.accel_sigma**2, 
                SENSOR.accel_sigma**2
                ])
        self.R_pseudo = 10 * np.diag([
                1.0,  # pseudo measurement #1 ##### TODO #####
                1.0,  # pseudo measurement #2 ##### TODO #####
                1.0,  # pseudo measurement #3 ##### TODO #####
                1.0,  # pseudo measurement #4 ##### TODO #####
                ])
        self.R_gps = np.diag([
                    SENSOR.gps_n_sigma**2,  # y_gps_n
                    SENSOR.gps_e_sigma**2,  # y_gps_e
                    SENSOR.gps_Vg_sigma**2,  # y_gps_Vg
                    SENSOR.gps_course_sigma**2,  # y_gps_course
                    ])
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999

    def update(self, measurement: MsgSensors) -> MsgState:
        ##### TODO #####
        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x) - SENSOR.gyro_x_bias
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y) - SENSOR.gyro_y_bias
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z) - SENSOR.gyro_z_bias

        # invert sensor model to get altitude and airspeed
        abs_pressure = self.lpf_abs.update(measurement.abs_pressure)
        diff_pressure = self.lpf_diff.update(measurement.diff_pressure)
        self.estimated_state.altitude = abs_pressure / (CTRL.rho*CTRL.gravity)
        self.estimated_state.Va = np.sqrt(2*diff_pressure/CTRL.rho)
        # estimate phi and theta with ekf
        u_attitude=np.array([
                [self.estimated_state.p],
                [self.estimated_state.q],
                [self.estimated_state.r],
                [self.estimated_state.Va],
                ])
        xhat_attitude, P_attitude=self.attitude_ekf.propagate_model(u_attitude)
        y_accel=np.array([
                [measurement.accel_x],
                [measurement.accel_y],
                [measurement.accel_z],
                ])
        xhat_attitude, P_attitude=self.attitude_ekf.measurement_update(
            y=y_accel, 
            u=u_attitude,
            h=self.h_accel,
            R=self.R_accel)
        self.estimated_state.phi = xhat_attitude.item(0)
        self.estimated_state.theta = xhat_attitude.item(1)
        # estimate pn, pe, Vg, chi, wn, we, psi with ekf
        # p is not included because it is th roll rate and soes not contribute to determining position
        # It does not directly affect position, velocity, or heading states—only the roll angle, 𝜙 ϕ does.
        u_smooth = np.array([
                [self.estimated_state.q],
                [self.estimated_state.r],
                [self.estimated_state.Va],
                [self.estimated_state.phi],
                [self.estimated_state.theta],
                ])
        xhat_position, P_position=self.position_ekf.propagate_model(u_smooth)
        y_pseudo = np.array([[0.], [0.], [0.], [0.]]) # pseudo measurement, I changed this to be a 4x1 array
        xhat_position, P_position=self.position_ekf.measurement_update(
            y=y_pseudo,
            u=u_smooth,
            h=self.h_pseudo,
            R=self.R_pseudo)
        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):
            y_gps = np.array([
                    [measurement.gps_n],
                    [measurement.gps_e],
                    [measurement.gps_Vg],
                    [wrap(measurement.gps_course, xhat_position.item(3))],
                    ])
            xhat_position, P_position=self.position_ekf.measurement_update(
                y=y_gps,
                u=u_smooth,
                h=self.h_gps,
                R=self.R_gps)
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course
        self.estimated_state.north = xhat_position.item(0)
        self.estimated_state.east = xhat_position.item(1)
        self.estimated_state.Vg = xhat_position.item(2)
        self.estimated_state.chi = xhat_position.item(3)
        self.estimated_state.wn = xhat_position.item(4)
        self.estimated_state.we = xhat_position.item(5)
        self.estimated_state.psi = xhat_position.item(6)
        # not estimating these
        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state

    def f_attitude(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
            system dynamics for propagation model: xdot = f(x, u)
                x = [phi, theta].T
                u = [p, q, r, Va].T
        '''
        phi, theta = x.flatten()
        p, q, r, Va = u.flatten()
        
        phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
    
        xdot = np.array([[phi_dot], [theta_dot]])
        return xdot

    def h_accel(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            measurement model y=h(x,u) for accelerometers
                x = [phi, theta].T
                u = [p, q, r, Va].T
        '''
        ##### TODO #####
        phi, theta = x.flatten()
        p, q, r, Va = u.flatten()

        # compute the acceleration due to gravity
        y_accel_x = q * Va * np.sin(theta) + CTRL.gravity * np.sin(theta)
        y_accel_y = r * Va * np.cos(theta) - p * Va * np.sin(theta) - CTRL.gravity * np.cos(theta) * np.sin(phi)
        y_accel_z = -q * Va * np.cos(theta) - CTRL.gravity * np.cos(theta) * np.cos(phi)
        y = np.array([[y_accel_x], [y_accel_y], [y_accel_z]])
        return y

    def f_smooth(self, x, u):
        '''
            system dynamics for propagation model: xdot = f(x, u)
                x = [pn, pe, Vg, chi, wn, we, psi].T
                u = [p, q, r, Va, phi, theta].T
        '''
        ##### TODO #####
        pn, pe, Vg, chi, wn, we, psi = x.flatten()
        q, r, Va, phi, theta = u.flatten()
        pn_dot = Vg * np.cos(chi)
        pe_dot = Vg * np.sin(chi)
        Vg_dot = q * Va * np.sin(theta) + CTRL.gravity * np.sin(theta) 
        chi_dot = CTRL.gravity / max(Vg, 1e-6) * np.tan(phi)
        psi_dot = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)
        Vg_dot = (Va * psi_dot * (we * np.cos(psi) - wn * np.sin(psi))) / max(Vg, 1e-6)
        wn_dot = 0.0
        we_dot = 0.0
        xdot = np.array([[pn_dot], [pe_dot], [Vg_dot], [chi_dot], [wn_dot], [we_dot], [psi_dot]])

        return xdot
    
    def h_pseudo(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pn, pe, Vg, chi, wn, we, psi = x.flatten()

        # Enforce wind constraints using airspeed and ground speed
        Vn = Vg * np.cos(chi)
        Ve = Vg * np.sin(chi)
        
        y = np.array([
            [pn], 
            [pe], 
            [Vn - wn],  # Ground speed in North should match wind + airspeed
            [Ve - we]   # Ground speed in East should match wind + airspeed
        ])
        return y

    def h_gps(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            measurement model for gps measurements: y=y(x, u)
                x = [pn, pe, Vg, chi, wn, we, psi].T
                u = [p, q, r, Va, phi, theta].T
            returns
                y = [pn, pe, Vg, chi]
        '''
        ##### TODO #####         
        pn, pe, Vg, chi, wn, we, psi = x.flatten()

        y = np.array([[pn], [pe], [Vg], [chi]])
        return y