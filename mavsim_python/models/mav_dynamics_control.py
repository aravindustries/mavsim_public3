import numpy as np
from models.mav_dynamics import MavDynamics as MavDynamicsForces
# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
import parameters.aerosonde_parameters_max as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler


class MavDynamics(MavDynamicsForces):
    def __init__(self, Ts):
        super().__init__(Ts)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())
        # update the message class for the true state
        self._update_true_state()


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)
        super()._rk4_step(forces_moments)
        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3]
        gust = wind[3:6]

        ##### TODO #####
        # convert steady-state wind vector from world to body frame
        R_wb = quaternion_to_rotation(self._state[6:10])
        wind_body = R_wb.T @ steady_state
        # add the gust 
        wind_body += gust
        # convert total wind to world frame
        self._wind = R_wb @ wind_body

        # velocity vector relative to the airmass ([ur , vr, wr]= ?)
        vel_body = np.array([[self._state.item(3), self._state.item(4), self._state.item(5)]]).T
        vel_rel_airmass = vel_body - self._wind

        # compute airspeed (self._Va = ?)
        self._Va = np.linalg.norm(vel_rel_airmass)

        # compute angle of attack (self._alpha = ?)
        ur, vr, wr = vel_rel_airmass
        ur = ur[0]
        vr = vr[0]
        wr = wr[0]

        self._alpha = np.arctan2(wr, ur)
        
        # compute sideslip angle (self._beta = ?)
        self._beta = np.arcsin(vr / self._Va)


    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        ##### TODO ######
        # extract states (phi, theta, psi, p, q, r)
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        p, q, r = self._state[10:13]
        

        # compute gravitational forces ([fg_x, fg_y, fg_z])
        g = MAV.gravity
        weight = MAV.mass * g
        R_wb = quaternion_to_rotation(self._state[6:10]).T
        fg_ned = np.array([[0], [0], [weight]])
        fg_body = R_wb @ fg_ned
        # fg_x, fg_y, fg_z = fg_body.T

        # compute Lift and Drag coefficients (CL, CD)
        rho = MAV.rho
        S = MAV.S_wing
        Va = self._Va
        alpha = self._alpha
        beta = self._beta
        delta_e = delta.elevator
        delta_a = delta.aileron
        delta_r = delta.rudder
        delta_t = delta.throttle
        sigma_alpha = (1 + np.exp(-MAV.M*(alpha - MAV.alpha0)) + np.exp(MAV.M*(alpha + MAV.alpha0))) / ((1 + np.exp(-MAV.M*(alpha - MAV.alpha0))) * (1 + np.exp(MAV.M*(alpha + MAV.alpha0))))
        CL_1 = (1 - sigma_alpha) * (MAV.C_L_0 + MAV.C_L_alpha*alpha) + sigma_alpha*(2*np.sign(alpha)*np.sin(alpha)**2*np.cos(alpha))
        # CL = MAV.C_L_0 + MAV.C_L_alpha * alpha + MAV.C_L_q*(MAV.c*q/(2*Va)) + MAV.C_L_delta_e * delta_e
        CD = MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha*alpha)**2 / (np.pi * MAV.e * MAV.AR) + MAV.C_D_q*(MAV.c*q/(2*Va)) + MAV.C_D_delta_e*delta_e
        # CD = MAV.C_D_0 + MAV.C_D_alpha * alpha + MAV.C_D_q*(MAV.c*q/(2*Va)) + MAV.C_D_delta_e * delta_e
        
        CL = CL_1 + MAV.C_L_q*(MAV.c*q/(2*Va)) + MAV.C_L_delta_e*delta_e
        # compute Lift and Drag Forces (F_lift, F_drag)
        F_lift = 0.5 * rho * Va**2 * S * CL
        F_drag = 0.5 * rho * Va**2 * S * CD

        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta_t)

        # compute longitudinal forces in body frame (fx, fz)
        # f_body = np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]]) @ np.array([-F_drag, -F_lift])
        rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                            [np.sin(alpha), np.cos(alpha)]])
        F_drag = float(F_drag)
        F_lift = float(F_lift)
        force_vector = np.array([[-F_drag], [-F_lift]])  # Make it a column vector

        f_body = rotation_matrix @ force_vector
        fx, fz = f_body.flatten()  # Extract as scalars

        # ASK BRADY ABOUT THIS---------------------------------????
        fz = fz + fg_body.item(2)
        fx = fx + fg_body.item(0) + thrust_prop
        
        # compute lateral forces in body frame (fy)
        CY = MAV.C_Y_0 + MAV.C_Y_beta*beta + MAV.C_Y_p*(MAV.b*p/(2*Va)) + MAV.C_Y_r*(MAV.b*r/(2*Va)) + MAV.C_Y_delta_a*delta_a + MAV.C_Y_delta_r*delta_r
        fy = 0.5 * rho * Va**2 * S * CY
        fy = fy + fg_body.item(1)
        fy = float(fy)

        # compute logitudinal torque in body frame (My)
        My = 0.5 * rho * Va**2 * S * MAV.c * (MAV.C_m_0 + MAV.C_m_alpha*alpha + MAV.C_m_q*(MAV.c*q/(2*Va)) + MAV.C_m_delta_e*delta_e)
        My = My
        # compute lateral torques in body frame (Mx, Mz)
        Mx = 0.5 * rho * Va**2 * S * MAV.b * (MAV.C_ell_0 + MAV.C_ell_beta*beta + MAV.C_ell_p*(MAV.b*p/(2*Va)) + MAV.C_ell_r*(MAV.b*r/(2*Va)) + MAV.C_ell_delta_a*delta_a + MAV.C_ell_delta_r*delta_r)
        Mx = Mx - torque_prop 
        Mz = 0.5 * rho * Va**2 * S * MAV.b * (MAV.C_n_0 + MAV.C_n_beta*beta + MAV.C_n_p*(MAV.b*p/(2*Va)) + MAV.C_n_r*(MAV.b*r/(2*Va)) + MAV.C_n_delta_a*delta_a + MAV.C_n_delta_r*delta_r)
        Mz = Mz
        fx = fx.item() if isinstance(fx, np.ndarray) else float(fx)
        fy = fy.item() if isinstance(fy, np.ndarray) else float(fy)
        fz = fz.item() if isinstance(fz, np.ndarray) else float(fz)
        Mx = Mx.item() if isinstance(Mx, np.ndarray) else float(Mx)
        My = My.item() if isinstance(My, np.ndarray) else float(My)
        Mz = Mz.item() if isinstance(Mz, np.ndarray) else float(Mz)

        forces_moments = np.array([fx, fy, fz, Mx, My, Mz])
        


        return forces_moments

    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller
        ##### TODO #####
        # Propeller parameters
        D_prop = MAV.D_prop  # Propeller diameter (m)
        C_Q0 = MAV.C_Q0  
        C_Q1 = MAV.C_Q1  
        C_Q2 = MAV.C_Q2  
        C_T0 = MAV.C_T0  
        C_T1 = MAV.C_T1  
        C_T2 = MAV.C_T2  

        # Motor parameters
        K_V = MAV.KV  # Motor speed constant (RPM/Volt)
        K_Q = MAV.KQ  # Motor torque constant
        R = MAV.R_motor  # Motor resistance (Ohm)
        i_0 = MAV.i0  # No-load current (A)
        
        # Compute input voltage
        V_in = delta_t * MAV.V_max  # Assuming full voltage at max throttle

        # Compute quadratic coefficients
        a = (MAV.rho * D_prop**5 / (4 * np.pi**2)) * C_Q0
        b = (MAV.rho * D_prop**4 / (2 * np.pi)) * C_Q1 * Va + (K_Q * K_V) / R
        c = (MAV.rho * D_prop**3) * C_Q2 * Va**2 - (K_Q / R) * V_in + K_Q * i_0

        # Compute angular velocity
        # omega = np.roots([a, b, c])
        omega_p = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

        n = omega_p / (2 * np.pi)  # Convert to RPM
        J = Va / (n * D_prop)

        CT = C_T2 * J**2 + C_T1 * J + C_T0
        CQ = C_Q2 * J**2 + C_Q1 * J + C_Q0

        T_p = MAV.rho * n**2 * D_prop**4 * CT
        Q_p = MAV.rho * n**2 * D_prop**5 * CQ

        # Compute Thrust
        thrust_prop = ((MAV.rho * D_prop**4 * C_T0) / (4 * np.pi**2)) * omega_p**2 + \
                ((MAV.rho * D_prop**3 * C_T1 * Va) / (2 * np.pi)) *omega_p + \
                (MAV.rho * D_prop**2 * C_T2 * Va**2)

        # Compute Torque
        torque_prop = ((MAV.rho * D_prop**5 * C_Q0) / (4 * np.pi**2)) * omega_p**2 + \
                ((MAV.rho * D_prop**4 * C_Q1 * Va) / (2 * np.pi)) * omega_p + \
                (MAV.rho * D_prop**3 * C_Q2 * Va**2)

        return T_p, Q_p

    def _update_true_state(self):
        # rewrite this function because we now have more information
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0
