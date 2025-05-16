from tools.transfer_function import TransferFunction
import numpy as np


class WindSimulation:
    def __init__(self, Ts, gust_flag = True, steady_state = np.array([[0., 0., 0.]]).T):
        # steady state wind defined in the inertial frame
        self._steady_state = steady_state
        ##### TODO #####

        #   Dryden gust model parameters (pg 56 UAV book)

        altitude = 50
        Lu = 200
        Lv = 200
        Lw = 50
        sigma_u = 1.06
        sigma_v = 1.06
        sigma_w = 0.7
        Va = 25
        
        Hu_num = sigma_u*np.sqrt(2*Va/(Lu*np.pi))
        Hu_a0 = 1
        Hu_a1 = Va/Lu

        Hv_co = sigma_v*np.sqrt(3*Va/(Lv*np.pi))
        Hv_b0 = Hv_co
        Hv_b1 = Hv_co*(Va/(np.sqrt(3)*Lv))
        Hv_a0 = 1
        Hv_a1 = 2*Va/Lv
        Hv_a2 = (Va/Lv)**2

        Hw_co = sigma_w*np.sqrt(3*Va/(Lw*np.pi))
        Hw_b0 = Hw_co
        Hw_b1 = Hw_co*(Va/(np.sqrt(3)*Lw))
        Hw_a0 = 1
        Hw_a1 = 2*Va/Lw
        Hw_a2 = (Va/Lw)**2



    

        # Dryden transfer functions (section 4.4 UAV book) - Fill in proper num and den
        self.u_w = TransferFunction(num=np.array([[Hu_num]]), den=np.array([[Hu_a0, Hu_a1]]),Ts=Ts)
        self.v_w = TransferFunction(num=np.array([[Hv_b0,Hv_b1]]), den=np.array([[Hv_a0,Hv_a1,Hv_a2]]),Ts=Ts)
        self.w_w = TransferFunction(num=np.array([[Hw_b0,Hw_b1]]), den=np.array([[Hw_a0,Hw_a1,Hw_a2]]),Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        gust *= 0 
        return np.concatenate(( self._steady_state, gust ))

