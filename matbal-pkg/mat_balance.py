import numpy as np
from scipy.optimize import root
from typing import List

class MaterialBalance:    
    c00 = [ 1.58136372e-08, -2.11511088e-05,  8.41243666e-03,  3.44397433e-03, 9.78500838e+01]
    f_pz = np.poly1d(c00)
    
    def __init__(self, wgp:np.ndarray, g_0:np.ndarray, p_initial:np.ndarray) -> None:
        self.wgp = wgp
        self.g_0 = g_0
        self.p_initial = self.p_z(p_initial)
    
    @staticmethod
    def z_value(p_mpa:float, t_kelv:float=None, t_krit:float=190, p_krit:float=4.6)->float:
        '''Формула Мандрыгина для Z

        Args:
            p_mpa (float): Давление, МПА
            t_kelv (float, optional): Температура, К. Defaults to None.
            t_krit (float, optional): Критич. Температура, К. Defaults to 190.
            p_krit (float, optional):  Критич. Давление, МПа. Defaults to 4.6.

        Returns:
            float: Значение сверхсжимаемости Z
        '''
        return 5.64E-06 * p_mpa**2 -0.002039437*p_mpa +  1.0062
        # возможно понадобится позже
        # t = 300 if t_kelv is None else t_kelv
        # return 1-0.427*p_mpa/p_krit*(t/t_krit)**(-3.688)
    
    @staticmethod 
    def p_z(press):
        return press
        # return MaterialBalance.f_pz(press)
    
    @staticmethod
    def eq_press_one_date(
            p:np.ndarray,
            p0:np.ndarray,
            wgp:np.ndarray,
            J:np.ndarray,
            G_0:np.ndarray,)->np.ndarray:    
        flow_sum = MaterialBalance.get_flows(p,J)
        f = p - p0*(1 - (wgp  + flow_sum)/G_0)           
        return f
        
    @staticmethod
    def get_flows(press:np.ndarray, J:np.ndarray):
        return (J * (press**2 - press.reshape(-1,1)**2)).sum(0)
        
    @staticmethod
    def get_press_all_dates(
            p_initial:np.ndarray,
            wgp_by_dates:np.ndarray,
            J_ij:np.ndarray,
            rgip_initial:np.ndarray,
            method)->List[np.ndarray]:
        res = []
        guess = p_initial
        p_prev = p_initial
        for wgpt_curr, wgp_curr in list(zip(wgp_by_dates.cumsum(0), wgp_by_dates))[::]:    
            sol = root(
                fun=MaterialBalance.eq_press_one_date, 
                x0=guess,
                args =  (
                        p_prev,
                        wgp_curr,
                        J_ij,
                        rgip_initial-wgpt_curr+wgp_curr,
                    ),
                method=method)
            # assert all(map(lambda x: abs(x) < 1, sol.fun)), f'{sol.fun}'
            res.append(sol)
            p_prev = sol.x
            guess = sol.x
        return res
        


    def calc(self, J_ij:np.ndarray, method="hybr"):
        res = self.get_press_all_dates(
            p_initial=self.p_initial,
            wgp_by_dates=self.wgp,
            J_ij=J_ij,
            rgip_initial=self.g_0,
            method=method
        )
        return res        