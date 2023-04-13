import numpy as np
from scipy.optimize import root
from typing import List

class MaterialBalance:    
    @classmethod
    def get_j2d(cls, j_1d:np.ndarray, n:int):
        indecies = np.triu_indices_from(np.ones((n,n)),1)
        b = np.zeros((n,n))
        b[indecies] = j_1d
        return b + b.T
    @classmethod
    def eq_press_one_date(cls,
            p:np.ndarray,
            p0:np.ndarray,
            wgp:np.ndarray,
            J:np.ndarray,
            G_0:np.ndarray,)->np.ndarray:    
            flow_sum = cls.get_flows(p,J).sum(0)
            f = p - p0*(1 - (wgp  + flow_sum)/G_0)           
            return f
    @classmethod
    def get_flows(cls,press:np.ndarray, J:np.ndarray):
            return (J * (press**2 - press.reshape(-1,1)**2))
    @classmethod
    def get_press_all_dates(cls,
        p_initial:np.ndarray,
        wgp_by_dates:np.ndarray,
        J_ij:np.ndarray,
        rgip_initial:np.ndarray,
        method='hybr')->List[np.ndarray]:


        res = []
        guess = p_initial
        p_prev = p_initial
        for wgpt_curr, wgp_curr in list(zip(wgp_by_dates.cumsum(0), wgp_by_dates))[::]:    
            sol = root(
                fun=cls.eq_press_one_date, 
                x0=guess,
                args =  (
                    p_prev,
                    wgp_curr,
                    J_ij,
                    rgip_initial-wgpt_curr+wgp_curr,
                ),
                    method=method
                )
            # assert all(map(lambda x: abs(x) < 1, sol.fun)), f'{sol.fun}'
            res.append(sol)
            p_prev = sol.x
            guess = sol.x
        return res

if __name__ == '__main__':
    print(MaterialBalance.get_j2d(np.arange(6),4))