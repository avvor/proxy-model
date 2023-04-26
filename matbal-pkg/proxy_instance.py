"""_summary_

Returns:
    _type_: _description_
"""

import numpy as np

from .mat_balance import MaterialBalance


class ProxyInstance(MaterialBalance):
    """_summary_

    Args:
        MaterialBalance (_type_): _description_
    """    
    def __init__(self, j_2d:np.ndarray,g_0:np.ndarray, p_initial:np.ndarray) -> None:
        self.g_0 = g_0
        self.p_initial = p_initial
        self.j_2d = j_2d 
    
    def calc_press(self, wgp:np.ndarray)->np.ndarray:
        """_summary_

        Args:
            wgp (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """

        res= self.get_press_all_dates(
            p_initial = self.p_initial,
            wgp_by_dates = wgp,
            J_ij = self.j_2d,
            rgip_initial = self.g_0
        )
        return np.array([r.x for r in res])

class ProxyInstanceSimple(MaterialBalance):
    """_summary_

    Args:
        MaterialBalance (_type_): _description_
    """
    def __init__(self, p_initial:np.ndarray, c00_p_from_wgpt:np.ndarray) -> None:
        self.p_initial_test = p_initial
        self.c00_p_from_wgpt = c00_p_from_wgpt
    def calc_press(self, wgp:np.ndarray)-> np.ndarray:
        """_summary_

        Args:
            wgp (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        pass
            

if __name__== "__main__":
    wgp_all = np.array(
      [[   0,    0,   0,   0,],
       [ 250,    0,   0,   0,],
       [ 250,  250,   0,   0,],
       [ 250,  250, 250,   0,],
       [ 250,  250, 250, 250,],
       [   0,  250, 250, 250,],
       [   0,    0, 250, 250,],
       [   0,    0,   0, 250,],
       [   0,    0,   0,   0,],
       [   0,    0,   0,   0,],
       [   0,    0,   0,   0,]])

    # J_ij = np.ones((4,4))*.0005
    J_ij = ProxyInstance.get_j2d(np.arange(6),4)*.001
    print(J_ij)
    p_initial_test = np.ones((4,))*600
    rgip0  = np.ones((4,))*wgp_all.sum()/4*1.01

    matbal = ProxyInstance(J_ij,rgip0,p_initial_test)
    rpr = matbal.calc_press(wgp_all)    
    flows = np.array([matbal.get_flows(p, J_ij) for p in rpr]).sum(1)    
    print(rgip0 -(flows + wgp_all.cumsum(0)))
    print(rpr)


