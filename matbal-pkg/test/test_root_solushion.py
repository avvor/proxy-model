from solver import SolverMatbal
from proxy_instance import ProxyInstance
import numpy as np
from typing import List, Tuple
def test_matbal(border_j1d_like_ones_6wells:List[Tuple[float,float]],
        RPR_6wells:np.ndarray,
        RGIP_6wells:np.ndarray,
        WGP_6wells:np.ndarray):
    j_2d = ProxyInstance.get_j2d(np.array(border_j1d_like_ones_6wells),6)
    res = ProxyInstance(j_2d, RGIP_6wells[0],RPR_6wells[0])
    print(res.calc_press(WGP_6wells))


def test_optimize(border_j1d_like_ones_6wells:List[Tuple[float,float]],
        RPR_6wells:np.ndarray,
        RGIP_6wells:np.ndarray,
        WGP_6wells:np.ndarray):
    j_2d = ProxyInstance.get_j2d(np.array(border_j1d_like_ones_6wells),6)
    solver = SolverMatbal(wgp=WGP_6wells, rpr=RPR_6wells, rgip=RGIP_6wells)
    res = solver.optimize(border_j1d_like_ones_6wells)
