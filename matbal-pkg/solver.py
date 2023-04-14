from mat_balance import MaterialBalance
from proxy_instance import ProxyInstance
from scipy.optimize import minimize
from geneticalgorithm import geneticalgorithm as ga
import numpy as np
import pandas as pd
from typing import Tuple, Any
class SolverMatbal:

    def __init__(self,
        wgp:np.ndarray,
        rpr:np.ndarray,
        rgip:np.ndarray) -> None:

        self.wgp = wgp
        self.rpr = rpr
        self.rgip = rgip
        self.n = rgip.shape[1]
    
    
    def optimize(self,border_j1d_like_ones:np.ndarray,lower=None, upper=None, start=None, stop=None)->Tuple[ProxyInstance,Any]:

        lower_value = 0 if lower is None else lower
        upper_value = 10 if upper is None else upper

        first_step = 0 if start is None else start
        last_step = -1 if start is None else stop

        p_initial = self.rpr[first_step]
        rgpip_initial = self.rgip[first_step]
        press_model = self.rpr[first_step:last_step,:]
        wgp =self.wgp[first_step:last_step,:]
        def fun(j_1d):
            j_2d = ProxyInstance.get_j2d(j_1d, self.n)
            res = ProxyInstance.get_press_all_dates(
                p_initial=p_initial,
                wgp_by_dates= wgp,
                J_ij=j_2d,
                rgip_initial=rgpip_initial)
            press_calc = np.array([r.x for r in res])            
            return np.abs(press_calc - press_model).mean()
        bnds = [
                (0,0) if marker == 0 else (lower_value,upper_value)
            for marker in border_j1d_like_ones]
        x0 = np.random.rand(border_j1d_like_ones.shape[0])
        x0[np.where(border_j1d_like_ones==0)] = 0        
        sol = minimize(
            fun=fun,
            x0=x0,
            method='Nelder-Mead',
            tol= 1e-3,
            options={'maxiter': 1e+8, 'disp': False},
            bounds=bnds)
        res_mat_bal = ProxyInstance(
            j_2d=ProxyInstance.get_j2d(sol.x, self.n),
            g_0=rgpip_initial,
            p_initial=p_initial)
        return res_mat_bal, sol

    def optimize_ga(self,border_j1d_like_ones:np.ndarray,lower=None, upper=None, start=None, stop=None)->Tuple[ProxyInstance,Any]:
        lower_value = 0 if lower is None else lower
        upper_value = 10 if upper is None else upper

        first_step = 0 if start is None else start
        last_step = -1 if start is None else stop

        p_initial = self.rpr[first_step]
        rgpip_initial = self.rgip[first_step]
        press_model = self.rpr[first_step:last_step,:]
        wgp =self.wgp[first_step:last_step,:]
        def fun(j_1d):
            j_2d = ProxyInstance.get_j2d(j_1d, self.n)
            res = ProxyInstance.get_press_all_dates(
                p_initial=p_initial,
                wgp_by_dates= wgp,
                J_ij=j_2d,
                rgip_initial=rgpip_initial)
            press_calc = np.array([r.x for r in res])            
            return np.abs(press_calc - press_model).mean()
        bnds = np.array([
                [0,0] if marker == 0 else [lower_value,upper_value]
            for marker in border_j1d_like_ones])
        x0 = np.random.rand(border_j1d_like_ones.shape[0])
        x0[np.where(border_j1d_like_ones==0)] = 0        
        algorithm_param = {'max_num_iteration': 2000,\
                   'population_size':500,\
                   'mutation_probability':.01,
                   'elit_ratio': 0.001, #Было ,1
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.002, #было ,3
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':200}
        model=ga(function=fun,dimension=x0.shape[0],variable_type='real',variable_boundaries=bnds,
                algorithm_parameters=algorithm_param)
        model.run()
        return model





if __name__=="__main__":
    rpr = np.array(
    [[600.,         600.,         600.,         600.        ],
     [531.05413987, 576.63881791, 572.94708307, 570.84510766],
     [454.69732107, 489.91255557, 506.21097239, 502.35642655],
     [339.69846911, 378.89648146, 385.12311733, 399.29920824],
     [156.46019498, 231.13065874, 242.222144,   249.51958359],
     [137.18210648,  80.42342431, 123.67250723, 141.297137  ],
     [ 68.29802457,  55.36310549,  19.86119529,  65.49058269],
     [ 39.51831552,  29.41599039,  20.6854553,    4.65626652],
     [ 30.42259237,  22.70053657,  19.34849947,   7.09191312],
     [ 25.86781089,  19.63361272,  18.07747266,   9.577713  ],
     [ 23.18557944,  18.01675226,  17.23598003,  11.80540114]
     ])
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
    rgip = np.array(
        [[1010.,         1010.,         1010.,         1010.        ],
        [ 893.94113545,  970.6753435,   964.46092313,  960.92259792,],
        [ 650.72454578,  858.09637815,  892.35654945,  888.82252661,],
        [ 381.01438115,  587.78107168,  768.40362991,  802.80091726,],
        [ 119.75223439,  311.10512164,  477.99994641,  631.14269756,],
        [   8.76785988,   90.46870059,  260.39311509,  430.37032445,],
        [   4.97863944,    6.88395277,   41.75471891,  236.38268887,],
        [   5.78615791,    5.31328402,   10.41501027,   18.48554779,],
        [   7.69835252,    7.7170737,     9.35367348,   15.23090031,],
        [   8.50282927,    8.64896416,    9.34308766,   13.50511891,],
        [   8.96310072,    9.17648347,    9.53450759,   12.32590822]])
    solver = SolverMatbal(wgp_all, rpr, rgip) 
    # for _ in range(10):к
    #     m_bal, sol = solver.optimize(np.ones((6)))
    #     print(sol.fun, sol.x)
    # press_calc = m_bal.calc_press(wgp_all)
    # print((press_calc - rpr).reshape(-1).shape)
    # print(pd.DataFrame({'data':(np.abs(press_calc - rpr)).reshape(-1)}).describe())
    print(solver.optimize_ga(np.ones((6))))
    

