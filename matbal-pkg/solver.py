from mat_balance import MaterialBalance
from scipy.optimize import minimize
import numpy as np
import pandas as pd

class SolverMatbal:
    def __init__(self, mat_bal:MaterialBalance, rpr:np.ndarray, voronoi_regs:np.ndarray):
        self.matbal = mat_bal
        self.rpr = mat_bal.p_z(rpr)
        self.n=len(voronoi_regs)
        self.voronoi_regs=voronoi_regs
        
    def get_j2d(self, j_1d: np.ndarray):
        indecies = np.triu_indices_from(np.ones((self.n, self.n)),1)
        b = np.zeros((self.n, self.n))
        b[indecies] = j_1d
        return b + b.T

    def get_press(self, j_1d:np.ndarray):
        J_2d = self.get_j2d(j_1d)
        res_calc = self.matbal.calc(J_2d)
        press = np.array([r.x for r in res_calc])
        return press

    def get_diff(self, j_1d:np.ndarray):
        press_calc = self.get_press(j_1d)    
        fact_model = np.array([press_calc.reshape(-1), self.rpr.reshape(-1)]).T
        diff = (np.diff(fact_model, axis=1))**2
        return diff
    
    def opt_mininize(self, lower: int=0, upper: int = 9999):
        bnds = [(0, 0) if x==0 else (lower, upper) for x in self.voronoi_regs[np.triu_indices_from(self.voronoi_regs, 1)].reshape(-1)]
        x0 = np.zeros(len(bnds))
        power_func = np.vectorize(lambda x:x)
        sol = minimize(fun=lambda x: power_func(self.get_diff(x)).sum(),
            x0=x0,
            method='powell',
            tol= 1e-3,
            options={'maxiter': 1e+8, 'disp': False},
            bounds=bnds)
        return sol
    
    def opt_root(self, lower: int=0, upper: int = 9999):
        bnds = [(0, 0) if x==0 else (lower, upper) for x in self.voronoi_regs[np.triu_indices_from(self.voronoi_regs, 1)].reshape(-1)]
        x0 = np.zeros(len(bnds))
        power_func = np.vectorize(lambda x:x)
        sol = minimize(fun=lambda x: power_func(self.get_diff(x)).sum(),
            x0=x0,
            # method='powell',
            # tol= 1e-3,
            # options={'maxiter': 1e+8, 'disp': True},
            bounds=bnds)
        return sol