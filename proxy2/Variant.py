import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Timestamp
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.linalg import lstsq

df = pd.read_excel('files/smspecs_yearly.xlsx', sheet_name=None,  index_col=0)
class Variant:
    _df:pd.DataFrame=None
    dates:List[Timestamp]=None
    base:'Variant'=None

    @classmethod
    def from_xls_smspecs_yearly(cls, df, key, base=None):
        obj :Variant= cls()
        obj.base = base
        obj.full_name = key
        obj.dates = []
        for bin_var, date_line in zip(key.split('-')[0], key.split('-')[1:]):
            if bin_var == '1':
                if date_line =='xxxx':
                    obj.dates.append(pd.to_datetime('2013-01-01', format='%Y-%m-%d'))
                else:
                    obj.dates.append(pd.to_datetime(f'{date_line}-01-01', format='%Y-%m-%d'))
            else:
                if date_line == 'xxxx':
                    obj.dates.append(pd.to_datetime(f'2031-01-01', format='%Y-%m-%d'))
                else:
                    obj.dates.append(pd.to_datetime(f'{date_line}-01-01', format='%Y-%m-%d'))
        
        obj._df:pd.DataFrame = df / 10**9        
        obj._trendC00 = None
        obj._RGI = None        
        return obj
        
    @property
    def _get_totals_by_event(self)->np.ndarray:
        # return self._df[self._df.index.isin(self.dates)].total.to_numpy()
        # return [None if date is None else self._df.loc[date].total for date in self.dates]
        df_res = pd.DataFrame(index=self.dates,  data={'total':[0]*len(self.dates)})
        return df_res
        
    @property
    def get_borders(self)->pd.DataFrame:
        df_res = pd.merge(self._df, self._get_totals_by_event, left_index=True, right_index=True,how='right')
        df_res = df_res.rename(columns={'total_x':'total'})     
        df_res = df_res.drop(['total_y','yearly'], axis=1)
        df_res = df_res.fillna(math.inf)
        return df_res

    @property
    def name(self) -> int: return int(self.full_name.split('-')[0],base=2)
    def __repr__(self) -> str:
        return '\n'.join([
            repr(self._df),
            f'name={self.name:03b} dates={self.dates}',
            self.full_name
        ])

    @property
    def get_trends(self)->List[RGI]:
        if self._RGI is None:
            self._RGI = []
            df_c = self._df
            min_val = - math.inf
            for cur_total in self.get_borders.total:
                x_ar = df_c[(min_val <= df_c.total) & (df_c.total <cur_total)].total.to_numpy()
                y_ar = df_c[(min_val <= df_c.total) & (df_c.total <cur_total)].yearly.to_numpy()
                if len(x_ar) == 0:
                    # if self.base is None:
                    x_ar = self._df.total.to_numpy()
                    y_ar = self._df.yearly.to_numpy()
                    # else:
                        # x_ar = self.base._df.total.to_numpy()
                        # y_ar = self.base._df.yearly.to_numpy()
                # else:
                if len(x_ar) <= 4:
                    interp = RGI((x_ar,), y_ar, method='linear', bounds_error=False,fill_value=None)
                else:                
                    interp = RGI((x_ar,), y_ar, method='pchip', bounds_error=False,fill_value=None)                
                self._RGI.append(interp)
                
                min_val = cur_total
        return self._RGI

    def get_trend_val(self, val:float)->float:
        for _RGI, total in zip(self.get_trends, self.get_borders.total):
            if val < total:
                return _RGI([val])[0]
        assert True, f'Чо то с трендами, {self}'

    def show(self):
        fig, axs = plt.subplots(ncols=2)
        axs[0].plot(self._df.total, self._df.yearly, '.')
        x_arr = np.linspace(self._df.total.min(), self._df.total.max(), 500)
        axs[0].plot(x_arr, [ self.get_trend_val(x) for x in x_arr])
        axs[0].set_xlim(0)
        # axs[1].set_xlim(0)
        if not self.base is None:
            dq = self.get_dq
            axs[1].plot(dq.x, dq.dq)
        plt.show()

    @property
    def get_dq(self)->pd.DataFrame:
        # x_arr = self.base._df[self.base._df.total > self.get_borders.total[0]].total
        x_arr = self._df[self.base._df.total > self.get_borders.total[0]].total
        return pd.DataFrame({
            'x':x_arr,
            'dq':[
                    self.get_trend_val(total) - self.base.get_trend_val(total)
                for total in x_arr]                
        })