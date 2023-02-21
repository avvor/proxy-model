import pandas as pd
import numpy as np
from typing import List, Dict

n=3 # количество событий
deg=4 # max степеyь полинома

def prepare_train_and_test_sets(path: str):
    df = pd.read_excel(path, sheet_name=None,  index_col=0)
    # получаем базовую обучающую выборку 2**n вариантов
    base_train_df=pd.DataFrame()
    for key in [k for k in df if k.endswith('-xxxx'*n)]:
        df[key]['type']=int(key[:n], 2)
        base_train_df=pd.concat([base_train_df, df[key]])
    base_train_df['year']=base_train_df.index.year.to_list()
    return base_train_df, df

def init_proxy(vars, base_train_df)->pd.DataFrame:
    # преобразуем условия, заданные пользователем, в df удобный для работы
    df_vars=pd.DataFrame(data=vars)
    columns=[str(i) for i in df_vars.index]
    df_status=pd.DataFrame(columns=columns)
    df_status['year']=base_train_df.year.unique()

    for i, col in enumerate(columns):
        df_status[col]=np.where((df_vars.status[i]==1) & (df_status.year>=df_vars.year[i]), str(1), str(0))

    df_status['bin_str']=df_status[columns].agg(''.join, axis=1)
    df_status['type']=df_status['bin_str'].apply(lambda x: int(x, 2)).astype(int)
    df_status.drop(columns=columns, axis=1, inplace=True)
    
    df_res=pd.DataFrame(columns=['year', 'x', 'y'])
    df_res=df_status.merge(df_res, how='left')
    df_res=df_res.merge(base_train_df, how='inner', left_on=['year', 'type'], right_on=['year', 'type'])
    return df_res

def get_base_coefs(base_train_df):
    # получаем матрицу коэффициентов. апроксимируем полиномами степени deg бызовые варианты
    base_coefs=np.zeros((2**n, deg+1))
    for i in range(2**n):
        d=base_train_df.loc[base_train_df.type==i]
        base_coefs[i]=np.polyfit(d.total, d.yearly, deg)
    return base_coefs


def get_proxy_v1(vars, base_train_df):
    # получаем матрицу коэффициентов. апроксимируем полиномами степени deg бызовые варианты
    base_coefs=get_base_coefs(base_train_df)
    df_res=init_proxy(vars, base_train_df)
    
    # proxy_v1, идеи как у Никиты, код мой
    df_res.loc[0, 'x']=df_res.loc[0, 'total']
    for i in range(len(df_res)):
        fb=np.poly1d(base_coefs[df_res.type[i]])
        df_res.loc[i, 'y']=fb(df_res.x[i])
        if (i+1)<len(df_res):
            df_res.loc[i+1, 'x']=df_res.x[i]+df_res.y[i]
            
    # df_res.drop(['daily', 'total', 'yearly'], axis=1, inplace=True)
    df_res.drop(['total', 'yearly'], axis=1, inplace=True)
    return df_res


def get_proxy_v2(vars, base_train_df, df):
    base_coefs=get_base_coefs(base_train_df)
    df_res=init_proxy(vars, base_train_df)
    # proxy_v2, с учетом приростов только от базового варианта
    base_type=0
    df_res.loc[0, 'x']=df_res.loc[0, 'total']
    for i in range(len(df_res)):
    #proxy_v2
        if df_res.type[i]==base_type:
            base_f=np.poly1d(base_coefs[df_res.type[i]])
            df_res.loc[i, 'y']=base_f(df_res.x[i])
        else:
            # вычисляем тренд по приростам
            train_set_keys=[k for k in df if int(k[:3], 2)==df_res.type[i] and k[3:]!='-xxxx'*3]
            x, y=[],[]
            for key in train_set_keys:
                g=df[key]
                g['growth_by_year']=g.yearly-base_train_df.loc[base_train_df.type==base_type].yearly
                g=g.loc[g.growth_by_year.cumsum()>0]
                if len(g)>0:
                    x.append(g.iloc[0].total)
                    y.append(g.iloc[0].growth_by_year)
            growth_coefs = np.polyfit(x, y, deg) if len(x)>=(deg+1) else np.polyfit(x, y, len(x)-1)
        
            base_f=np.poly1d(base_coefs[base_type])
            growth_f=np.poly1d(growth_coefs)
            df_res.loc[i, 'y']=base_f(df_res.x[i])+growth_f(df_res.x[i])
            base_type=df_res.type[i]
        if (i+1)<len(df_res):
            df_res.loc[i+1, 'x']=df_res.x[i]+df_res.y[i]
    df_res.drop(['total', 'yearly'], axis=1, inplace=True)
    return df_res





