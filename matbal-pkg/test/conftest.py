import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from mat_balance import MaterialBalance


@pytest.fixture()
def WGP_6wells()->np.ndarray:
    df = pd.read_excel("matbaL-pkg/test/fixture_files/data_6wells.xlsx")
    WGP = df.pivot_table('WGP',index='dates',columns='well_name').to_numpy()
    return WGP

@pytest.fixture()
def RPR_6wells()->np.ndarray:
    df = pd.read_excel("matbaL-pkg/test/fixture_files/data_6wells.xlsx")
    RPR = df.pivot_table('RPR',index='dates',columns='well_name').to_numpy()
    return RPR

@pytest.fixture()
def RGIP_6wells()->np.ndarray:
    df = pd.read_excel("matbaL-pkg/test/fixture_files/data_6wells.xlsx")
    RGIPG = df.pivot_table('RGIPG',index='dates',columns='well_name').to_numpy()
    return RGIPG

@pytest.fixture()
def border_j1d_like_ones_6wells()->List[int]:
            return [
            1, #[0, 1]  0
            1, #[0, 2]  1
            1, #[0, 3]  2
            1, #[0, 4]  3
            1, #[0, 5]  4
            0,            #[1, 2]  5
            1, #[1, 3]  6
            0,            #[1, 4]  7
            0,            #[1, 5]  8
            1, #[2, 3]  9
            0,            #[2, 4]  10
            1, #[2, 5]  11
            0,            #[3, 4]  12
            0,            #[3, 5]  13
            1, #[4, 5]] 14
        ]