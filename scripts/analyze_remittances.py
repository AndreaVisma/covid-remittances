
import pandas as pd

data_folder = "C:\\git-projects\\covid-remittances\\data\\"
mex_rem = "mex_remittances.xlsx"
pak_rem = "pak_remittances_detail.pkl"
phil_rem = "phil_remittances_detail.pkl"

mex = pd.read_excel(data_folder + mex_rem)
pak = pd.read_pickle(data_folder + pak_rem)
phil = pd.read_pickle(data_folder + pak_rem)