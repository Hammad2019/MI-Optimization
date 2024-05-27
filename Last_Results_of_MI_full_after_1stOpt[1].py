import pandas as pd
import numpy as np
import Base_Multiple_Impute as bmi

def calc_yh(th_df, ds, ro, y_column_name):
    th_df1 = th_df[0:1]
    th_df2 = th_df[1:2]
    th_df3 = th_df[2:3]
    th_df4 = th_df[3:4]
    th_df5 = th_df[4:5]
    #th_df6 = th_df[5:6]
    y = ds.iat[ro,2]
    yh1 = bmi.calc_y(th_df1, ds, ro, y_column_name)
    yh2 = bmi.calc_y(th_df2, ds, ro, y_column_name)
    yh3 = bmi.calc_y(th_df3, ds, ro, y_column_name)
    yh4 = bmi.calc_y(th_df4, ds, ro, y_column_name)
    yh5 = bmi.calc_y(th_df5, ds, ro, y_column_name)
    #yh6 = bmi.calc_y(th_df6, ds, ro, y_column_name)
    Syh = yh1 + yh2 + yh3 + yh4 + yh5
    Myh = round((Syh / 5),4)
    return Myh 

dataSe = pd.read_csv("indian_liver_patient2.csv")
dataSe = dataSe.dropna()

thet_df = pd.DataFrame()
thet_df = pd.read_csv("Selected_Vectors.csv")

mi_thet_df = pd.DataFrame()
mi_thet_df = pd.read_csv("th_df.csv")

result = pd.DataFrame(columns=['y_val', 'y_hat_before', 'y_hat_after', 'abs_Er_before', 'abs_Er_after'])

for i in range(len(dataSe)):
    y_val = dataSe.iat[i,2]
    yh_bef = calc_yh(mi_thet_df, dataSe, i, 'Total_Bilirubin')
    yh_af = calc_yh(thet_df, dataSe, i, 'Total_Bilirubin')
    abs_bef = np.abs(y_val - yh_bef)
    abs_af = np.abs(y_val - yh_af)
    result.loc[i] = [y_val, yh_bef, yh_af, abs_bef, abs_af]

result.to_csv("last_result_full.csv", index=False)
