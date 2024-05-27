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

def get_best_yh(th_df, ds, ro, y_column_name):
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
    err1 = round(abs(y - yh1),4)
    err2 = round(abs(y - yh2),4)
    err3 = round(abs(y - yh3),4)
    err4 = round(abs(y - yh4),4)
    err5 = round(abs(y - yh5),4)
    err6 = round(abs(y - Myh),4)
    
    err_list = [err1, err2, err3, err4, err5, err6]
    values_list = [yh1, yh2, yh3, yh4, yh5, Myh]
    
    min_qual_idx = np.argmin(err_list)
    best_val = values_list[min_qual_idx]
    return best_val, min_qual_idx 

def calc_last_yh(th_df, ds, ro, y_column_name, best_vals):
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

    Syh1 = yh1 + yh2 + yh3 + yh4 + yh5
    Myh1 = round((Syh1 / 5),4)
    
    y_h_rslt = [yh1, yh2, yh3, yh4, yh5, Myh1]
    
    st_idx = best_vals[0]
    nd_idx = best_vals[1]
    #rd_idx = best_vals[2]
    Syh = y_h_rslt[st_idx] + y_h_rslt[nd_idx] + Myh1
    Myh = round((Syh / 3),4)
    return Myh

def calc_missin_yh(th_df, ds, ro, y_column_name, best_vals):
    th_df1 = th_df[0:1]
    th_df2 = th_df[1:2]
    th_df3 = th_df[2:3]
    th_df4 = th_df[3:4]
    th_df5 = th_df[4:5]
    y = ds.iat[ro,2]
    yh1 = bmi.calc_y(th_df1, ds, ro, y_column_name)
    yh2 = bmi.calc_y(th_df2, ds, ro, y_column_name)
    yh3 = bmi.calc_y(th_df3, ds, ro, y_column_name)
    yh4 = bmi.calc_y(th_df4, ds, ro, y_column_name)
    yh5 = bmi.calc_y(th_df5, ds, ro, y_column_name)

    Syh1 = yh1 + yh2 + yh3 + yh4 + yh5
    Myh1 = round((Syh1 / 5),4)
    
    y_h_rslt = [yh1, yh2, yh3, yh4, yh5, Myh1]
    
    st_idx = best_vals[0]
    nd_idx = best_vals[1]
    Syh = y_h_rslt[st_idx] + y_h_rslt[nd_idx] + Myh1
    Myh = round((Syh / 3),4)
    return Myh

def select_final_mating(vals_2_idx):
    vals_idx = []
    for i in range(2):
        max_qual_idx = np.argmax(vals_2_idx)
        vals_idx.append(max_qual_idx)
        vals_2_idx[max_qual_idx] = -1
    return vals_idx

dataSe = pd.read_csv("indian_liver_patient_fill.csv")
dataSe = dataSe.dropna()

thet_df = pd.DataFrame()
thet_df = pd.read_csv("Selected_Vectors_Albumin_GR.csv")

mi_thet_df = pd.DataFrame()
mi_thet_df = pd.read_csv("th_df.csv")

mat_list = [0,0,0,0,0,0]
for i in range(len(dataSe)):
    y_val = dataSe.iat[i,2]
    yh_bef = calc_yh(mi_thet_df, dataSe, i, 'Albumin_and_Globulin_Ratio')
    yh_af = get_best_yh(thet_df, dataSe, i, 'Albumin_and_Globulin_Ratio')
    yh_af_val = yh_af[0]
    yh_af_idx = yh_af[1]
    mat_list[yh_af_idx] = mat_list[yh_af_idx] + 1
    abs_bef = np.abs(y_val - yh_bef)
    abs_af = np.abs(y_val - yh_af_val)

best_vals = select_final_mating(mat_list)
print('best_vals :',best_vals)

final_result = pd.DataFrame(columns=['y_val', 'y_hat_before', 'y_hat_after_1', 'y_hat_after_2', 'abs_Er_before', 'abs_Er_after_1', 'abs_Er_after_2'])

for i in range(len(dataSe)):
    y_val = dataSe.iat[i,2]
    yh_bef = calc_yh(mi_thet_df, dataSe, i, 'Albumin_and_Globulin_Ratio')
    yh_af1 = calc_yh(thet_df, dataSe, i, 'Albumin_and_Globulin_Ratio')
    yh_af = calc_last_yh(thet_df, dataSe, i, 'Albumin_and_Globulin_Ratio', best_vals)
    abs_bef = np.abs(y_val - yh_bef)
    abs_af = np.abs(y_val - yh_af)
    abs_af1 = np.abs(y_val - yh_af1)
    final_result.loc[i] = [y_val, yh_bef, yh_af1, yh_af, abs_bef, abs_af1, abs_af]
    
final_result.to_csv("Albumin_and_Globulin_Ratio.csv", index=False)

dataSe_fill = pd.read_csv("indian_liver_patient_fill.csv")
y_missing_209 = calc_missin_yh(thet_df, dataSe_fill, 209, 'Albumin_and_Globulin_Ratio', best_vals)
dataSe_fill.at[209, 'Albumin_and_Globulin_Ratio'] = round(y_missing_209,1)

y_missing_241 = calc_missin_yh(thet_df, dataSe_fill, 241, 'Albumin_and_Globulin_Ratio', best_vals)
dataSe_fill.at[241, 'Albumin_and_Globulin_Ratio'] = round(y_missing_241,1)

y_missing_253 = calc_missin_yh(thet_df, dataSe_fill, 253, 'Albumin_and_Globulin_Ratio', best_vals)
dataSe_fill.at[253, 'Albumin_and_Globulin_Ratio'] = round(y_missing_253,1)

y_missing_312 = calc_missin_yh(thet_df, dataSe_fill, 312, 'Albumin_and_Globulin_Ratio', best_vals)
dataSe_fill.at[312, 'Albumin_and_Globulin_Ratio'] = round(y_missing_312,1)

dataSe_fill.to_csv("indian_liver_patient_fill_final.csv", index=False)
print('y_missing_209 :', round(y_missing_209,1))
print('y_missing_241 :', round(y_missing_241,1))
print('y_missing_253 :', round(y_missing_253,1))
print('y_missing_312 :', round(y_missing_312,1))
