import pandas as pd
import numpy as np

#To Calculate theta values
def calc_theta_df(x, y_column_name):
    y = x[[y_column_name]]
    x = x.drop(columns=[y_column_name])
    x.insert(0,'Theta0', 1)
    xT = x.T
    yT = y.T
    th = np.dot(np.dot(yT,x), np.linalg.inv(np.dot(xT,x)))
    th = pd.DataFrame(th)
    return th

#To Calculate y_hat with new theta values
def calc_y(th_ds, x, ro, y_column_name):
    y = x[[y_column_name]]
    x = x.drop(columns=[y_column_name])
    x.insert(0,'Theta0', 1)
    yh = th_ds.iat[0, 0]
    i = 1
    while i < len(th_ds.columns):        
        yh = yh + th_ds.iat[0, i] * x.iat[ro, i]
        i += 1
    return round(yh, 5)

#To calculate theta values and get y_hat (all Multiple Imputation operations)
def get_yh(x, ro, y_column_name):
    y = x[[y_column_name]]
    x = x.drop(columns=[y_column_name])
    x.insert(0,'Theta0', 1)
    xT = x.T
    yT = y.T
    th = np.dot(np.dot(yT,x), np.linalg.inv(np.dot(xT,x)))
    th = pd.DataFrame(th)
    yh = th.iat[0, 0]
    i = 1
    while i < len(th.columns):
        yh = yh + th.iat[0, i] * x.iat[ro, i]
        i += 1
    return round(yh, 5)
