import pandas as pd
import numpy as np
import Base_Multiple_Impute as bmi
import MyGA as ga

dataSe = pd.read_csv("indian_liver_patient.csv")
pd.options.display.max_columns = False
print(dataSe.shape)
print(dataSe.head())
print(dataSe)
dataSe["Sorting_col"] = np.random.randint(1, 600, dataSe.shape[0])
result = dataSe.sort_values(by=['Sorting_col'])
print(result.shape)
print(result)
result.loc[result.Gender == "Female", "Gender"] = "0"
result.loc[result.Gender == "Male", "Gender"] = "1"
result = result.drop(columns=['Dataset'])
result.to_csv("indian_liver_patient2.csv", index=False)
print(result)
