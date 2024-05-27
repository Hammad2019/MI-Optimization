import pandas as pd
import numpy as np
import Base_Multiple_Impute as bmi
import MyGA as ga

dataSe = pd.read_csv("indian_liver_patient2.csv")
print(dataSe.shape)

ds1 = dataSe[:49]
ds2 = dataSe[50:99]
ds3 = dataSe[100:149]
ds4 = dataSe[150:199]
ds5 = dataSe[200:249]
ds6 = dataSe[250:299]
ds7 = dataSe[300:349]
ds8 = dataSe[350:399]
ds9 = dataSe[400:449]
ds10 = dataSe[450:499]

th_ds1 = bmi.calc_theta_df(ds1, 'Total_Bilirubin')
th_ds2 = bmi.calc_theta_df(ds2, 'Total_Bilirubin')
th_ds3 = bmi.calc_theta_df(ds3, 'Total_Bilirubin')
th_ds4 = bmi.calc_theta_df(ds4, 'Total_Bilirubin')
th_ds5 = bmi.calc_theta_df(ds5, 'Total_Bilirubin')
th_ds6 = bmi.calc_theta_df(ds6, 'Total_Bilirubin')
th_ds7 = bmi.calc_theta_df(ds7, 'Total_Bilirubin')
th_ds8 = bmi.calc_theta_df(ds8, 'Total_Bilirubin')
th_ds9 = bmi.calc_theta_df(ds9, 'Total_Bilirubin')
th_ds10 = bmi.calc_theta_df(ds10, 'Total_Bilirubin')

th_df = pd.DataFrame()

th_df = th_df.append(th_ds1, ignore_index=True)
th_df = th_df.append(th_ds4, ignore_index=True)
th_df = th_df.append(th_ds6, ignore_index=True)
th_df = th_df.append(th_ds7, ignore_index=True)
th_df = th_df.append(th_ds8, ignore_index=True)
th_df = th_df.append(th_ds9, ignore_index=True)

dataSe = dataSe.dropna()

print(dataSe.shape)

# Population_size
sol_per_pop = 36

# Mating pool size
num_parents = 3

# Initial population
new_population = th_df

qualities = ga.pop_fitness(new_population, dataSe, 'Total_Bilirubin', 2)
print('qualities before :', qualities)

# Selecting the best parents in the population for mating
parents = ga.select_mating(new_population, qualities, num_parents)

# Generating next generation using crossover
new_population = ga.crossover(parents, sol_per_pop)
new_population.to_csv("new_population.csv", index=False)

qualities = ga.pop_fitness(new_population, dataSe, 'Total_Bilirubin', 2)
print('qualities Second :',qualities)

final_population = ga.select_mating(new_population, qualities, 6)
final_population.to_csv("Selected_Vectors.csv", index=False)

qualities = ga.pop_fitness(final_population, dataSe, 'Total_Bilirubin', 2)
print('qualities after :',qualities)
