import pandas as pd
import numpy as np
import Base_Multiple_Impute as bmi

def fitness_fun_th(th_ds, ds, y_colNm, y_colidx):
    diff = 0
    for i in range(len(ds)):
        yh = bmi.calc_y(th_ds, ds, i, y_colNm)
        difr = 1/round((ds.iat[i, y_colidx] - yh),5)
        diff = diff + difr
        i += 1
    quality = round(diff / len(ds),8)    
    return quality

def fitness_fun(th_ds, ds, y_colNm, y_colidx):
    diff = 0
    for i in range(len(ds)):
        yh = bmi.calc_y(th_ds, ds, i, y_colNm)
        difr = round((ds.iat[i, y_colidx] - yh),5)
        diff = diff + difr
        i += 1
    quality = round(diff / len(ds),8)
    return quality

def fitness_fun_th_abs(th_ds, ds, y_colNm, y_colidx):
    diff = 0
    for i in range(len(ds)):
        yh = bmi.calc_y(th_ds, ds, i, y_colNm)
        difr = 1/np.abs(ds.iat[i, y_colidx] - yh)
        diff = diff + difr
        i += 1
    quality = round(diff / len(ds),8)
    return quality

def pop_fitness(th_df, ds, y_colNm, y_colidx):
    qualities = np.zeros(th_df.shape[0])
    i = 1
    c = 0
    while i < len(th_df) +1:
        qualities[c] = fitness_fun_th_abs(th_df[c: i], ds, y_colNm, y_colidx)
        i += 1
        c += 1
    return qualities

def select_mating(th_df, qualities, num_parents):
    parents = pd.DataFrame()
    for parent_num in range(num_parents):
        max_qual_idx = np.argmax(qualities)
        parents = parents.append(th_df[max_qual_idx :max_qual_idx +1])
        qualities[max_qual_idx] = -1
    return parents

def crossover(parents, sol_per_pop):
    new_population = np.empty((sol_per_pop,len(parents.columns)))
    itra = len(parents.columns) - 1
    parents = parents.to_numpy()
    new_population[0:parents.shape[0], :] = parents

    new_population[3, 0:0+1] = parents[1, 0:0+1]
    new_population[3, 0+1:] = parents[0, 0+1:]

    new_population[4, 0:0+1] = parents[2, 0:0+1]
    new_population[4, 0+1:] = parents[1, 0+1:]

    new_population[5, 0:0+1] = parents[0, 0:0+1]
    new_population[5, 0+1:] = parents[2, 0+1:]

    i = 0
    r = 6
    for i in range(itra):
        new_population[r, 0:  i+1] = parents[0, 0:i+1]
        new_population[r, i+1:i+2] = parents[1, i+1:i+2]
        new_population[r, i+2:   ] = parents[0, i+2:]

        new_population[r+1, 0:  i+1] = parents[1, 0:i+1]
        new_population[r+1, i+1:i+2] = parents[2, i+1:i+2]
        new_population[r+1, i+2:   ] = parents[1, i+2:]

        new_population[r+2, 0:  i+1] = parents[2, 0:i+1]
        new_population[r+2, i+1:i+2] = parents[0, i+1:i+2]
        new_population[r+2, i+2:   ] = parents[2, i+2:]
        r += 3
        i += 1

    new_population = pd.DataFrame(new_population)
    return new_population
