import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(csv_in):
    # Question 1a
    dat = pd.read_csv(csv_in, header=None).as_matrix()
    dat = np.matrix(dat)
    return dat


def get_quant_measures(df):
    # Question 1b
    mean = float(df.mean())
    sd = float(df.std())
    var = float(df.var())

    return mean, sd, var


def visualization(df):
    # Question 1c
    plt.figure()
    plt.title('Histogram of all values')
    plt.xlabel('Bin')
    plt.ylabel('Frequency')
    plt.hist(df, 40)

    plt.figure()
    plt.title('Time Series of all values')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.plot(df)

    plt.figure()
    plt.title('Scoping in on the values at the end')
    plt.xlabel('Time')
    plt.ylabel('Value')
    x = np.arange(970, 999, 1)
    plt.plot(x, df[970:999])

    new_df = df[:975]
    new_params = get_quant_measures(new_df)

    old_params = get_quant_measures(df)
    print ("At instance 975, the value jumps up from -48 to -45")
    print ("Excluding indices 975 onwards, the new mean is {:f}, the new stddev is {:f},"
          "and the new variance is {:f}".format(new_params[0], new_params[1], new_params[2]))

    print ("For reference, the mean, std dev, and variance of the entire data"
           "set is {:f}, {:f}, {:f}".format(old_params[0], old_params[1], old_params[2]))

dat = read_data('A1-Problem1.csv')
mean, sd, var = get_quant_measures(dat)
print ("The Mean, Std Dev, and Var are {}, {}, and {}, respectively".format(mean, sd, var))
visualization(dat)
