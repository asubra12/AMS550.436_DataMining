import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(csv_in):
    dat = pd.read_csv(csv_in, header=None).as_matrix()

    resid = dat - np.mean(dat, axis=0)
    resid = np.transpose(resid)  # Usually data vectors are columns, here it is rows
    cov = resid.dot(np.transpose(resid)) / (resid[0, :].size - 1)
    evals, evecs = np.linalg.eig(cov)

    return dat, cov, evals, evecs


def predict(dat):
    x_in = np.ones((len(dat[:,0]), 2))
    x_in[:,1] = dat[:,0]
    y_in = dat[:,1]

    x_mat = np.matrix(x_in)  # Turn em into matrices
    y_mat = np.transpose(np.matrix(y_in))

    y_predict = x_mat * np.linalg.inv(x_mat.T * x_mat) * x_mat.T * y_mat

    plt.figure()
    plt.scatter(x_mat[:,1], y_mat, c='b')
    plt.scatter(x_mat[:,1], y_predict, c='r')
    plt.title('Basic prediction (2a)')
    plt.xlabel('X')
    plt.ylabel('Y')


def predict_w_error(dat):
    x_in = np.ones((len(dat[:,0]), 2))
    x_in[:,1] = dat[:,0]
    y_in = dat[:,1]

    x_mat = np.matrix(x_in)  # Turn em into matrices
    y_mat = np.transpose(np.matrix(y_in))



    return


def remove_outliers(dat):
    std0 = np.std(dat[:,0])
    std1 = np.std(dat[:,1])
    std2 = 1
    new_dat = dat[np.where(dat[:,0] < 9)]
    new_dat = new_dat[np.where(new_dat[:,0] > -4)]
    new_dat = new_dat[np.where(new_dat[:,1] > -10)]
    x_in = np.ones((len(dat[:,0]), 2))
    x_in[:,1] = dat[:,0]
    y_in = dat[:,1]

    x_in = np.delete(x_in, 299, 0)
    y_in = np.delete(y_in, 299, 0)

    x_mat = np.matrix(x_in)  # Turn em into matrices
    y_mat = np.transpose(np.matrix(y_in))

    y_predict = x_mat * np.linalg.inv(x_mat.T * x_mat) * x_mat.T * y_mat

    plt.figure()
    plt.scatter(x_mat[:,1], y_mat, c='b')
    plt.scatter(x_mat[:,1], y_predict, c='r')
    plt.title('Prediction after removal of outliers (2c)')
    plt.xlabel('X')
    plt.ylabel('Y')


def quad_fit(dat):
    x_in = np.ones((len(dat[:,0]), 3))
    x_in[:,1] = dat[:,0]
    x_in[:,2] = np.square(dat[:,0])
    y_in = dat[:,1]

    x_in = np.delete(x_in, 299, 0)
    y_in = np.delete(y_in, 299, 0)

    x_mat = np.matrix(x_in)  # Turn em into matrices
    y_mat = np.transpose(np.matrix(y_in))

    y_predict = x_mat * np.linalg.inv(x_mat.T * x_mat) * x_mat.T * y_mat

    plt.figure()
    plt.scatter(x_mat[:,1], y_mat, c='b')
    plt.scatter(x_mat[:,1], y_predict, c='r')
    plt.title('Quadratic Prediction (2d)')
    plt.xlabel('X')
    plt.ylabel('Y')
    return

dat, cov, evals, evecs = read_data('A1-Problem2.csv')
predict(dat)
remove_outliers(dat)
quad_fit(dat)



x_lin = np.ones((500, 2))
x_lin[:,1] = np.linspace(-6, 10, num=500)

x_quad = np.ones((500, 3))
x_quad[:,1] = np.linspace(-6, 10, num=500)
x_quad[:,2] = np.square(np.linspace(-6,10,num=500))
