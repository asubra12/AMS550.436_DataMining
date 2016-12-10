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


def scree_plot(evals):
    # Code below borrowed from T. Budavari, Data Mining Lecture 06
    plt.figure(figsize=(5, 5))
    plt.subplot(211); plt.ylim(0, max(evals)); plt.plot(evals, 'o-'); plt.ylabel('Eigenvalues')
    plt.subplot(212); plt.ylim(0, 110); cl = np.cumsum(evals); plt.ylabel('Percent'); plt.plot(100 * cl / cl[-1], 'o-r');
    plt.xlabel('Eigenvalue')
    return


def plot_largest_pc(evecs, dat):
    k = 2
    dat_new = np.transpose(dat)  # Need columns to be instances, not rows
    evecs_to_use = np.transpose(evecs[:, 0:k])
    new_subspace = evecs_to_use.dot(dat_new)

    plt.figure()
    plt.plot(new_subspace[0, :], new_subspace[1, :], 'bo')
    plt.title('Plotting the largest component vs next largest')
    plt.xlabel('Largest Principal Component')
    plt.ylabel('Second Largest Principal Component')

    # plt.figure()
    # plt.plot(new_subspace[(k-2),:], new_subspace[(k-1),:], 'ro')
    # plt.title('Plotting the penultimate smallest component vs smallest')
    # plt.xlabel('2nd smallest')
    # plt.ylabel('Smallest')
    return


def whitening(eval_edit, evecs, dat):
    dat_to_use = np.transpose(dat)
    temp = np.transpose(evecs).dot(dat_to_use)

    for i in eval_edit:
        if i < 0:
            i = 0

    Z = np.diag(1/np.sqrt(eval_edit)).dot(temp)

    plt.figure()
    plt.plot(Z[0,:], Z[1,:], 'bo')
    plt.title('Plotting the largest component vs next largest after Whitening')
    plt.xlabel('Largest Principal Component')
    plt.ylabel('Second Largest Principal Component')
    return

dat, cov, evals, evecs = read_data('A1-Problem3.csv')
scree_plot(evals)
plot_largest_pc(evecs, dat)
whitening(evals, evecs, dat)
