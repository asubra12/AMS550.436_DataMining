import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


def comp_hist(feat, labels, normal, comp, title):
    num_features = len(feat)
    fig, ax = plt.subplots(1, num_features)
    fig.suptitle(title)

    print num_features

    if num_features == 1:
        ax.hist(normal[:, feat], bins=20, color='r')
        ax.hist(comp[:, feat], bins=20, color='g')
        ax.set_xlabel(labels[0])
        return

    for i in range(num_features):
        print 'i =', i
        ax[i].hist(normal[:, feat[i]], bins=20, color='r')
        ax[i].hist(comp[:, feat[i]], bins=20, color='g')
        ax[i].set_xlabel(labels[i])
    return


def comp_scatter(feat, labels, normal, comp, title):
    num_features = len(feat)

    fig, ax = plt.subplots(num_features, num_features, sharex='col', sharey='row')
    fig.suptitle(title)

    if num_features == 1:
        print 'Only one variable, no need to do cross-scatters'
        plt.close(fig)
        return

    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                if i == num_features - 1:
                    ax[i, j].set_xlabel(labels[j])
                if j == 0:
                    ax[i, j].set_ylabel(labels[i])
                pass
            else:
                ax[i, j].scatter(normal[:, feat[j]], normal[:, feat[i]], s=20, c='r')
                # ax[i,j].scatter(cad[:, rabb_features[j]], cad[:, rabb_features[i]], s=20, c='b')
                ax[i, j].scatter(comp[:, feat[j]], comp[:, feat[i]], s=20, c='g')
                if i == num_features - 1:
                    ax[i, j].set_xlabel(labels[j])
                if j == 0:
                    ax[i, j].set_ylabel(labels[i])
    return


headers = ['']

data = np.asmatrix(pd.read_csv('arrythmia.data', header=None))
miss_val = np.where(data=='?')
m_rows = miss_val[0]
m_cols = miss_val[1]
qmarks = np.bincount(miss_val[1])  # Number of times qmarks appear by index

'''
Missing Vector angles in degrees on front plane of
T Wave: 8, P Wave: 22 , QRST: 1, J: 376
Missing Heart Rate: 1
This suggests that we should not use J Wave in our analysis.
However, we want to use heart rate because it's clinically relevant, so lets just remove the entry that is missing it
'''

T = 10
P = 11
QRST = 12
J = 13
HR = 14

data[:, [T, P, QRST, J]] = float('NaN')  # Set these to NaN so we don't get errors later
data = np.delete(data, m_rows[m_cols==HR], axis=0)  # Del rows where we're missing HR, because we want to use HR later
labels = data[:, -1]
data = data.astype(int)

new_labels = np.zeros(len(labels))
for i in range(len(labels)):
    new_labels[i] = int(labels[i])
labels = new_labels

# labels = np.in1d(labels, [1, 2, 10])  # On ly get patients that are normal, have CAD, or right atrial branch block
norm = data[labels==1, :]
cad = data[labels==2, :]
rabb = data[labels==10, :]
brady = data[labels==6, :]

rabb_features = np.asarray([5, 18, 163, 167, 169]) - 1
feat_label = ['QRS (ms)', 'S Width', 'S Height', 'T Height', 'QRST Area']

if len(rabb_features) != len(feat_label):
    print 'Num Features and Num Labels do not match'
    sys.exit()

comp_hist(rabb_features, feat_label, norm, rabb, 'Right Atrial Branch Block Features')
comp_scatter(rabb_features, feat_label, norm, rabb, 'Right Atrial Branch Block Features')


cad_features = np.asarray([15, 163, 167, 169]) - 1
feat_label = ['HR', 'S Height', 'T Height', 'QRST Area']

if len(cad_features) != len(feat_label):
    print 'Num Features and Num Labels do not match'
    sys.exit()

comp_hist(cad_features, feat_label, norm, cad, 'Coronary Artery Disease Features')
comp_scatter(cad_features, feat_label, norm, cad, 'Coronary Artery Disease Features')


brad_features = np.asarray([15]) - 1
feat_label = ['HR']
comp_hist(brad_features, feat_label, norm, brady, 'Sinus Bradycardia')
comp_scatter(brad_features, feat_label, norm, brady, 'Sinus Bradycardia')



'''
Right Bundle Branch Block (rabb) is characterized by a broad QRS (feature 5), a
    secondary R wave (R', feature 19), in leads V1-3, and a wide S wave (feature 18)
Coronary Artery Disease is characterized by changes in heart rate (feature 15), ST elevation (feature 163, 167, 169)
'''