import matplotlib.pyplot as plt


def comp_hist(feat, labels, normal, comp, title):
    num_features = len(feat)
    fig, ax = plt.subplots(1, num_features)
    fig.suptitle(title)

    for i in range(num_features):
        ax[i].hist(normal[:, feat[i]], bins=20, color='r')
        ax[i].hist(comp[:, feat[i]], bins=20, color='g')
        ax[i].set_xlabel(labels[i])
    return
