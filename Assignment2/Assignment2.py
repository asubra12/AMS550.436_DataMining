import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

df = pd.read_csv('A2-Problem.csv',index_col='Pixel',dtype=np.uint8)  # Read in the csv file 'A2-Problem.csv'. Index_col assigns the column titled 'Pixel' as the column of row labels. Import all values as 8bit ints.
X = np.stack([df[k].reshape((40,40)) for k in df.columns], axis=2)  # Maps a list of 1600 pixels to their respective position in a 40x40 image matrix.

plt.figure(figsize=(12,3))  # Creates an empty figure
for i in range(X.shape[2]):  # For each of the R, G, B values
    plt.subplot(1,4,i+1)   # Create a subplot within the empty figure
    plt.imshow(X[:,:,i], interpolation='nearest', cmap=mpl.cm.gray)  # Plot a gray scale version of either the R, G, or B values
plt.subplot(1, 4, 4); plt.imshow(X, interpolation='nearest')  # Plot the original image


from sklearn.cluster import KMeans

dfmat = df.as_matrix()

names = ['R', 'G', 'B']
for i in range(3):
    plt.figure()
    plt.hist(X[:,:,i])
    plt.title('Histogram of ' + names[i] + ' values from the Image')
    plt.xlabel(names[i] + ' Values')
    plt.ylabel('Frequency')

max_clusters = 15  # Attempt a total of 7 clusters
scores = np.zeros(max_clusters)
for j in range(1,max_clusters+1):  # Attempt a different number of clusters
    nbrs = KMeans(n_clusters=j)  # Initialize a KMeans algorithm
    labels = nbrs.fit_predict(dfmat)
    score = nbrs.score(dfmat)
    scores[j-1] = score

plt.figure()
plt.plot(scores)
plt.scatter(2, scores[2], marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dfmat[:,0], dfmat[:,1], dfmat[:,2], )






