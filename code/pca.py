import csv
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

# read in data 
filename = "data_clean.csv"
    
with open(filename, 'r') as inp:
    csvreader = csv.reader(inp) # intialze reader

    data = np.array(list(csvreader)).astype(float)

# get y values, i.e. true outputs -----------------------------------------------------------------------------------------
y = data[:,-1]

# normalize data
for i in range(len(data[0])):
    data[:,i]=(data[:,i]-np.min(data[:,i]))/(np.max(data[:,i])-np.min(data[:,i]))

# get x values, i.e. input vector
x = data[:,:-1]

# calculate the PCA, with specified number of dimensions to be 3
pca = PCA(n_components=3)
x2 = pca.fit_transform(x)

# Create PCA plot to show how entagled the true outputs for the tumors are

# define color map, plot size and projection, and name the graph
fig = plt.figure(figsize = (7, 7))
ax = plt.axes(projection ="3d")

my_cmap = plt.get_cmap('prism')

plt.title("PCA")

# plot the points on the scatter graph ------------------------------------------------------------------------------------
ax.scatter3D(x2[:,0], x2[:,1], x2[:,2], c=-y, cmap=my_cmap)

# create legend for colors
green_patch = mpatches.Patch(color='lime', label='Benign')
red_patch = mpatches.Patch(color='red', label='Malignant')

# add legend to graph
plt.legend(handles=[green_patch, red_patch])

# show plot
plt.show()


# Creating box plot to show the how the distribution of the columns -------------------------------------------------------

# define figure size and axes
fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([0, 0, 1, 1])

# plot values
ax.boxplot(x, showfliers=False)

# show box plot
plt.show()
