import matplotlib.pyplot as plt
import numpy as np


data=np.loadtxt("../data/rand_points.txt",delimiter=" ")

ax=plt.axes(projection="3d")


ax.scatter3D(data[:,0],data[:,1],data[:,2],s=0.01)
plt.show()


####

def triplet():
	triplet=np.zeros((1,3))
	for i in range(3):
		triplet[0,i]=1e8*np.random.rand()
	return triplet

n=100000

points=np.zeros((n,3))

for i in range(n):
	points[i]=triplet()

axpy=plt.axes(projection='3d')
axpy.scatter3D(points[:,0],points[:,1],points[:,2],s=0.01)
plt.show()
