##### INITIALIZATION ##########################################################

import numpy as np #numerical python library
import pdb #debugging & halting

pdb.set_trace()

N = 10**2

pdb.set_trace()

col = np.ones((4,1))
row = np.ones((1,4))

pdb.set_trace()

#numpy matrix
mat = np.array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

pdb.set_trace()

A = np.ones( (N,N) )
B = np.ones((N,N))
C = np.zeros((N,N))

##### COMPUTATION #############################################################

# Compute matrix product manually

pdb.set_trace()

for i in range(0,N):
	for j in range(0,N):
		for k in range(0,N):
			C[i,j] = 	C[i,j] + A[i,k]*B[k,j]

# Compute matrix multiplication as C = A*B

pdb.set_trace()

C = np.dot(A,B)

## Possible extensions
# inner and outer product
# slicing 
# elementwise product

##### ACCESS ##################################################################

pdb.set_trace()

c = np.random.rand(4,6)  #generate random matrix
print(c[1,2])	#Element 2,3 (row,col)
print(c[0,:])    #First row
print(c[:,0])	#First column

print(c[1:,])	#All, except first row
print(c[-2:,])	#Last two rows
print(c[::2,:]) #Strides: Every other row

##### FUNCTIONS ###############################################################

#Use of external function

pdb.set_trace()

from my_func import my_sum

mysum = my_sum(5,7) #function call

##### PLOTTING ################################################################

pdb.set_trace()

import matplotlib.pyplot as plt

x = np.linspace(0,2*np.pi,100)
ycos = np.cos(x)
ysin = np.sin(x)

pdb.set_trace()

#The command will not plot but buffering it
plt.plot(x,ysin)
#To show the plot
plt.show()

pdb.set_trace()

#Two plots
plt.plot(x,ycos,x,ysin)
plt.show() 

##### DATA IM&EXPORT ##########################################################

pdb.set_trace()

data = np.random.rand(10,10)

#Write data to a csv file, remembering the comma separation
np.savetxt("my_data.csv", data, delimiter=", ")

#Load csv file

out = open("my_data.csv","rb")
loaded = np.loadtxt(out,delimiter=", ")

#Verify that data loaded are lossless
print(np.array_equal(data,loaded))
