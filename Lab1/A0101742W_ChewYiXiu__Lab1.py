import os
import numpy as np
import numpy.linalg as la


file = open("data.txt")
data = np.genfromtxt(file,delimiter=",")
file.close()

print "data =\n", data
#set vales of matrix b
b = np.matrix(np.reshape(data[:,0:2],(40,1)))
print "b =\n", b

M1 = np.zeros([40,2])
M1[::2]= np.matrix(np.reshape(data[:,2:4],(20,2)))
M2 = np.zeros([40,1])
M2[::2] = 1
#left half of matrix M
Mleft = np.hstack((M1,M2))
Mright = np.zeros([40,3])
#right half of matrix M
Mright[1::2] = Mleft[::2] 
M = np.matrix(np.hstack((Mleft,Mright)))
print "M =\n", M

a, e, r, s = la.lstsq(M, b)
print "a =\n", a
print "M*a =\n", M*a
#Square of norm of difference
print "Sum-squared error =\n", np.square(la.norm(M*a-b))
#Residue
print"Residue =\n", e
