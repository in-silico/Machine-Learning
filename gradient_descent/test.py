#! /usr/bin/python

from gradient_descent import *
from numpy import *
import matplotlib.pyplot as plt

class F:# implements Function:
  
  def evaluate(self, x):
    return 2*x[0]*x[0] + 1.5*x[0]*x[1] + x[1]*x[1] + 3*x[0] + 4*x[1] + 10
    
  def gradient(self, x):
    return array([4*x[0] + 1.5*x[1] + 3 , 1.5*x[0] + 2*x[1] + 4])
    
  def get_random_pos(self):
    return array([0,0])


def testGD():
  x = F()
  g = GradientDescUpdate(100, 0.5)
  l_error = g.find_local_minimum(x)
  print g.actual_pos
  print l_error[-1]
  plt.plot(range(len(l_error)),l_error)
  plt.show()

def genData(n):
  x1 = random.multivariate_normal([0,0],[[1,0],[0,1]],n)
  x2 = random.multivariate_normal([6,6],[[2,0],[0,1.5]],n)
  y1 = zeros((n,2))
  y2 = zeros((n,2))
  y1[:,0] = ones(n)
  y2[:,1] = ones(n)
  plt.plot(x1[:,0],x1[:,1],'rx')
  plt.plot(x2[:,0],x2[:,1],'bx')
  plt.show()
  X = vstack((x1,x2))
  Y = vstack((y1,y2))
  print Y
  save("datos.npy",(X,Y))


if __name__ == "__main__": 
  genData(100)
