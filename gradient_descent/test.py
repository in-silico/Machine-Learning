#! /usr/bin/python

from gradient_descent import *
import matplotlib.pyplot as plt

class F:# implements Function:
  
  def evaluate(self, x):
    return 2*x[0]*x[0] + 1.5*x[0]*x[1] + x[1]*x[1] + 3*x[0] + 4*x[1] + 10
    
  def gradient(self, x):
    return array([4*x[0] + 1.5*x[1] + 3 , 1.5*x[0] + 2*x[1] + 4])
    
  def get_random_pos(self):
    return array([0,0])


if __name__ == "__main__": 

  x = F()
  g = GradientDescUpdate(100, 0.5)
  l_error = g.find_local_minimum(x)
  print g.actual_pos
  print l_error[-1]
  plt.plot(range(len(l_error)),l_error)
  plt.show()

