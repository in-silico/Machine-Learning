#! /usr/bin/python

from gradient_descent import *
import matplotlib.pyplot as plt

class Logistic_regression:
  
  def __init__(self, X, Y):
    self.k = size(Y,1)
    self.d = size(X,1) 
    self.n = size(X,0)
  
  def compute_Z(W):
    Z = zeros(n)
    
    for i in xrange (0, n):
      for j in xrange (0, k):
        Z[i] += exp( dot(W[j],X[i]) )
  
    return Z
  
  def evaluate(self, W):
  
    log_Z = log(compute_Z(W))
    
    ans = 0
    for i in xrange(0, n):
      for j in xrange(0, k):
        ans += dot(W[j],X[i]) - log_Z[i]
      
    return -ans
    
    
  def gradient(self, W):
    
    P = compute_P(W)
    D = self.Y - P
    
    return dot(D.T,X)
    
    
  def get_random_pos(self):
    return zeros( (self.k, self.d) )


if __name__ == "__main__": 

  x = Logistic_regression()
  
  g = GradientDescUpdate(100, 0.5)
  l_error = g.find_local_minimum(x)
  print g.actual_pos
  print l_error[-1]
  plt.plot(range(len(l_error)),l_error)
  plt.show()

