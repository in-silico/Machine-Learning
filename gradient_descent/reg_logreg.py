#! /usr/bin/python

from gradient_descent import *
import matplotlib.pyplot as plt

class MAPLogReg:
  
  def __init__(self, X, Y):
    self.k = size(Y,1)
    self.d = size(X,1) 
    self.n = size(X,0)
    self.psigma = 10 #Prior standard deviatiom
  
  def compute_Z(self,W):
    Z = zeros(self.n)
    
    for i in xrange (0, self.n):
      for j in xrange (0, self.k):
        Z[i] += exp( dot(W[j],X[i]) )
  
    return Z

  def compute_P(self, W):
      Z = self.compute_Z(W)
      P = zeros((self.n,self.k))
      for i in xrange(0, self.n):
        for j in xrange(0, self.k):
          P[i][j] = exp(dot(W[j],X[i])) / Z[i]

      return P
  
  def log_likehd(self, W):  
    log_Z = log(self.compute_Z(W))    
    ans = 0
    for i in xrange(0, self.n):
      for j in xrange(0, self.k):
        ans += dot(W[j],X[i]) - log_Z[i]
    return -ans

  def evaluate(self, W):
      logprior = sum(sum(W**2))/(2*self.psigma**2)
      return self.log_likehd(W) + logprior #return log posterior
    
  def gradient(self, W):
    P = self.compute_P(W)
    D = P - Y
    return dot(D.T,X) + W/(self.psigma**2)
    
  def get_random_pos(self):
    return zeros( (self.k, self.d) )


if __name__ == "__main__": 
  (X_1,Y) = load('datos.npy')
  #print Y
  n = size(X_1,0)
  X =  ones((n,1))
  X = hstack((X_1,X))
  #print X,Y
  lr = MAPLogReg(X,Y)
  lr.psigma=2
  
  g = GradientDescUpdate(150, 0.1)
  l_error = g.find_local_minimum(lr)
  print g.actual_pos
  print l_error[-1]
  #plt.plot(range(len(l_error)),l_error)
  #plt.show()
  plt.plot(X[0:n/2,0],X[0:n/2,1],'rx')
  plt.plot(X[n/2:n,0],X[n/2:n,1],'bx')
  x = linspace(-5,5,200)
  w = g.actual_pos[0,:]
  print w
  y = (-w[0]*x - w[2]) / w[1]
  plt.plot(x,y)
  #plt.hold()
  plt.show()

