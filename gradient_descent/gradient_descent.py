#! /usr/bin/python

from numpy import *

'''
interface Function:
  "Test comment"
   
  def get_random_pos(self):
    ""
  
  def evaluate(self, x):
    "Evaluate the function in an specific value"
    
  def gradient(self, x):
    "Evaluate the gradient of function in an specific value"
    
'''    

class GradientDescent:
  "Class to find a local minimum of a function using gradient descent"
  
  
  def __init__(self,num_iter, step):
    self.num_iter = num_iter
    self.step = step
  
  def find_local_minimum(self,function):
    
    self.actual_pos = function.get_random_pos()
    l_error = []
    
    for i in xrange(0, self.num_iter): 
      g_error = function.evaluate(self.actual_pos)
      l_error.append(g_error)
      self.actual_pos = self.actual_pos - self.step*function.gradient(self.actual_pos)
    
    return l_error
    
class GradientDescUpdate:
  "Class to find a local minimum of a function using gradient descent"
    
  def __init__(self,num_iter, step, eps=1e-5):
    self.num_iter = num_iter
    self.step = step
    self.eps = eps
  
  def find_local_minimum(self,function):
    
    self.actual_pos = function.get_random_pos()
    l_error = []
    
    for i in xrange(0, self.num_iter): 
      g_error = function.evaluate(self.actual_pos)
      l_error.append(g_error)
      if ( i>0 and abs(l_error[-1] - l_error[-2]) <= self.eps):
        break
      if (i>0 and l_error[-1] >= l_error[-2]):
        self.step /= 2.0
      self.actual_pos = self.actual_pos - self.step*function.gradient(self.actual_pos)
    
    return l_error
