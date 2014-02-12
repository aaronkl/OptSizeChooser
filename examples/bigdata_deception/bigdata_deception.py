import numpy as np
import sys
import math
import time

eFive = np.e**5

def bigdata_deception(s, x):
  s = s[0]
  x = x[0]
  return 5-x*np.log(s)+x/2

# Write a function like this called 'main'
def main(job_id, params):
  print "evaluating " + str(params['SIZE']) + str(params['X'])
  return bigdata_deception(params['SIZE'], params['X'])
