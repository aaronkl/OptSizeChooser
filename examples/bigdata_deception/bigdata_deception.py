import numpy as np
import sys
import math
import time

def bigdata_deception(x):
  s = x[0]
  x = x[1]
  return 1-x*s+x/2

# Write a function like this called 'main'
def main(job_id, params):
  return bigdata_deception(params['X'])
