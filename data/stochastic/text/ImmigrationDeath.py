# ImmigrationDeath.py
# Usage:
# python ImmigrationDeath.py size immigrationPropensityFunction equilibriumPopulation
# - size is the number of species.
#
# The output is
# <number of species>
# <number of reactions>
# <list of initial amounts>
# <packed reactions>
# <list of propensity factors>

import sys
# For the propensity function.
from math import *
from random import *

def output(size, f, equilibriumPopulation):
  s = ''
  # <number of species>
  s += '%d\n' % size
  # <number of reactions>
  s += '%d\n' % (2*size)
  # <list of initial amounts>
  s += (str(equilibriumPopulation) + ' ') * size + '\n'
  # <packed reactions>
  for n in range(size):
    s += '0 1 ' + str(n) + ' 1 ' + '1 ' + str(n) + ' 1 0 '
  s += '\n'
  # <list of propensity factors>
  for n in range(size):
    p = float(f(n))
    s += str(p) + ' ' + str(p / equilibriumPopulation) + ' '
  s += '\n'
  return s

if __name__ == '__main__':
  errorMessage = \
  """Usage:
  python ImmigrationDeath.py size immigrationPropensityFunction equilibriumPopulation
  - size is the number of species."""

  if len(sys.argv) != 4:
    print errorMessage
    raise AssertionError, "Wrong number of command line arguments.  Exiting..."

  # 
  # Parse the arguments.
  #

  # Get the number of species.
  size = int(sys.argv[1])
  assert size > 0
  # Get the propensity function
  f = lambda n : eval(sys.argv[2])
  # The equilibrium population.
  equilibriumPopulation = int(sys.argv[3])

  # Seed the random number generator so we get consistent results.
  seed(1)
  
  # 
  # Write the model.
  #
  sys.stdout.write(output(size, f, equilibriumPopulation))
