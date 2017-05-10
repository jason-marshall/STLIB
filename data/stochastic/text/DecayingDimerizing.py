# DecayingDimerizing.py
# Usage:
# python DecayingDimerizing.py multiplicity
#
# The output is
# <number of species>
# <number of reactions>
# <list of initial amounts>
# <packed reactions>
# <list of propensity factors>

import sys
import random

def output(multiplicity):
  s = ''
  # <number of species>
  s += '%d\n' % (3 * multiplicity)
  # <number of reactions>
  s += '%d\n' % (4 * multiplicity)
  # <list of initial amounts>
  s += '100000 0 0 ' * multiplicity + '\n'
  # <packed reactions>
  for n in range(multiplicity):
    i = 3 * n
    j = i + 1
    k = j + 1
    # S_i -> 0
    s += '1 ' + str(i) + ' 1 0 '
    # 2 S_i -> S_j
    s += '1 ' + str(i) + ' 2 1 ' + str(j) + ' 1 '
    # S_j -> 2 S_i
    s += '1 ' + str(j) + ' 1 1 ' + str(i) + ' 2 '
    # S_j -> S_k
    s += '1 ' + str(j) + ' 1 1 ' + str(k) + ' 1 '
  s += '\n'
  # <list of propensity factors>
  #s += '1 0.002 0.5 0.04 ' * multiplicity + '\n'
  random.seed(1)
  for i in range(multiplicity):
    r = random.random()
    s += '%s %s %s %s ' % (r, 0.002*r, 0.5*r, 0.04*r)
  return s

if __name__ == '__main__':
  errorMessage = \
  """Usage:
  python DecayingDimerizing.py multiplicity"""

  if len(sys.argv) != 2:
    print errorMessage
    raise "Wrong number of command line arguments.  Exiting..."

  # Get the multiplicity.
  multiplicity = int(sys.argv[1])
  assert multiplicity > 0

  print output(multiplicity)
