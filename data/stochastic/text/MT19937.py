# MT19937.py
# Usage:
# python MT19937.py
#
# Generate a state for the Mersenne twister.

import sys
from random import randint

def output():
  s = ''
  # The array.
  for n in range(624):
    s += '%d ' % randint(0, 2**32 - 1)
  # The position.
  s += ' 625\n'
  return s

if __name__ == '__main__':
  errorMessage = \
  """Usage:
  python MT19937.py"""

  if len(sys.argv) != 1:
    print errorMessage
    raise "Wrong number of command line arguments.  Exiting..."

  print output()
