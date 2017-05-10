# AutoRegulatory.py
# Usage:
# python AutoRegulatory.py multiplicity
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
  s += '%d\n' % (5 * multiplicity)
  # <number of reactions>
  s += '%d\n' % (8 * multiplicity)
  # <list of initial amounts>
  # Start with a gene population of 1. If all of the gene populations are zero,
  # the initial sum of the propensities is much smaller than typical.
  s += '10 0 1 0 0 ' * multiplicity + '\n'
  # <packed reactions>
  for n in range(multiplicity):
    gene = 5 * n
    p2Gene = gene + 1
    rna = gene + 2
    p = gene + 3
    p2 = gene + 4
    # gene + p2 -> p2Gene
    s += '2 ' + str(gene) + ' 1 ' + str(p2) + ' 1 1 ' + str(p2Gene) + ' 1 '
    # p2Gene -> gene + p2
    s += '1 ' + str(p2Gene) + ' 1 2 ' + str(gene) + ' 1 ' + str(p2) + ' 1 '
    # gene -> gene + rna
    s += '1 ' + str(gene) + ' 1 2 ' + str(gene) + ' 1 ' + str(rna) + ' 1 '
    # rna -> rna + p
    s += '1 ' + str(rna) + ' 1 2 ' + str(rna) + ' 1 ' + str(p) + ' 1 '
    # 2 p -> p2
    s += '1 ' + str(p) + ' 2 1 ' + str(p2) + ' 1 '
    # p2 -> 2 p
    s += '1 ' + str(p2) + ' 1 1 ' + str(p) + ' 2 '
    # rna ->
    s += '1 ' + str(rna) + ' 1 0 '
    # p ->
    s += '1 ' + str(p) + ' 1 0 '
  s += '\n'
  # <list of propensity factors>
  #s += '1 10 0.01 10 1 1 0.1 0.01 ' * multiplicity + '\n'
  random.seed(1)
  for i in range(multiplicity):
    r = random.random()
    s += '%s %s %s %s %s %s %s %s ' % (r, 10*r, 0.01*r, 10*r, r, r, 0.1*r, 
                                       0.01*r)
  return s

if __name__ == '__main__':
  errorMessage = \
  """Usage:
  python AutoRegulatory.py multiplicity"""

  if len(sys.argv) != 2:
    print errorMessage
    raise "Wrong number of command line arguments.  Exiting..."

  # Get the multiplicity.
  multiplicity = int(sys.argv[1])
  assert multiplicity > 0

  print output(multiplicity)
