"""Calculate the errors."""

import sys, os
sys.path.append('../../../../../../../applications/stochastic/state')
from Histogram import Histogram, histogramDistance

def exitOnError():
    print('''Usage:
python Error.py directory''')
    sys.exit(1)

if len(sys.argv) != 2:
    print('Error! Wrong number of arguments.')
    exitOnError()
directory = sys.argv[1]
# Read the converged solutions for each recorded species.
convergedFile = open('converged.txt', 'r')
converged = Histogram()
converged.read(convergedFile, 1)
# The simulation output.
data = open(directory + '/output.txt', 'r')
size = int(data.readline())
# The file to report the errors.
report = open(directory + '/error.txt', 'w')

# For each test.
histogram = Histogram(converged.size())
for i in range(size):
    # The run time.
    time = float(data.readline())
    # Compute the average error.
    error = 0
    for species in range(10):
        histogram.read(data, 1)
        error += histogramDistance(histogram, converged)
    error /= 10.
    report.write('%s %s\n' % (time, error))
