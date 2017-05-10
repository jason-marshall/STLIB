"""Calculate the errors."""

import sys, os
sys.path.append('../../../../../applications/stochastic/state')
from Histogram import Histogram, histogramDistance

def exitOnError():
    print('''Usage:
python Error.py directory''')
    sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Error! Wrong number of arguments.')
        exitOnError()
    directory = sys.argv[1]
    converged = Histogram()
    converged.read(open('converged.txt', 'r'))
    histogram = Histogram(converged.size())
    lines = open(directory + '/output.txt', 'r').readlines()
    size = len(lines)
    report = open(directory + '/error.txt', 'w')
    report.write('#time %s\n' % directory)
    # Determine the samples.
    samples = []
    sample = size
    while sample > 0:
        samples.append(sample)
        sample //= 2
    samples.sort()
    # For each sample size.
    for sample in samples:
        error = 0.
        time = 0.
        for start in range(0, size, sample):
            histogram.clear()
            if start + sample > size:
                break
            for i in range(start, start + sample):
                data = [float(_x) for _x in lines[i].split()]
                time += data[0]
                for i in range(1, len(data), 2):
                    histogram.accumulate(data[i], data[i+1], 0)
            error += histogramDistance(histogram, converged)
        time /= size//sample
        error /= size//sample
        report.write('%s %s\n' % (time, error))
