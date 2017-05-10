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
    # Read the converged solutions for each recorded species.
    convergedFile = open('converged.txt', 'r')
    numberOfHistograms = int(convergedFile.readline())
    converged = [Histogram() for _i in range(numberOfHistograms)]
    for h in converged:
        h.read(convergedFile)
    # The simulation output.
    lines = open(directory + '/output.txt', 'r').readlines()
    size = len(lines) // (numberOfHistograms+1)
    # The file to report the errors.
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
    histogram = Histogram(converged[0].size())
    for sample in samples:
        # The average run time.
        time = 0.
        for start in range(0, size, sample):
            if start + sample > size:
                break
            for i in range(start, start + sample):
                time += float(lines[i*(numberOfHistograms+1)])
        time /= size//sample
        # Compute the average error.
        averageError = 0.
        # For each recorded species.
        for h in range(numberOfHistograms):
            error = 0.
            for start in range(0, size, sample):
                histogram.clear()
                if start + sample > size:
                    break
                for i in range(start, start + sample):
                    n = i*(numberOfHistograms+1)+h+1
                    data = [float(_x) for _x in lines[n].split()]
                    for j in range(0, len(data), 2):
                        histogram.accumulate(data[j], data[j+1], 0)
                error += histogramDistance(histogram, converged[h])
            error /= size//sample
            averageError += error
        averageError /= numberOfHistograms
        report.write('%s %s\n' % (time, averageError))
