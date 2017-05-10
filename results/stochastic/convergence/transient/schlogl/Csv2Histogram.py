"""Generate trajectories for each solver."""

import sys

def exitOnError():
    print('''Usage:
python Csv2Histogram.py csv histogram''')
    sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Error! Wrong number of arguments.')
        exitOnError()
    numberOfHistograms = 1
    numberOfBins = 1000
    lines = open(sys.argv[1], 'r').readlines()
    file = open(sys.argv[2], 'w')
    file.write('%s\n' % numberOfHistograms)
    n = 3
    for i in range(numberOfHistograms):
        lowerBound = float(lines[n].split(',')[0])
        width = float(lines[n+1].split(',')[0]) - lowerBound
        bins = [float(_x.split(',')[1]) for _x in lines[n:n+numberOfBins]]
        file.write('%s\n%s\n' % (lowerBound, width))
        file.write(' '.join([str(_x) for _x in bins]) + '\n')
        file.write('0 ' * len(bins) + '\n')
        # Add two for the blank line and the header.
        n += numberOfBins + 2
    
