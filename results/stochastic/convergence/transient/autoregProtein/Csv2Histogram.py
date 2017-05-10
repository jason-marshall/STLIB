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
    lines = open(sys.argv[1], 'r').readlines()[3:]
    while not lines[-1].strip():
        del lines[-1]
    lowerBound = float(lines[0].split(',')[0])
    width = float(lines[1].split(',')[0]) - lowerBound
    bins = [float(_x.split(',')[1]) for _x in lines]
    file = open(sys.argv[2], 'w')
    file.write('%s\n%s\n' % (lowerBound, width))
    file.write(' '.join([str(_x) for _x in bins]) + '\n')
    file.write('0 ' * len(bins) + '\n')
    
