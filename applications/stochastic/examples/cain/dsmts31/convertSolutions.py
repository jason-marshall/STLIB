# Convert the solutions from the DSMTS 3.1 directory to a format that one
# may import in Cain.

from glob import glob

for meanName in glob('../../sbml/dsmts31/*-mean.csv'):
    stdDevName = meanName.replace('mean', 'sd')
    outputName = meanName.replace('../../sbml/dsmts31/', '').\
                 replace('-mean.csv', '.txt')
    # Open the input files and skip the header line.
    mean = open(meanName, 'r')
    mean.readline()
    stdDev = open(stdDevName, 'r')
    stdDev.readline()
    # Open the output file.
    output = open(outputName, 'w')
    for meanLine in mean:
        # Ignore trailing blank lines.
        meanLine = meanLine.rstrip()
        if not meanLine:
            break
        stdDevLine = stdDev.readline().rstrip()
        m = meanLine.split(',')
        s = stdDevLine.split(',')
        # Skip the time field.
        for i in range(1, len(m)):
            output.write('%s %s ' % (m[i], s[i]))
        output.write('\n')
    mean.close()
    stdDev.close()
    output.close()


