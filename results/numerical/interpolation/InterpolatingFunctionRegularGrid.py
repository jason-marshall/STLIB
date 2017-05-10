# -*- python -*-

import subprocess

file = open('../../../doc/numerical/tables/InterpolatingFunctionRegularGrid.txt', 'w')

file.write('<table border = "1" rules = "all">\n')
file.write('<tr> <th> Dimension <th> Order <th> Time\n')

arguments = ['4096', '64 64', '16 16 16', '8 8 8 8']

for dimension, order, suffix in \
        [(1, 'Linear', 'Linear'), (2, 'Linear', 'Linear'),
         (3, 'Linear', 'Linear'), (4, 'Linear', 'Linear'),
         (1, 'Cubic', 'Cubic'), (2, 'Cubic', 'Cubic'), 
         (1, 'Cubic with Derivative', 'CubicDerivative'),
         (2, 'Cubic with Derivative', 'CubicDerivative')]:
    file.write('<tr> <td> ' + str(dimension) + ' <td> ' + order + ' ')
    command = '../../../test/performance/numerical/interpolation/InterpolatingFunctionRegularGrid' + str(dimension) + 'D' + suffix
    process = subprocess.Popen(command + ' ' + arguments[dimension-1],
                               stdout=subprocess.PIPE, shell=True)
    time = float(process.stdout.readlines()[-1])
    file.write('<td> ' + '%.1f' % time + '\n')
file.write('</table>\n')
