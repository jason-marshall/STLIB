# -*- python -*-

import subprocess

file = open('../../../doc/numerical/tables/InterpolatingFunction1DRegularGrid.txt', 'w')

sizes = ['10', '100', '1000', '10000', '100000', '1000000']

file.write('<table border = "1" rules = "all">\n')
file.write('<tr> <th> Grid Size '
           + ''.join(['<th> ' + x + ' ' for x in sizes]) + '\n')

for suffix, name in [('Linear', 'Linear'), ('Cubic', 'Cubic'),
                     ('CubicDerivative', 'Cubic with Derivative')]:
    file.write('<tr> <th> ' + name + ' ')
    command = '../../../test/performance/numerical/interpolation/InterpolatingFunction1DRegularGrid' + suffix
    for size in sizes:
        process = subprocess.Popen(command + ' ' + size, stdout=subprocess.PIPE,
                                   shell=True)
        time = float(process.stdout.readlines()[-1])
        file.write('<td> ' + '%.1f' % time + ' ')
    file.write('\n')
file.write('</table>\n')
