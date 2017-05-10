# -*- python -*-

import subprocess
import math
import numpy
from matplotlib.pylab import *

command = '../../../test/performance/numerical/interpolation/PolynomialInterpolationSine'
outputBase = '../../../doc/numerical/graphics/interpolation/PolynomialInterpolationSine'

ordersAndDegrees = [('1', '0'), ('3', '0'), ('3', '1'), ('5', '0'), ('5', '1'),
                    ('5', '2'),]
output = {}

# Get the interpolation output.
for order, degree in ordersAndDegrees:
    process = subprocess.Popen(command + ' ' + order + ' ' + degree + ' 5 256',
                               stdout=subprocess.PIPE, shell=True)
    x = [float(_x) for _x in process.stdout.readline().split()]
    y = [float(_x) for _x in process.stdout.readline().split()]
    e = [float(_x) for _x in process.stdout.readline().split()]
    output[(order, degree)] = (x, y, e)

# Plot the interpolated values for all methods.
# Plot the sampled function in black.
x = numpy.arange(0, 0.5 * math.pi, math.pi / 1024)
y = numpy.sin(x)
plot(x, y, 'k', label='sin(x)')
# Plot the interpolants in various hues.
for order, degree in ordersAndDegrees:
    x, y, e = output[(order, degree)]
    plot(x, y, label='Order ' + order + ', Degree ' + degree)
# lower right
legend(loc=4)
title('Interpolated Values')
draw()
savefig(outputBase + 'Function.jpg')
clf()

# Plot the errors for all methods.
for order, degree in ordersAndDegrees:
    x, y, e = output[(order, degree)]
    plot(x, e, lw=2, label='Order ' + order + ', Degree ' + degree)
# lower left
legend(loc=3)
title('Error in Interpolants')
draw()
savefig(outputBase + 'Error.jpg')
clf()

# Plot the errors for degree 0.
for order, degree in ordersAndDegrees:
    if degree == '0':
        x, y, e = output[(order, degree)]
        plot(x, e, lw=2, label='Order ' + order)
    else:
        plot([],[], label='')
# lower left
legend(loc=3)
title('Error for Degree 0')
draw()
savefig(outputBase + 'Error_D0.jpg')
clf()

# Plot the errors for degree 1.
for order, degree in ordersAndDegrees:
    if degree == '1':
        x, y, e = output[(order, degree)]
        plot(x, e, lw=2, label='Order ' + order)
    else:
        plot([],[], label='')
# lower left
legend(loc=3)
title('Error for Degree 1')
draw()
savefig(outputBase + 'Error_D1.jpg')
clf()

# Plot the errors quintic interpolation.
for order, degree in ordersAndDegrees:
    if (order, degree) in [('5', '1'), ('5', '2')]:
        x, y, e = output[(order, degree)]
        plot(x, e, lw=2, label='Order ' + order + ', Degree ' + degree)
    else:
        plot([],[], label='')
# upper left
legend(loc=2)
title('Error for Quintic Interpolation')
draw()
savefig(outputBase + 'Error_O5.jpg')
clf()
