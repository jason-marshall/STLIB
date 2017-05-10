# Script to assess the accuracy of simple methods of computing content using
# a signed distance function.

import math
import numpy

def contentBinary(array, dx):
    # Flatten the array.
    length = 1
    for extent in array.shape:
        length *= extent
    flat = numpy.reshape(array, (length,))
    # The content of a voxel.
    voxel = dx**len(array.shape)
    # Compute the content.
    content = 0.
    for x in flat:
        if x <= 0:
            content += voxel
    return content

def contentPartial(array, dx):
    # Only implemented for 2-D arrays for now.
    assert len(array.shape) == 2
    # Flatten the array.
    length = 1
    for extent in array.shape:
        length *= extent
    flat = numpy.reshape(array, (length,))
    # The content of a voxel.
    voxel = dx**len(array.shape)
    # Compute the content.
    t = dx / math.sqrt(math.pi)
    # y = a x + b
    a = - 0.5 * math.sqrt(math.pi) * dx
    b = 0.5 * voxel
    content = 0.
    for x in flat:
        if x <= - t:
            content += voxel
        elif x < t:
            content += a * x + b
    return content

def contentPartialNegative(array, dx):
    # Only implemented for 2-D arrays for now.
    assert len(array.shape) == 2
    # Flatten the array.
    length = 1
    for extent in array.shape:
        length *= extent
    flat = numpy.reshape(array, (length,))
    # The content of a voxel.
    voxel = dx**len(array.shape)
    # Compute the content.
    t = dx / math.sqrt(math.pi)
    # y = a x
    a = - math.sqrt(math.pi) * dx
    content = 0.
    for x in flat:
        if x <= - t:
            content += voxel
        elif x < 0:
            content += a * x
    return content

def ballDistance(x, y):
    """Distance to the unit ball."""
    return math.sqrt(x * x + y * y) - 1.

# Try a couple of different extents, and hence spacings.
for extent in [10, 20, 40]:
    # Define a 2-D grid.
    halfRadius = 1.5
    lower = (-halfRadius, -halfRadius)
    upper = (halfRadius, halfRadius)
    dx = (upper[0] - lower[0]) / (extent - 1)
    print('\ndx = %s' % dx)
    ball = numpy.zeros((extent, extent))
    for i in range(ball.shape[0]):
        x = lower[0] + dx * i
        for j in range(ball.shape[1]):
            y = lower[1] + dx * j
            ball[i, j] = ballDistance(x, y)

    for f in [contentBinary, contentPartial, contentPartialNegative]:
        content = f(ball, dx)
        error = (content - math.pi) / math.pi
        print('%s = %.5s, error = %s, e/dx = %s' %
              (f.__name__, content, error, error / dx))
