#
# Make files containing uniform random points on a unit sphere.
#

import random
import math

def print_points(size, filename):
    assert(size >= 1)
    
    # Open the file.
    fout = open(filename, 'w')
    
    # Write the number of vertices and faces.
    fout.write('%i\n' % size)
    fout.write('%i\n' % 0)

    # Write the vertices.
    g = random.Random(42)
    i = 0
    while i < size:
        x, y, z = 2 * g.random() - 1, 2 * g.random() - 1, 2 * g.random() - 1
        a = x*x + y*y + z*z
        if a > 1e-6 and a <= 1:
            a = math.sqrt(a)
            fout.write('%f %f %f\n' % (x / a, y / a, z / a))
            i += 1

    fout.close()

    return

print_points(1000000, "sphere.6.ascii")
