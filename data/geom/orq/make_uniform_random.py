#
# Make files containing uniform random points in the unit cube.
#

import random

def print_points(size, filename):
    assert(size >= 1)
    
    # Open the file.
    fout = open(filename, 'w')

    # Space dimension and simplex dimension.
    fout.write('3 3\n')
    # Write the number of vertices.
    fout.write('%i\n' % size)
    # Write the vertices.
    g = random.Random(42)
    for i in xrange(size):
	fout.write('%f %f %f\n' % (g.random(), g.random(), g.random()))
    # Write the number of faces.
    fout.write('%i\n' % 0)
    fout.close()

    return

        
print_points(100, "uniform_random.2.ascii")
print_points(1000, "uniform_random.3.ascii")
print_points(10000, "uniform_random.4.ascii")
print_points(100000, "uniform_random.5.ascii")
print_points(1000000, "uniform_random.6.ascii")
#print_points(10000000, "uniform_random.7.ascii")
