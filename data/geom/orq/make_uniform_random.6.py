#
# Make files containing uniform random points in the unit cube.
#

import random

def print_points(size, filename):
    assert(size >= 1)
    
    # Open the file.
    fout = open(filename, 'w')
    
    # Write the number of vertices and faces.
    fout.write('%i\n' % size)
    fout.write('%i\n' % 0)

    # Write the vertices.
    g = random.Random(42)
    for i in xrange(size):
	fout.write('%f %f %f\n' % (g.random(), g.random(), g.random()))

    fout.close()

    return

        
#print_points(100, "uniform_random.2.ascii")
#print_points(1000, "uniform_random.3.ascii")
#print_points(10000, "uniform_random.4.ascii")
#print_points(100000, "uniform_random.5.ascii")
print_points(1000000, "uniform_random.6.ascii")
#print_points(10000000, "uniform_random.7.ascii")
