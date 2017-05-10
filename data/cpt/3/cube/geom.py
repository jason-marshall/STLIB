import sys

if len(sys.argv) < 4:
    print('Bad command line arguments.')
    print('Usage:')
    print('%s gridsPerDimension gridSize distance ["(x0,y0,z0,x1,y1,z1)"]' %
          sys.argv[0])
    sys.exit(1)

# The program name.
del sys.argv[0]

# Number of grids in each dimension.
gridsPerDimension = int(sys.argv[0])
del sys.argv[0]
assert gridsPerDimension > 0

# The grid size (along a single dimension).
gridSize = int(sys.argv[0])
del sys.argv[0]
assert gridSize > 0

# How far to compute the distance.
distance = float(sys.argv[0])
del sys.argv[0]
assert distance > 0

# The Cartesian domain of the lattice.
if sys.argv:
    domain = eval(sys.argv[0])
    del sys.argv[0]
    assert len(domain) == 6
else:
    domain = (-1, -1, -1, 2, 2, 2)

latticeExtent = gridsPerDimension * gridSize

# How far to compute the distance.
print(distance)
# The Cartesian domain of the lattice.
print('%f %f %f %f %f %f' % domain)
# The lattice extents.
print('%d %d %d' % (latticeExtent, latticeExtent, latticeExtent))
# The number of grids.
print('%d' % gridsPerDimension**3)
# Print the extents of each grid.
for k in range(0, latticeExtent, gridSize):
    for j in range(0, latticeExtent, gridSize):
        for i in range(0, latticeExtent, gridSize):
            print('%d %d %d %d %d %d' % (gridSize, gridSize, gridSize, i, j, k))

