import sys

if len(sys.argv) < 4:
    print 'Bad command line arguments.'
    print 'Usage:'
    print '%s "(g0,g1,g2)" "(s0,s1,s2)" distance ["(x0,y0,z0,x1,y1,z1)"]' \
	% sys.argv[0]
    print '(g0,g1,g2) specify the number of grids in each dimension.'
    print '(s0,s1,s2) specifies the size of a single grid.'
    raise StandardError

# The program name.
del sys.argv[0]

# Number of grids in each dimension.
gridsPerDimension = eval(sys.argv[0])
del sys.argv[0]
assert len(gridsPerDimension) == 3
for d in range(3):
    assert gridsPerDimension[d] > 0

# The grid size (along a single dimension).
gridSize = eval(sys.argv[0])
del sys.argv[0]
assert len(gridSize) == 3
for d in range(3):
    assert gridSize[d] > 0

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

latticeExtents = (gridsPerDimension[0] * gridSize[0],
                  gridsPerDimension[1] * gridSize[1],
                  gridsPerDimension[2] * gridSize[2])

# How far to compute the distance.
print distance
# The Cartesian domain of the lattice.
print '%f %f %f %f %f %f' % domain
# The lattice extents.
print '%d %d %d' % latticeExtents
# The number of grids.
print '%d' % (gridsPerDimension[0] * gridsPerDimension[1] * 
              gridsPerDimension[2])
# Print the extents of each grid.
for k in range(0, latticeExtents[2], gridSize[2]):
    for j in range(0, latticeExtents[1], gridSize[1]):
	for i in range(0, latticeExtents[0], gridSize[0]):
            print i, j, k, i + gridSize[0], j + gridSize[1], k + gridSize[2]

