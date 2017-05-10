# circle.py
# Make the b-rep for a circle.

import math

# I chose this so the edge length is about equal to the grid spacing.
n = 1570

# Number of vertices.
print n

# Number of edges.
print n

# The vertices.
for i in range( n ):
    x = 0.5 + 0.25 * math.cos( i * 2 * math.pi / n )
    y = 0.5 + 0.25 * math.sin( i * 2 * math.pi / n )
    print x, y

# The vertices.
for i in range( n ):
    print i, (i + 1) % n
    
