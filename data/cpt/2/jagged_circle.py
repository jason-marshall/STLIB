# jagged_circle.py
# Make the b-rep for a jagged circle.

import math
import random

n = 100

# Space dimension, simplex dimension.
print "2 1"

# Number of vertices.
print n

# The vertices.
for i in range( n ):
    r = 0.25 + random.uniform( -0.1, 0.1 )
    x = 0.5 + r * math.cos( i * 2 * math.pi / n )
    y = 0.5 + r * math.sin( i * 2 * math.pi / n )
    print x, y

# Number of edges.
print n

# The vertices.
for i in range( n ):
    print i, (i + 1) % n
    
