# star.py
# Make the b-rep for a 8 pointed star.

import math

n = 2 * 256

# Space dimension, simplex dimension.
print "2 1"

# Number of vertices.
print n

# The vertices.
for i in range(n):
    r = 0.1 + 0.4 * ((i + 1) % 2)
    x = r * math.cos(i * 2 * math.pi / n)
    y = r * math.sin(i * 2 * math.pi / n)
    print x, y

# Number of edges.
print n

# The vertices.
for i in range(n):
    print i, (i + 1) % n
    
