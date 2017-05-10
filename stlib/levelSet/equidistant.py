# Make a set of equidistant points on the unit sphere.

import math
from random import random

def normalized(x):
    im = 1. / math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    return (x[0]*im, x[1]*im, x[2]*im)

def normalizeAll(p):
    for x in p:
        x = normalized(x)

def difference(x, y):
    return (x[0] - y[0], x[1] - y[1], x[2] - y[2])

def distance(x, y):
    return math.sqrt((x[0] - y[0]) * (x[0] - y[0]) +
                     (x[1] - y[1]) * (x[1] - y[1]) +
                     (x[2] - y[2]) * (x[2] - y[2]))


def push(p):
    dt = 0.001
    for i in range(len(p)):
        x = p[i]
        f = (0., 0., 0.)
        for y in p:
            if x is y:
                continue
            v = difference(x, y)
            m = distance(x, y)
            v = normalized(v)
            f = (f[0] + dt * v[0] / m,
                 f[1] + dt * v[1] / m,
                 f[2] + dt * v[2] / m)
        p[i] = normalized((x[0] + f[0], x[1] + f[1], x[2] + f[2]))
    
points = [(random(), random(), random()) for i in range(64)]
normalizeAll(points)


def randomUnitVector():
    x = (random(), random(), random())
    return normalized(x)

for i in range(2000):
    push(points)

maxDistance = 0
for i in range(10000):
    x = randomUnitVector()
    d = 1e10
    for p in points:
        d = min(distance(x, p), d)
    maxDistance = max(maxDistance, d)
print(maxDistance)

for x in points:
    print('{%sf, %sf, %sf},' % x)
