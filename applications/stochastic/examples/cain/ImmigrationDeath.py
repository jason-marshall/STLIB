# Write the reference solution for the immigration-death process.
# 0 -> X, k1 = 1
# X -> 0, k2 = 0.1

from math import *

k1 = 1.
k2 = 0.1
x0 = 0.
t0 = 0.

def mean(t):
    return k1 / k2 + (x0 - k1 / k2) * exp(-k2 * (t - t0))

def var(t):
    return (k1 / k2) * (1 - exp(-k2 * (t - t0))) *\
           (1 + (k2 * x0 / k1) * exp(-k2 * (t - t0)))

f = open('ImmigrationDeathTransient.txt', 'w')
for t in range(51):
    f.write(repr(mean(t)) + ' ' + repr(sqrt(var(t))) + '\n')
f.close()

f = open('ImmigrationDeathSteadyState.txt', 'w')
f.write(repr(k2 / k1) + ' ' + repr(sqrt(k2 / k1)) + '\n')
f.close()
