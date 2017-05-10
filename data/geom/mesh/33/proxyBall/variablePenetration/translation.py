import sys
# The number of cores.
n = float(sys.argv[1])
# First move the projectile to be in contact with the plate. Then penetrate
# by a tenth of the radius times a factor that is proportional to the 
# approximate edge length.
offset = 9.23077e-07 + 0.000889 * 0.1 / n**(1./3)
print('x')
print('y')
print('z - ' + str(offset))
