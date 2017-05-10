# rectangle.py
# Make a rectangle mesh.
# Usage:
# python rectangle.py x0 y0 x1 y1 num_points_per_side mesh

import sys, string, math

if len( sys.argv ) != 7:
    print "Usage:"
    print "python rectangle.py x0 y0 x1 y1 num_points_per_side mesh\n"
    raise "Bad command line arguments.  Exiting..."


# The corners.
x0 = string.atof( sys.argv[1] )
y0 = string.atof( sys.argv[2] )
x1 = string.atof( sys.argv[3] )
y1 = string.atof( sys.argv[4] )

# The number of points per side.
num_points_per_side = string.atoi( sys.argv[5] )
if ( num_points_per_side < 1 ):
  raise( "Bad number of points." )

num_points = 4 * num_points_per_side

# The mesh file.
file = open( sys.argv[6], "w" )
file.write( "2 1\n" )
file.write( "%d\n" % num_points )
# South side.
for n in range( num_points_per_side ):
  x = x0 + n * (x1 - x0) / num_points_per_side
  y = y0
  file.write( "%g %g\n" % ( x, y ) )
# East side.
for n in range( num_points_per_side ):
  x = x1
  y = y0 + n * (y1 - y0) / num_points_per_side
  file.write( "%g %g\n" % ( x, y ) )
# North side.
for n in range( num_points_per_side ):
  x = x1 - n * (x1 - x0) / num_points_per_side
  y = y1
  file.write( "%g %g\n" % ( x, y ) )
# West side.
for n in range( num_points_per_side ):
  x = x0
  y = y1 - n * (y1 - y0) / num_points_per_side
  file.write( "%g %g\n" % ( x, y ) )

file.write( "%d\n" % num_points )
for n in range( num_points ):
  file.write( "%d %d\n" % ( n, (n+1) % num_points ) )
file.close()
