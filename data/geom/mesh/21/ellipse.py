# ellipse.py
# Make a ellipse mesh.
# Usage:
# python ellipse.py x y x_radius y_radius num_points mesh

import sys, string, math

if len( sys.argv ) != 7:
    print "Usage:"
    print "python ellipse.py x y x_radius y_radius num_points mesh\n"
    raise "Bad command line arguments.  Exiting..."


# The center.
x = string.atof( sys.argv[1] )
y = string.atof( sys.argv[2] )

# The radii.
x_radius = string.atof( sys.argv[3] )
y_radius = string.atof( sys.argv[4] )

# The number of points.
num_points = string.atoi( sys.argv[5] )
if ( num_points < 1 ):
  raise( "Bad number of points." )

# The mesh file.
file = open( sys.argv[6], "w" )
file.write( "2 1\n" )
file.write( "%d\n" % num_points )
for n in range( num_points ):
  file.write( "%g %g\n" % ( x + x_radius * math.cos( n * (2 * math.pi) / 
                                                     num_points ),
                            y + y_radius * math.sin( n * (2 * math.pi) / 
                                                     num_points ) ) )
file.write( "%d\n" % num_points )
for n in range( num_points ):
  file.write( "%d %d\n" % ( n, (n+1) % num_points ) )
file.close()
