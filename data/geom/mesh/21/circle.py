# circle.py
# Make a circle mesh.
# Usage:
# python circle.py x y radius num_points mesh

import sys, string, math

if len( sys.argv ) != 6:
    print "Usage:"
    print "python circle.py x y radius num_points mesh\n"
    raise "Bad command line arguments.  Exiting..."


# The center.
x = string.atof( sys.argv[1] )
y = string.atof( sys.argv[2] )

# The radius.
radius = string.atof( sys.argv[3] )

# The number of points.
num_points = string.atoi( sys.argv[4] )
if ( num_points < 1 ):
  raise( "Bad number of points." )

# The mesh file.
file = open( sys.argv[5], "w" )
file.write( "2 1\n" )
file.write( "%d\n" % num_points )
for n in range( num_points ):
  file.write( "%g %g\n" % ( x + radius * math.cos( n * (2 * math.pi) / 
                                                   num_points ),
                            y + radius * math.sin( n * (2 * math.pi) / 
                                                   num_points ) ) )
file.write( "%d\n" % num_points )
for n in range( num_points ):
  file.write( "%d %d\n" % ( n, (n+1) % num_points ) )
file.close()
