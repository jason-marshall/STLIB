# transform.py

import math

num = 200

t = 2 * math.pi / (num + 1 )
for n in range( num ):
  file = open( "transform.%03d.txt" % n, "w" )
  file.write( "%g * x + %g * y + %g\n" % ( math.cos( 2 * n * t ), 
                                           -math.sin( 2 * n * t ),
                                           0.25 * math.cos( n * t ) ) )
  file.write( "%g * x + %g * y + %g\n" % ( math.sin( 2 * n * t ), 
                                           math.cos( 2 * n * t ),
                                           0.25 * math.sin( n * t ) ) )
  file.close()
                                         