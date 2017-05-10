# ignore2.py

import sys, string

if len( sys.argv ) != 3:
    print "Usage:"
    print "python ignore2.py in out\n"
    raise "Bad command line arguments.  Exiting..."

file = open( sys.argv[1], "r" )
lines = file.readlines()
file.close()

file = open( sys.argv[2], "w" )
for line in lines:
    rest = string.split( line )[2:]
    for s in rest:
        file.write( s + " " )
    file.write( "\n" )
file.close()

