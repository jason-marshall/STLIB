# gp_file.py
# Make a gnuplot file.
# Usage:
# python gp_file.py name

import sys

if len( sys.argv ) != 2:
    print "Usage:"
    print "python gp_file.py name\n"
    raise "Bad command line arguments.  Exiting..."

name = sys.argv[1]
file = open( name + ".gnu", "w" )

file.write( "set size ratio 1 1, 1\n" )
file.write( "set noborder\n" )
file.write( "set noxtics\n" )
file.write( "set noytics\n" )
file.write( "set nokey\n" )
file.write( "set terminal postscript eps 22 color\n" )
file.write( "set output \"%s.eps\"\n" % name )
file.write( "plot '%s.dat' with lines linewidth 2, 'boundary.dat' with lines linewidth 3\n" % name )
