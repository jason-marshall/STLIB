# gp_file.py
# Make a gnuplot file.
# Usage:
# python gp_file.py number

import sys

if len( sys.argv ) != 2:
    print "Usage:"
    print "python gp_file.py number\n"
    raise "Bad command line arguments.  Exiting..."

number = sys.argv[1]
file = open( "mesh.%s.gnu" % number, "w" )

file.write( "set size ratio 1 1, 1\n" )
file.write( "set noborder\n" )
file.write( "set noxtics\n" )
file.write( "set noytics\n" )
file.write( "set nokey\n" )
file.write( "set terminal postscript eps 22 color\n" )
file.write( "set output \"mesh.%s.eps\"\n" % number )
file.write( "plot 'mesh.%s.dat' with lines linewidth 2, 'boundary.%s.dat' with lines linewidth 3\n" % ( number, number ) )
