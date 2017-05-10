# iss2exodus.py
# Convert an indexed simplex set to an ascii exodus file.
# Usage:
# python iss2exodus.py in out
#
# One can convert the ascii exodus file to a binary file with ncgen.

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "python iss2exodus.py in out\n"
    print "This program converts an indexed simplex set to an ascii exodus file."
    raise "Bad command line arguments.  Exiting..."


# Open the input file.
in_file = open(sys.argv[1], "r")

# The space dimension and simplex dimension.
number_strings = string.split(in_file.readline())
if len(number_strings) != 2:
    raise "Error reading the space dimension and simplex dimension."
if eval(number_strings[0]) != 3:
    raise("Error: Bad space dimension.")
if eval(number_strings[1]) != 3:
    raise("Error: Bad simplex dimension.")


#
# The vertices.
#
num_vertices = eval(in_file.readline())
print "%d vertices." % num_vertices
print "Reading the vertices..."
vertices = []
for n in range(num_vertices):
    vertices.append(in_file.readline())
print "Done."


#
# The indexed simplices.
#
num_simplices = eval(in_file.readline())
print "%d simplices." % num_simplices
print "Reading the simplices..."
simplices = []
for n in range(num_simplices):
    number_strings = string.split(in_file.readline())
    # Convert from C to fortran indices here.
    simplices.append((string.atoi(number_strings[0]) + 1,
                        string.atoi(number_strings[1]) + 1,
                        string.atoi(number_strings[2]) + 1,
                        string.atoi(number_strings[3]) + 1))
print "Done."

in_file.close()




#
# Open the output file.
#
out_file = open(sys.argv[2], "w")

# Opening bracket.
out_file.write("netcdf %s {\n" % string.split(sys.argv[2], '.')[0])


#
# Dimensions.
#
out_file.write("dimensions:\n")
out_file.write("\tlen_string = 33 ;\n")
out_file.write("\tlen_line = 81 ;\n")
out_file.write("\ttime_step = UNLIMITED ; // (0 currently)\n")
out_file.write("\tnum_dim = 3 ;\n")
out_file.write("\tnum_nodes = %d ;\n" % num_vertices)
out_file.write("\tnum_elem = %d ;\n" % num_simplices)
out_file.write("\tnum_el_blk = 1 ;\n")
out_file.write("\tnum_el_in_blk1 = %d ;\n" % num_simplices)
out_file.write("\tnum_nod_per_el1 = 4 ;\n")
	
#
# Variables.
#
out_file.write("""variables:
\tdouble time_whole(time_step) ;
\tint eb_prop1(num_el_blk) ;
\t\teb_prop1:name = "ID" ;
\tint connect1(num_el_in_blk1, num_nod_per_el1) ;
\t\tconnect1:elem_type = "TETRA" ;
\tdouble coordx(num_nodes) ;
\tdouble coordy(num_nodes) ;
\tdouble coordz(num_nodes) ;
""")

#
# Global attributes.
#
# CONTINUE Fix the file name and time stamp.
out_file.write("""// global attributes:
\t\t:api_version = 4.01f ;
\t\t:version = 3.01f ;
\t\t:floating_point_word_size = 8 ;
\t\t:file_size = 1 ;
\t\t:title = "cubit(C:/msys/1.0/home/seanmauch/m.e): 07/18/2005: 12:19:12" ;
""")

#
# Data.
#
out_file.write("data:\n")

# The elements.
out_file.write(" connect1 =\n")
for n in range(num_simplices - 1):
    out_file.write("  %d, %d, %d, %d,\n" % simplices[n])
out_file.write("  %d, %d, %d, %d;\n" % simplices[-1])

# The nodes.
out_file.write(" coordx =\n")
for n in range(num_vertices - 1):
    out_file.write("  %s,\n" % string.split(vertices[n])[0])
out_file.write("  %s;\n" % string.split(vertices[-1])[0])

out_file.write(" coordy =\n")
for n in range(num_vertices - 1):
    out_file.write("  %s,\n" % string.split(vertices[n])[1])
out_file.write("  %s;\n" % string.split(vertices[-1])[1])

out_file.write(" coordz =\n")
for n in range(num_vertices - 1):
    out_file.write("  %s,\n" % string.split(vertices[n])[2])
out_file.write("  %s;\n" % string.split(vertices[-1])[2])


# Closing bracket.
out_file.write("}\n")

#
# Close the output file.
#
out_file.close()
