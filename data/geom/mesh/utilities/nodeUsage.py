# nodeUsage.py
# Count how many times each node is used.
#
# Usage:
# python nodeUsage.py mesh meshWithUsage

import sys, string

if len(sys.argv) != 3:
    print("Usage:")
    print("python nodeUsage.py mesh meshWithUsage\n")
    print("This program reads a mesh and writes a mesh with node")
    print("usage information.")
    raise "Bad command line arguments.  Exiting..."

#
# Open the files.
#
inFile = open(sys.argv[1], "r")
outFile = open(sys.argv[2], "w")

#
# The space dimension and the simplex dimension.
#
numberStrings = inFile.readline().split()
assert len(numberStrings) == 2
spaceDimension = int(numberStrings[0])
cellDimension = int(numberStrings[1])
print("space dimension = %d, simplex dimension = %d" % (spaceDimension, 
                                                        cellDimension))

#
# Read the nodes.  Store these as strings.
#
numNodes = int(inFile.readline())
print("Reading the %d node locations..." % numNodes)
nodeStrings = []
for n in range(numNodes):
    nodeStrings.append(inFile.readline())
print("Done.")

#
# Read the elements.
#
numElements = int(inFile.readline())
print("Reading the %d elements...." % numElements)
elements = []
for n in range(numElements):
    numberStrings = string.split(inFile.readline())
    elements.append(map(int, numberStrings))
inFile.close()
print("Done.")

#
# See how many times each node is used.
#
print("Determining usage of the nodes...")
usage = [0] * numNodes
for element in elements:
    for node in element:
        usage[node] += 1

# Write the mesh with node usage information.
print("Writing the mesh with node usage information...")
outFile.write("%d %d\n" % (spaceDimension, cellDimension))
outFile.write("%d\n" % numNodes)
for i in range(numNodes):
    outFile.write("%s %d\n" % (nodeStrings[i][:-1], usage[i]))
outFile.write("%d\n" % numElements)
for element in elements:
    for node in element:
        outFile.write("%d " % node)
    outFile.write("\n")
outFile.close()
print("Done.")

histogram = {}
for count in usage:
    if count in histogram:
        histogram[count] += 1
    else:
        histogram[count] = 1
print('Node usage histogram:')
print(histogram)

