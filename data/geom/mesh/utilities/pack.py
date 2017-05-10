# pack.py
# Pack the nodes.  Remove the unused nodes from the node location list.
# Adjust the node indices accordingly.
#
# Usage:
# python pack.py mesh packedMesh

import sys

if len(sys.argv) != 3:
    print("Usage:")
    print("python pack.py mesh packedMesh\n")
    raise "Bad command line arguments.  Exiting..."

# Open the files.
inFile = open(sys.argv[1], "r")
outFile = open(sys.argv[2], "w")

#
# The space dimension and the simplex dimension.
#
numberStrings = inFile.readline().split()
assert len(numberStrings) == 2
spaceDimension = int(numberStrings[0])
simplexDimension = int(numberStrings[1])
print("space dimension = %d, simplex dimension = %d" % (spaceDimension, 
                                                        simplexDimension))

#
# Read the nodes.
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
    numberStrings = inFile.readline().split()
    elements.append(map(int, numberStrings))
inFile.close()
print("Done.")

#
# See which nodes are used.
#
print("Packing the nodes...")
used = [0] * numNodes
for element in elements:
    for node in element:
        used[node] = 1

#
# Calculate the new indices.
#
indices = [0] * numNodes
for i in range(1, numNodes):
    if used[i]:
        indices[i] = indices[i-1] + 1
    else:
        indices[i] = indices[i-1]
numPackedNodes = indices[-1] + 1

#
# Adjust the node indices.
#
for element in elements:
    for n in range(len(element)):
        element[n] = indices[element[n]]
print("Done.")

print("Packed mesh: %d nodes, %d elements" % (numPackedNodes, numElements))

#
# Write the packed mesh.
#
print("Writing the packed mesh...")
outFile.write("%d %d\n" % (spaceDimension, simplexDimension))
outFile.write("%d\n" % numPackedNodes)
for i in range(numNodes):
    if used[i]:
        outFile.write(nodeStrings[i])

outFile.write("%d\n" % numElements)
for element in elements:
    outFile.write(' '.join(map(str, element)) + '\n')
outFile.close()
print("Done.")
