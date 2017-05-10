"""
buildHexMeshBoundary.py
Build the quad mesh boundary of a hex mesh.
Usage:
python buildHexMeshBoundary.py hexMesh quadMesh
"""

from QuadMesh import QuadMesh
from HexMesh import HexMesh

def buildHexMeshBoundary(hexMesh):
    quadMesh = QuadMesh()
    assert hexMesh.vertices
    quadMesh.spaceDimension = len(hexMesh.vertices[0])
    # Borrow the vertex list.
    quadMesh.vertices = hexMesh.vertices
    # Make the list of cells.
    # For each cell.
    for n in range(len(hexMesh.cells)):
        # For each side.
        for s in range(6):
            # If this is a boundary face.
            if hexMesh.cellAdjacencies[n][s] == None:
                quadMesh.cells.append(hexMesh.getIndexedFace(n, s))
    # Pack the mesh. This gets separate storage for the vertices.
    quadMesh.pack()
    return quadMesh

if __name__ == '__main__':
    import sys, string

    if len(sys.argv) != 3:
        print "Usage:"
        print "python buildHexMeshBoundary.py hexMesh quadMesh\n"

    hexMesh = HexMesh()
    inFile = open(sys.argv[1], "r")
    hexMesh.read(inFile)
    inFile.close()

    quadMesh = buildHexMeshBoundary(hexMesh)
    outFile = open(sys.argv[2], "w")
    quadMesh.write(outFile)
    outFile.close()
