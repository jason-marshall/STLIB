# HexMesh.py

import string

def intersect(x, y):
    """Return the intersection of the two sets."""
    result = []
    for element in x:
        if element in y:
            result.append(element)
    return result

class HexMesh:
    """For a unit hexahedron in the first octant, the vertices have the
    following numbering:
    0 (0, 0, 0)
    1 (1, 0, 0)
    2 (1, 1, 0)
    3 (0, 1, 0)
    4 (0, 0, 1)
    5 (1, 0, 1)
    6 (1, 1, 1)
    7 (0, 1, 1)
    The six sides of the hexadron are numbered:
    0 -x
    1 +x
    2 -y
    3 +y
    4 -z
    5 +z
    """
    def __init__(self):
        # Positively oriented quadrilaterals that form the sides.
        self.sideIndices = ((0, 4, 7, 3),
                            (1, 2, 6, 5),
                            (0, 1, 5, 4),
                            (2, 3, 7, 6),
                            (0, 3, 2, 1),
                            (4, 5, 6, 7))
        self.clear()

    def clear(self):
        self.vertices = []
        self.cells = []
        self.vertexCellIncidences = []
        self.cellAdjacencies = []

    def getIndexedFace(self, index, side):
        c = self.cells[index]
        s = self.sideIndices[side]
        return (c[s[0]], c[s[1]], c[s[2]], c[s[3]])

    def updateTopology(self):
        #
        # Vertex-cell incidences.
        #
        self.vertexCellIncidences = []
        for n in range(len(self.vertices)):
            self.vertexCellIncidences.append([])
        for n in range(len(self.cells)):
            for index in self.cells[n]:
                self.vertexCellIncidences[index].append(n)
        #
        # Cell adjacencies.
        #
        self.cellAdjacencies = []
        # For each cell.
        for n in range(len(self.cells)):
            cell = self.cells[n]
            adjacencies = []
            # For each side.
            for side in self.sideIndices:
                # The cells incident to the first vertex.
                adjacent = self.vertexCellIncidences[cell[side[0]]][:]
                # Remove this cell.
                del adjacent[adjacent.index(n)]
                # Intersect with the cells incident to the other 3 vertices.
                for i in range(1, 4):
                    adjacent = \
                        intersect(adjacent,
                                  self.vertexCellIncidences[cell[side[i]]])
                if adjacent:
                    # There can be at most one adjacent cell in this direction.
                    if len(adjacent) != 1:
                        raise "Bad adjacency detected."
                    adjacencies.append(adjacent[0])
                else:
                    # None indicates that there is no adjacent cell.
                    adjacencies.append(None)
            self.cellAdjacencies.append(tuple(adjacencies))

    def read(self, file):
        self.clear()
        # Read the space dimension and cell dimension.
        s = string.split(file.readline())
        if len(s) != 2:
            raise "Error: Could not read space and cell dimensions."
        if int(s[0]) != 3:
            raise "Error: Bad space dimension."
        if int(s[1]) != 3:
            raise "Error: Bad cell dimension."
        # Read the vertices.
        numberOfVertices = int(file.readline())
        for n in range(numberOfVertices):
            self.vertices.append(tuple(map(float, 
                                           string.split(file.readline()))))
        # Read the cells.
        numberOfCells = int(file.readline())
        for n in range(numberOfCells):
            self.cells.append(tuple(map(int, string.split(file.readline()))))
        # Compute the incidence relations.
        self.updateTopology()

    def write(self, file):
        assert self.vertices
        # Write the space dimension and cell dimension.
        file.write("3 3\n")
        # Write the vertices.
        file.write("%d\n" % len(self.vertices))
        if self.vertices:
            format = "%f" + " %f" * (len(self.vertices[0]) - 1) + "\n"
        for vertex in self.vertices:
            file.write(format % vertex)
        # Write the cells.
        file.write("%d\n" % len(self.cells))
        format = "%d" + " %d" * 7 + "\n"
        for cell in self.cells:
            file.write(format % cell)

if __name__ == '__main__':
    #from QuadMesh import QuadMesh
    from buildHexMeshBoundary import buildHexMeshBoundary
    #
    # One cell.
    #
    hexString = """3 3
8
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
1
0 1 2 3 4 5 6 7
"""
    # Make a mesh file.
    file = open("tmp.txt", "w")
    file.write(hexString)
    file.close()
    # Read the file.
    file = open("tmp.txt", "r")
    mesh = HexMesh();
    mesh.read(file)
    file.close()
    assert len(mesh.vertices) == 8
    assert mesh.vertices[0] == (0, 0, 0)
    assert len(mesh.cells) == 1
    assert mesh.cells[0] == (0, 1, 2, 3, 4, 5, 6, 7)
    assert len(mesh.vertexCellIncidences) == len(mesh.vertices)
    for list in mesh.vertexCellIncidences:
        assert len(list) == 1
        assert list[0] == 0
    assert len(mesh.cellAdjacencies) == len(mesh.cells)
    assert mesh.cellAdjacencies[0] == (None, None, None, None, None, None)
    # Boundary.
    quadMesh = buildHexMeshBoundary(mesh)
    assert len(quadMesh.vertices) == 8
    assert len(quadMesh.cells) == 6
    #
    # Two cells.
    #
    hexString = """3 3
12
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
0 0 2
1 0 2
1 1 2
0 1 2
2
0 1 2 3 4 5 6 7
4 5 6 7 8 9 10 11
"""
    # Make a mesh file.
    file = open("tmp.txt", "w")
    file.write(hexString)
    file.close()
    # Read the file.
    file = open("tmp.txt", "r")
    mesh = HexMesh();
    mesh.read(file)
    file.close()
    assert len(mesh.vertices) == 12
    assert mesh.vertices[0] == (0, 0, 0)
    assert len(mesh.cells) == 2
    assert mesh.cells[0] == (0, 1, 2, 3, 4, 5, 6, 7)
    assert mesh.cells[1] == (4, 5, 6, 7, 8, 9, 10, 11)
    assert len(mesh.vertexCellIncidences) == len(mesh.vertices)
    for n in range(0, 4):
        assert len(mesh.vertexCellIncidences[n]) == 1
        assert mesh.vertexCellIncidences[n][0] == 0
    for n in range(4, 8):
        assert len(mesh.vertexCellIncidences[n]) == 2
        assert 0 in mesh.vertexCellIncidences[n]
        assert 1 in mesh.vertexCellIncidences[n]
    for n in range(8, 12):
        assert len(mesh.vertexCellIncidences[n]) == 1
        assert mesh.vertexCellIncidences[n][0] == 1
    assert len(mesh.cellAdjacencies) == len(mesh.cells)
    assert mesh.cellAdjacencies[0] == (None, None, None, None, None, 1)
    assert mesh.cellAdjacencies[1] == (None, None, None, None, 0, None)
    # Boundary.
    quadMesh = buildHexMeshBoundary(mesh)
    assert len(quadMesh.vertices) == 12
    assert len(quadMesh.cells) == 10
