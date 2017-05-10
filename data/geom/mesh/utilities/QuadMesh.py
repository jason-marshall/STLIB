# QuadMesh.py

import string

class QuadMesh:
    """For a unit quadrilateral in the first quadrant, the vertices have the
    following numbering:
    0 (0, 0)
    1 (1, 0)
    2 (1, 1)
    3 (0, 1)
    The four sides of the quadrilateral are numbered:
    0 -x
    1 +x
    2 -y
    3 +y
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.spaceDimension = None
        self.vertices = []
        self.cells = []

    def read(self, file):
        self.clear()
        # Read the space dimension and cell dimension.
        s = string.split(file.readline())
        if len(s) != 2:
            raise "Error: Could not read space and cell dimensions."
        # The space dimension can be 2 or greater.
        self.spaceDimension = int(s[0])
        if self.spaceDimension < 2:
            raise "Error: Bad space dimension."
        if int(s[1]) != 2:
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

    def write(self, file):
        # Write the space dimension and cell dimension.
        file.write("%d 2\n" % self.spaceDimension)
        # Write the vertices.
        file.write("%d\n" % len(self.vertices))
        if self.vertices:
            format = "%f" + " %f" * (self.spaceDimension - 1) + "\n"
        for vertex in self.vertices:
            file.write(format % vertex)
        # Write the cells.
        file.write("%d\n" % len(self.cells))
        format = "%d" + " %d" * 3 + "\n"
        for cell in self.cells:
            file.write(format % cell)

    def pack(self):
        #
        # Determine which vertices are used.
        #
        used = [False] * len(self.vertices)
        for cell in self.cells:
            for index in cell:
                used[index] = True
        numberUsed = 0
        for x in used:
            if x:
                numberUsed = numberUsed + 1

        # The packed vertices.
        packedVertices = [None] * numberUsed
        vertexIndex = 0
        # This list maps the old vertex indices to the packed vertex indices.
        indices = [None] * len(self.vertices)
        # Loop over the vertices.
        for n in range(len(self.vertices)):
            # If the vertex is used in the packed mesh.
            if used[n]:
                # Calculate its index in the packed mesh.
                indices[n] = vertexIndex
                # Add the vertex to the packed mesh.
                packedVertices[vertexIndex] = self.vertices[n]
                vertexIndex = vertexIndex + 1
        # Check the validity of the packed vertices.
        assert not None in packedVertices
        # Replace the full set of vertices with the packed vertices.
        self.vertices = packedVertices

        # Map the vertex indices from the old mesh to the packed mesh.
        for n in range(len(self.cells)):
            v = list(self.cells[n])
            for m in range(len(v)):
                v[m] = indices[v[m]]
            self.cells[n] = tuple(v)

if __name__ == '__main__':
    #
    # One cell.
    #
    quadString = """2 2
4
0 0
1 0
1 1
0 1
1
0 1 2 3
"""
    # Make a mesh file.
    file = open("tmp.txt", "w")
    file.write(quadString)
    file.close()
    # Read the file.
    file = open("tmp.txt", "r")
    mesh = QuadMesh();
    mesh.read(file)
    file.close()
    assert len(mesh.vertices) == 4
    assert mesh.vertices[0] == (0, 0)
    assert len(mesh.cells) == 1
    assert mesh.cells[0] == (0, 1, 2, 3)
    #
    # Two cells.
    #
    quadString = """2 2
6
0 0
1 0
1 1
0 1
2 0
2 1
2
0 1 2 3
1 4 5 2
"""
    # Make a mesh file.
    file = open("tmp.txt", "w")
    file.write(quadString)
    file.close()
    # Read the file.
    file = open("tmp.txt", "r")
    mesh = QuadMesh();
    mesh.read(file)
    file.close()
    assert len(mesh.vertices) == 6
    assert mesh.vertices[0] == (0, 0)
    assert len(mesh.cells) == 2
    assert mesh.cells[0] == (0, 1, 2, 3)
    assert mesh.cells[1] == (1, 4, 5, 2)
    #
    # Pack cell.
    #
    quadString = """2 2
8
-1 0
-1 1
0 0
1 0
1 1
0 1
2 0
2 1
1
2 3 4 5
"""
    # Make a mesh file.
    file = open("tmp.txt", "w")
    file.write(quadString)
    file.close()
    # Read the file.
    file = open("tmp.txt", "r")
    mesh = QuadMesh();
    mesh.read(file)
    file.close()

    assert len(mesh.vertices) == 8
    mesh.pack()
    assert len(mesh.vertices) == 4
    assert mesh.vertices[0] == (0, 0)
    assert len(mesh.cells) == 1
    assert mesh.cells[0] == (0, 1, 2, 3)
