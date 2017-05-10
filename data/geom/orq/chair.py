#
# Return a list of the vertices on the surface of a brick.
#
def brick(min_pt, max_pt):

    #
    # The corners of the brick.
    #
    xmin = min_pt[0]
    ymin = min_pt[1]
    zmin = min_pt[2]
    xmax = max_pt[0]
    ymax = max_pt[1]
    zmax = max_pt[2]

    #
    # Initially the list of vertices is empty.
    #
    vertices = [];

    #
    # Make the bottom and top of the brick
    #
    for j in range(ymin, ymax + 1):
        for i in range(xmin, xmax + 1):
            vertices.append((i, j, zmin))
            vertices.append((i, j, zmax))

    #
    # Make the sides
    #
    for k in range(zmin + 1, zmax):
        for i in range(xmin, xmax + 1):
            vertices.append((i, ymin, k))
            vertices.append((i, ymax, k))
        for j in range(ymin + 1, ymax):
            vertices.append((xmin, j, k))
            vertices.append((xmax, j, k))

    return vertices

#
# Return the vertices on a chair.
#
def chair(res):
    assert(res >= 1)

    #
    # Initially the list of vertices is empty.
    #
    vertices = [];

    
    #
    # Make the four legs.
    #
    vertices[len(vertices):] = brick((0, 0, 0), (res, res, 18 * res))
    vertices[len(vertices):] = brick((17 * res, 0, 0), \
                                      (18 * res, res, 18 * res))
    vertices[len(vertices):] = brick((0, 17 * res, 0), \
                                      (res, 18 * res, 18 * res))
    vertices[len(vertices):] = brick((17 * res, 17 * res, 0), \
                                      (18 * res, 18 * res, 18 * res))
    
    #
    # Make the four leg crossbars.
    #
    vertices[len(vertices):] = brick((res, 0, 9 * res), \
                                      (17 * res, res, 10 * res))
    vertices[len(vertices):] = brick((res, 17 * res, 9 * res), \
                                      (17 * res, 18 * res, 10 * res))
    vertices[len(vertices):] = brick((0, res, 10 * res), \
                                      (res, 17 * res, 11 * res))
    vertices[len(vertices):] = brick((17 * res, res, 10 * res), \
                                      (18 * res, 17 * res, 11 * res))

    #
    # Make the seat.
    #
    vertices[len(vertices):] = brick((0, 0, 18 * res), \
                                      (18 * res, 18 * res, 20 * res))

    #
    # Make the back.
    #
    vertices[len(vertices):] = brick((0, 0, 20 * res), \
                                      (res, res, 40 * res))
    vertices[len(vertices):] = brick((17 * res, 0, 20 * res), \
                                      (18 * res, res, 40 * res))

    #
    # Make the back crossbars.
    #
    vertices[len(vertices):] = brick((res, 0, 28 * res), \
                                      (17 * res, res, 32 * res))
    vertices[len(vertices):] = brick((res, 0, 36 * res), \
                                      (17 * res, res, 40 * res))

    #
    # Remove duplicate vertices.
    #
    vertices.sort();
    i = 0
    length = len(vertices)
    while i < length:
        if vertices[i]:
            j = i + 1
            while j < length and vertices[i] == vertices[j]:
                vertices[j] = None
                j = j + 1
        i = i + 1

    distinct_vertices = []
    for x in vertices:
        if x:
            distinct_vertices.append(x)

    return distinct_vertices


#
# Print the vertices on a chair.
#
def print_chair(res, filename):
    assert(res >= 1)

    #
    # Make the chair.
    #
    vertices = chair(res)
    
    #
    # Open the file.
    #
    fout = open(filename, 'w')
    
    #
    # Write the number of vertices and faces.
    #
    fout.write('%i\n' % len(vertices))
    fout.write('%i\n' % 0)

    #
    # Write the vertices.
    #
    for pt in vertices:
	fout.write('%i %i %i\n' % pt)

    fout.close()

    return
