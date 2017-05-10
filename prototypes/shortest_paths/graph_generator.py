"""Functions for generating graphs.

This module implements the following functions:
( num_vertices, edges ) = rectangular_grid( x_size, y_size, edge_weight )
( num_vertices, edges ) = sparse( num_vertices, num_adjacent, edge_weight )
( num_vertices, edges ) = complete( num_vertices, edge_weight )

These functions have no knowlege of any graph data structures.  They just
return lists of edges."""

def rectangular_grid( x_size, y_size, edge_weight ):
    """Make a rectangular grid.

    The vertices are lattice points.  Each vertex is doubly
    connected to its adjacent vertices.  The edges along the grid boundaries
    wrap to the opposite boundary.

    ( num_vertices, edges ) = rectangular_grid( x_size, y_size, edge_weight )
    INPUT:
    x_size - The number of vertices in the x direction.
    y_size - The number of vertices in the y direction.
    edge_weight - A function with no arguments giving the edge weight.
    RETURN:
    num_vertices - The number of vertices in the graph.
    edges - A list of edges.  Each edge is represented by a tuple of
    a source vertex index, a target vertex index and an edge weight."""

    # the number of vertices.
    num_vertices = x_size * y_size

    # The vertex identifier function.
    vertex_id = lambda i, j, x_size, y_size :\
        ((i + x_size) % x_size) + ((j + y_size) % y_size) * x_size

    edges = []
    # The edges.
    for i in range( x_size ):
        for j in range( y_size ):
            edges.append( ( vertex_id( i, j, x_size, y_size ),
                            vertex_id( i-1, j, x_size, y_size ),
                            edge_weight() ) )
            edges.append( ( vertex_id( i, j, x_size, y_size ),
                            vertex_id( i+1, j, x_size, y_size ),
                            edge_weight() ) )
            edges.append( ( vertex_id( i, j, x_size, y_size ),
                            vertex_id( i, j-1, x_size, y_size ),
                            edge_weight() ) )
            edges.append( ( vertex_id( i, j, x_size, y_size ),
                            vertex_id( i, j+1, x_size, y_size ),
                            edge_weight() ) )

    return ( num_vertices, edges )




def sparse( num_vertices, num_adjacent, edge_weight ):
    """Make a sparse graph.

    Each vertex has num_adjacent adjacent edges and num_adjacent incident
    edges.

    ( num_vertices, edges ) = sparse( num_vertices, num_adjacent, edge_weight )
    INPUT:
    num_vertices - The number of vertices in the graph.
    num_adjacent - The number of adjacent edges for each vertex.
    edge_weight - A function with no arguments giving the edge weight.
    RETURN:
    num_vertices - The number of vertices in the graph.
    edges - A list of edges.  Each edge is represented by a tuple of
    a source vertex index, a target vertex index and an edge weight."""

    import random

    # The indices of the vertices.
    vertices = range( num_vertices )

    #
    # Make the list of edges.
    #
    edges = []
    # Do num_adjacent times.
    for na in range( num_adjacent ):
        # Shuffle the vertex indices.
        random.shuffle( vertices )
        # Add an edge to each vertex.
        for i in range( num_vertices ):
            edges.append( ( vertices[i], 
                            vertices[(i+1) % num_vertices],
                            edge_weight() ) )
    
    return ( num_vertices, edges )



def complete( num_vertices, edge_weight ):
    """Make a complete graph.

    There are num_vertices * (num_vertices - 1) edges.

    ( num_vertices, edges ) = complete( num_vertices, edge_weight )
    INPUT:
    num_vertices - The number of vertices in the graph.
    edge_weight - A function with no arguments giving the edge weight.
    RETURN:
    num_vertices - The number of vertices in the graph.
    edges - A list of edges.  Each edge is represented by a tuple of
    a source vertex index, a target vertex index and an edge weight."""

    # The indices of the vertices.
    vertices = range( num_vertices )

    #
    # Make the list of edges.
    #
    edges = []
    for i in vertices:
        for j in vertices:
            if i != j:
                edges.append( ( i, j, edge_weight() ) )
    
    return ( num_vertices, edges )
