"The Graph class used for computing shortest path trees."

import sys

from Vertex import *
from Edge import *

class Graph:
    "A Graph used for computing shortest path trees."

    #
    # Constructors
    #

    def clear( self ):
        "Clear the graph.  Remove all vertices and edges."
        # Clear the list of vertices.
        self.vertices = []
        # Clear the list of edges.
        self.edges = []
        
    def __init__( self ):
        "Make an empty graph."
        self.clear()

    def make( self, num_vertices, edges ):
        """Make a graph from a list of edges.

        graph.make( num_vertices, edges )
        num_vertices - The number of vertices in the graph.
        edges - A list of edges.  Each edge is represented by a tuple of
        a source vertex index, a target vertex index and an edge weight."""
        
        # Clear the graph.
        self.clear()
        
        # Add the vertices.
        for n in xrange( num_vertices ):
            self.add_vertex()

        # Add the edges.
        for ( source, target, weight ) in edges:
            self.add_edge( source, target, weight )


    #
    # Comparison
    #

    def __eq__( self, other ):
        "Return true if the vertices have equal distances and predecessors."

        # Check if each has the same number of vertices.
        if len( self.vertices ) != len( other.vertices ):
            return 0

        # Check each vertex.
        for i in range( len( self.vertices ) ):
            if not ( self.vertices[i].distance == other.vertices[i].distance
                     and
                     ( self.vertices[i].predecessor
                       == other.vertices[i].predecessor == None or
                       self.vertices[i].predecessor.identifier
                       == other.vertices[i].predecessor.identifier ) ):
                # CONTINUE REMOVE
                print self.vertices[i].distance, other.vertices[i].distance
                return 0
        return 1

    
    #
    # Manipulators
    #
    
    def add_vertex( self ):
        "Add a vertex to the graph."
        self.vertices.append( Vertex( len( self.vertices ) ) )

    def add_edge( self, source_index, target_index, weight ):
        "Add an edge to the graph."
        # The new edge.
        edge = Edge( self.vertices[source_index], self.vertices[target_index],
                     weight )
        
        # Add the edge.
        self.edges.append( edge )
        # Add info to the source and target vertices.
        self.vertices[source_index].adjacent_edges.append( edge )
        self.vertices[target_index].incident_edges.append( edge )

    #
    # Mathematical Operations
    #

    def initialize( self, source_index ):
        """Initialize for a shortest path computation.

        USAGE:
        initialize( self, source_index )
        INPUT:
        source_index - the index of the root vertex of the tree."""
        
        # Initialize the data in each vertex.
        for vertex in self.vertices:
            vertex.initialize()
            
        # Set the source vertex to known.
        source = self.vertices[source_index]
        source.status = KNOWN
        source.distance = 0
        source.predecessor = None


    def initialize_status( self, source_index ):
        """Initialize the status of each vertex.

        Useful for analyzing a shortest path computation.

        USAGE:
        initialize_status( self, source_index )
        INPUT:
        source_index - the index of the root vertex of the tree."""
        
        # Set the status in each vertex.
        for vertex in self.vertices:
            vertex.status = UNLABELED
            
        # Set the source vertex to known.
        source = self.vertices[source_index]
        source.status = KNOWN


    def marching_with_correctness_criterion( self, vertex_index, level ):
        """Compute the shortest path tree from the given vertex.

        Use the marching with correctness criterion algorithm.
        
        USAGE:
        determined_fraction = graph.determined_vertices( vertex_index, level )
        INPUT:
        vertex_index - the index of the root vertex.
        level - the level for the correctness criterion.
        RETURN:
        The fraction of determined vertices."""
        
        # The list of labeled unknown vertices to check during a step.
        labeled = []
        # Vertices that will be in the label set in the next step.
        labeled_next = []

        # Initialize the graph.
        self.initialize( vertex_index )
        # Label the neighbors of the root.
        labeled = self.vertices[vertex_index].label_adjacent()

        # All vertices are known when there are no labeled vertices left.
        # Loop while there are labeled vertices left.
        num_labeled = 0
        #num_iterations = 0
        while labeled:
            #num_iterations += 1
            num_labeled += len( labeled )
            
            # Find the minimum unknown distance.
            min_unknown = min( map( lambda a: a.distance, labeled ) )
            
            for vertex in labeled:
                if vertex.is_correct( min_unknown, level ):
                    vertex.status = KNOWN
                    labeled_next += vertex.label_adjacent()
                else:
                    # We'll have to examine this one next time.
                    labeled_next.append( vertex )

            # Get the labeled lists ready for the next step.
            labeled = labeled_next
            labeled_next = []

        return (1.0 * (len( self.vertices ) - 1)) / num_labeled

    
    def dijkstra( self, vertex_index ):
        """Compute the shortest path tree with Dijkstra's algorithm.

        vertex_index - the index of the root vertex.
        Return the number of iterations required."""
        
        # The list of labeled unknown vertices to check during a step.
        labeled = []

        # Initialize the graph.
        self.initialize( vertex_index )
        labeled = self.vertices[vertex_index].label_adjacent()

        # All vertices are known when there are no labeled vertices left.
        # Loop while there are labeled vertices left.
        num_iterations = 0
        while labeled:
            num_iterations += 1

            # Find the labeled vertex with minimum distance.
            labeled_distances = map( lambda a: a.distance, labeled )
            min_distance = min( labeled_distances )
            min_index = labeled_distances.index( min_distance )
            min_vertex = labeled.pop( min_index )
            
            min_vertex.status = KNOWN
            labeled += min_vertex.label_adjacent()
            
        return num_iterations
    
    def determined_vertices( self, vertex_index ):
        """Get statistics on the number of determined vertices.

        First the shortest-paths tree from vertex_index is computed.
        An ideal algorithm would set all determined vertices in the
        labeled set to known status at each step in the algorithm.  We
        look at what portion of the labeled vertices that are determined
        from the known vertices at each step of this ideal algorithm.
        
        USAGE:
        determined_fraction = graph.determined_vertices( vertex_index )
        INPUT:
        vertex_index - the index of the root vertex.
        RETURN:
        The fraction of determined vertices."""

        # Compute the shortest-paths tree.
        self.marching_with_correctness_criterion( vertex_index, 1 )

        # The list of labeled unknown vertices to check during a step.
        labeled = []
        # Vertices that will be in the label set next step.
        labeled_next = []

        # Initialize status fields in the graph.
        self.initialize_status( vertex_index )
        labeled = self.vertices[vertex_index].label_status_adjacent()

        num_labeled = 0
        # All vertices are known when there are no labeled vertices left.
        # Loop while there are labeled vertices left.
        while labeled:
            # Update the number of labeled vertices.
            num_labeled += len( labeled )

            # Loop over the labeled vertices.
            for v in labeled:
                # If the vertex is determined from a known vertex.
                if v.predecessor.status == KNOWN:
                    # This vertex is known.
                    v.status = KNOWN
                    # Add the adjacent neighbors of this known vertex.
                    labeled_next += v.label_status_adjacent()
                else:
                    # The vertex will stay in the labeled set.
                    labeled_next.append( v )

            labeled = labeled_next
            labeled_next = []

        return (1.0 * (len( self.vertices ) - 1)) / num_labeled
    

    def dijkstra_determined_vertices( self, vertex_index ):
        """Get statistics on the number of determined vertices using Dijkstra.

        First the shortest-paths tree from vertex_index is computed.
        Then we look at what portion of the labeled vertices are
        determined from the known vertices at each step of Dijkstra's
        algorithm.
        
        USAGE:
        determined_frac = graph.dijkstra_determined_vertices( vertex_index )
        INPUT:
        vertex_index - the index of the root vertex.
        RETURN:
        The fraction of determined vertices."""

        # Compute the shortest-paths tree.
        self.marching_with_correctness_criterion( vertex_index, 1 )

        # The list of labeled unknown vertices to check during a step.
        labeled = []

        # Initialize status fields in the graph.
        self.initialize_status( vertex_index )
        labeled = self.vertices[vertex_index].label_status_adjacent()

        num_determined = 0
        num_labeled = 0
        # All vertices are known when there are no labeled vertices left.
        # Loop while there are labeled vertices left.
        while labeled:
            # Find number of vertices that are determined from known vertices.
            num_determined += reduce( lambda s, v :
                                      s + (v.predecessor.status == KNOWN),
                                      labeled, 0 )
            # Update the number of labeled vertices.
            num_labeled += len( labeled )
            
            # Find the labeled vertex with minimum distance.
            labeled_distances = map( lambda a: a.distance, labeled )
            min_distance = min( labeled_distances )
            min_index = labeled_distances.index( min_distance )
            # Remove the min vertex from the queue.
            min_vertex = labeled.pop( min_index )
            # The min vertex is known.
            min_vertex.status = KNOWN
            # Add the adjacent neighbors of this known vertex.
            labeled += min_vertex.label_status_adjacent()

        return (1.0 * num_determined ) / num_labeled
    

    #
    # I/O
    #
    
    def display( self ):
        print "vertices:"
        for i in range( len( self.vertices ) ):
            print
            print "vertex ", i
            self.vertices[i].display()
        print
        print "edges:"
        for i in range( len( self.edges ) ):
            sys.stdout.write('%d ' % i)
            self.edges[i].display()
        
