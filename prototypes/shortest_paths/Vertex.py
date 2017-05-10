"The Vertex class for a graph."

# Vertex.py

# The possible states of a vertex.
KNOWN = 0
LABELED = 1
UNLABELED = 2

class Vertex:
    "A vertex in a graph."

    #
    # Constructor
    #
    def __init__( self, identifier ):
        """Construct a Vertex.

        Make all attributes except the identifier have null values."""
        self.identifier = identifier
        self.status = None
        self.distance = None
        self.predecessor = None
        self.incident_edges = []
        self.adjacent_edges = []
        self.min_incident_edge_weight = None
        self.unknown_incident_edge = None
        self.unknown_incident_index = None

    #
    # Mathematical Operations
    #

    def initialize( self ):
        """Initialize the vertex for a shortest path computation.

        Set the status of the vertex to UNLABELED.
        Sort the incident edges by weight.
        Initialize the unknown incident edge."""

        self.status = UNLABELED
        self.distance = 1e100
        self.predecessor = None
        self.incident_edges.sort( lambda a, b: cmp( a.weight, b.weight ) )
        self.unknown_incident_index = 0
        if len( self.incident_edges ) > 0:
            self.unknown_incident_edge = self.incident_edges[0]
            self.min_incident_edge_weight = self.incident_edges[0].weight
        else:
            self.unknown_incident_edge = None
        
    def label( self, vertex, weight ):
        """Label self from a vertex and a weight.

        If the status is UNLABELED, change the status to LABELED and
        set the distance and predecessor.  Otherwise, see if the distance
        from the vertex is lower than the current distance and if so,
        update the distance and predecessor."""
        if self.status == UNLABELED:
            self.status = LABELED
            self.distance = vertex.distance + weight
            self.predecessor = vertex
        else: # self.status == LABELED
            new_distance = vertex.distance + weight
            if new_distance < self.distance:
                self.distance = new_distance
                self.predecessor = vertex

    def label_adjacent( self ):
        """Label the unknown neighbors of the specified vertex.

        Return the neighbors that were unlabeled."""
        unlabeled_neighbors = []
        for edge in self.adjacent_edges:
            adjacent = edge.target
            if adjacent.status != KNOWN:
                if adjacent.status == UNLABELED:
                    unlabeled_neighbors.append( adjacent )
                adjacent.label( self, edge.weight )
        return unlabeled_neighbors

    def label_status_adjacent( self ):
        """Label the status of unkown neighbors of the specified vertex.

        USAGE:
        unlabeled_neighbors = vertex.label_status_adjacent()
        RETURN:
        The neighbors that were unlabeled."""
        
        unlabeled_neighbors = []
        for edge in self.adjacent_edges:
            adjacent = edge.target
            if adjacent.status == UNLABELED:
                unlabeled_neighbors.append( adjacent )
                adjacent.status = LABELED
        return unlabeled_neighbors

    def get_unknown_incident_edge( self ):
        """Get the current unknown incident edge.

        Set unknown_incident_edge to the current unknown incident edge.
        Set unknown_incident_index to this edge's index.
        If there are no unknown incident edges left,
        Set unknown_incident_edge to None."""

        # CONTINUE REMOVE
#        if self.unknown_incident_index >= len( self.incident_edges ):
#            raise "Error in get_min_unknown_incident_edge()."
        
        # Go through the incident edges until an unknown one is found.
        while ( self.unknown_incident_edge and
                self.unknown_incident_edge.source.status == KNOWN ):
            self.unknown_incident_index += 1
            if self.unknown_incident_index < len( self.incident_edges ):
                self.unknown_incident_edge = self.incident_edges[
                    self.unknown_incident_index ]
            else:
                self.unknown_incident_edge = None

    def get_next_unknown_incident_edge( self ):
        """Get the next unknown incident edge."""

        # Take one step.
        self.unknown_incident_index += 1
        if self.unknown_incident_index < len( self.incident_edges ):
            self.unknown_incident_edge = self.incident_edges[
                self.unknown_incident_index ]
        else:
            self.unknown_incident_edge = None
            
        # Go through the incident edges until an unknown one is found.
        self.get_unknown_incident_edge()

    def lower_bound( self, min_unknown, level ):
        """Return a lower bound on the distance of this vertex.

        INPUT:
        min_unknown - the minimum unknown distance.
        level - the depth of recursion"""

        # CONTINUE REMOVE
        if self.status == KNOWN:
            raise "Error in lower_bound()."
        
        # CONTINUE REMOVE
        if self.distance < min_unknown:
            raise "Distance error in lower_bound()."
        
        if level == 0:
            return min_unknown
        if level == 1:
            return min( self.distance, min_unknown
                        + self.min_incident_edge_weight )

        min_distance = self.distance
        for edge in self.incident_edges:
            if edge.source.status != KNOWN:
                distance = edge.weight + edge.source.lower_bound(
                    min_unknown, level - 2 )
                if distance < min_distance:
                    min_distance = distance
        return min_distance

    def is_correct( self, min_unknown, level ):
        "Return true if the distance of the vertex is known to be correct."

        # CONTINUE REMOVE
        if self.status == KNOWN:
            raise "Error in is_correct()."
        
        if level <= 1:
            if self.distance > self.lower_bound( min_unknown, level ):
                return 0
        else:
            self.get_unknown_incident_edge()
            while self.unknown_incident_edge:
                if ( self.distance > self.unknown_incident_edge.weight
                     + self.unknown_incident_edge.source.lower_bound(
                    min_unknown, level - 2 ) ):
                    return 0
                # Get the next unknown incident edge.
                self.get_next_unknown_incident_edge()
        return 1

##    def lower_bound( self, min_unknown ):
##        """Return a lower bound on the distance of this vertex.

##        PRECONDITION:
##        The status of this vertex is not KNOWN."""

##        if self.status == UNLABELED:
##            return self.min_unknown_incident_edge.weight + min_unknown

##        # Find the minimum unknown incident edge.
##        self.get_min_unknown_incident_edge()
##        # If there are any unknown edges that could affect the distance
##        if self.min_unknown_incident_edge:
##            return min( self.distance,
##                        self.min_unknown_incident_edge.weight + min_unknown )
##        return self.distance

##    def is_correct_0( self, min_unknown ):
##        "Return true if the distance of the vertex is known to be correct."

##        if ( self.distance <= min_unknown ):
##            return 1
##        return 0
        
##    def is_correct_1( self, min_unknown ):
##        "Return true if the distance of the vertex is known to be correct."

##        return self.is_correct_0( min_unknown + self.min_incident_edge_weight )
        
##    def is_correct_2( self, min_unknown ):
##        "Return true if the distance of the vertex is known to be correct."

##        # Find the minimum unknown incident edge.
##        self.get_min_unknown_incident_edge()

##        # If there are no incident edges from unknown vertices or
##        # the distance is known because it is <= weight + min_unknown.
##        if ( self.min_unknown_incident_edge == None or
##             self.distance <= self.min_unknown_incident_edge.weight
##             + min_unknown ):
##            return 1
##        return 0
        
##    def is_correct( self, min_unknown ):
##        "Return true if the distance of the vertex is known."

##        # Find the minimum unknown incident edge.
##        self.get_min_unknown_incident_edge()

##        # If there are no incident edges from unknown vertices.
##        if self.min_unknown_incident_edge == None:
##            return 1

##        while self.min_unknown_incident_edge:
##            if ( self.distance <= self.min_unknown_incident_edge.weight
##                 + min_unknown ):
##                return 1
##            if ( self.distance > self.min_unknown_incident_edge.weight
##                 + self.min_unknown_incident_edge.source.lower_bound(
##                min_unknown ) ):
##                return 0
##            # Get the next unknown incident edge.
##            self.get_next_unknown_incident_edge()

##        return 1

    def display( self ):
        print "status =", self.status
        print "distance =", self.distance
        if self.predecessor:
            print "predecessor =", self.predecessor.identifier
        else:
            print "predecessor = ", self.predecessor
        print "incident edges:"
        for edge in self.incident_edges:
            edge.display()
        print "adjacent edges:"
        for edge in self.adjacent_edges:
            edge.display()
