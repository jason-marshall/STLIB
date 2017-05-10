"The edge class for a graph."

class Edge:
    "An edge in a graph."
    
    def __init__( self, source = None, target = None, weight = None ):
        self.source = source
        self.target = target
        self.weight = weight
        
    def display( self ):
        print "source =", self.source.index, " target =", self.target.index, " weight =", self.weight
