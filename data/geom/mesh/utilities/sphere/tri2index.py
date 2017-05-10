import sys, string

# Read the triangle vertices.
tri_vert_string = sys.stdin.readlines()
tri_vert_string = map( string.split, tri_vert_string )

# Convert strings to floats.
tri_vert = []
for x in tri_vert_string:
  tri_vert.append( map( string.atof, x ) )

# Make a list of the vertices.
vert = []
for x in tri_vert:
  if x not in vert:
    vert.append( x )

# Make a list of the faces.
faces = []
for i in range( 0, len( tri_vert), 3 ):
  v0 = tri_vert[i]
  v1 = tri_vert[i+1]
  v2 = tri_vert[i+2]
  for j in range( len( vert ) ):
    if v0 == vert[j]:
      i0 = j
      break
  for j in range( len( vert ) ):
    if v1 == vert[j]:
      i1 = j
      break
  for j in range( len( vert ) ):
    if v2 == vert[j]:
      i2 = j
      break
  faces.append( [ i0, i1, i2 ] )

# Print the b-rep.
print len( vert )
print len( faces )
for x in vert:
  print x[0], x[1], x[2]
for x in faces:
  print x[0], x[1], x[2]

