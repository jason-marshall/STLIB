stlib/performance/geom/orq/mesh/README.txt

These examples test the data structures for the case
that the records lie on a mesh in 3D.  A window query is done for each record
in the file.  All the executables take the mesh file and the search
radius as command line arguments.  Most have additional command line
arguments such as leaf size or cell size.  For example,
	KDTree.exe ../../../data/geom/mesh/33/cube.txt 0.5 2
uses a kd-tree with a leaf size of 2 and a query radius of 0.5.
For the dense and sparse cell arrays you must give the size of the cell as 
a command line argument.  For the example,
	CellArray.exe ../../../data/geom/mesh/33/cube.txt 0.5 0.1 0.1 0.1
the size of the cell is 0.1 x 0.1 x 0.1 and the search radius is 0.5.  
cell is the same size as each of the query windows.



