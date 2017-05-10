set size ratio 1
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "mesh_0.05.eps"
plot 'mesh_0.05.dat' with lines
