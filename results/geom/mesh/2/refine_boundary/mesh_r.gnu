set size ratio 1
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "mesh_r.eps"
plot 'mesh_r.dat' with lines
