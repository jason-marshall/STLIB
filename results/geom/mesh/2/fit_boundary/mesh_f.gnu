set size ratio 1 0.5, 0.5
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "mesh_f.eps"
plot 'mesh_f.dat' with lines
