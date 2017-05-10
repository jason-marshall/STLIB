set size ratio 1 0.75, 0.75
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "mesh_gfgc.eps"
plot 'mesh_gfgc.dat' with lines
