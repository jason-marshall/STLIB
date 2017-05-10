set size ratio 1 0.75, 0.75
set noborder
set noxtics
set noytics
set nokey

set terminal postscript eps 22 color
set output "mesh_gfgcc.eps"
plot 'mesh_gfgcc.dat' with lines
