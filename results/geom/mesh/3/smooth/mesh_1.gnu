set size ratio 1 1, 1
set noborder
set noxtics
set noytics
set noztics
set nokey

set terminal postscript eps 22 color
set output "mesh_1.eps"
splot 'mesh_1.dat' with lines linewidth 2
