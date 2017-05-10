set size ratio 1 1, 1
set noborder
set noxtics
set noytics
set noztics
set nokey

set terminal postscript eps 22 color
set output "mesh_3.eps"
splot 'mesh_3.dat' with lines linewidth 2
