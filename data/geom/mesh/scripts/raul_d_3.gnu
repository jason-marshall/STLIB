#set size square
#set size ratio sqrt(3)/2
#set terminal pdf
set noborder
set noxtics
set noytics
set noztics
set nokey

set view 60,260,1,1
set terminal postscript color
set output "cylinder.eps"
splot 'raul_d_3.dat' with lines
set terminal gif
set output "cylinder.gif"
splot 'raul_d_3.dat' with lines
