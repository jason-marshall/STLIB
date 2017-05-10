set size square
#set size ratio sqrt(3)/2
set noborder
set noxtics
set noytics
set noztics
set nokey

#set view 60,30,1,1
set terminal pdf
set output "brick2.pdf"
splot 'brick2.dat' with lines
set terminal gif
set output "brick2.gif"
splot 'brick2.dat' with lines

set terminal pdf
set output "brick2d.pdf"
splot 'brick2d.dat' with lines
set terminal gif
set output "brick2d.gif"
splot 'brick2d.dat' with lines

set terminal pdf
set output "brick2d_1.pdf"
splot 'brick2d_1.dat' with lines
set terminal gif
set output "brick2d_1.gif"
splot 'brick2d_1.dat' with lines

set terminal pdf
set output "brick2d_2.pdf"
splot 'brick2d_2.dat' with lines
set terminal gif
set output "brick2d_2.gif"
splot 'brick2d_2.dat' with lines

set terminal pdf
set output "brick2d_3.pdf"
splot 'brick2d_3.dat' with lines
set terminal gif
set output "brick2d_3.gif"
splot 'brick2d_3.dat' with lines

set terminal pdf
set output "brick2d_4.pdf"
splot 'brick2d_4.dat' with lines
set terminal gif
set output "brick2d_4.gif"
splot 'brick2d_4.dat' with lines

