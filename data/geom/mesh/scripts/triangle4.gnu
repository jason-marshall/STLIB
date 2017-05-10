#set size square
set size ratio sqrt(3)/2
set noborder
set noxtics
set noytics
set nokey

#set title "Original, Max C.N.=1" font "Courier,14"
set terminal pdf
set output "triangle4.pdf"
plot 'triangle4.dat' with lines
set terminal gif
set output "triangle4.gif"
plot 'triangle4.dat' with lines

#set title "Distorted, Max C.N.=2.4e+11" font "Courier,14"
set terminal pdf
set output "triangle4d.pdf"
plot 'triangle4d.dat' with lines
set terminal gif
set output "triangle4d.gif"
plot 'triangle4d.dat' with lines

#set title "1 Sweep, Max C.N.=2.4e+11" font "Courier,14"
set terminal pdf
set output "triangle4d_1.pdf"
plot 'triangle4d_1.dat' with lines
set terminal gif
set output "triangle4d_1.gif"
plot 'triangle4d_1.dat' with lines

#set title "2 Sweeps, Max C.N.=2.49" font "Courier,14"
set terminal pdf
set output "triangle4d_2.pdf"
plot 'triangle4d_2.dat' with lines
set terminal gif
set output "triangle4d_2.gif"
plot 'triangle4d_2.dat' with lines

#set title "3 Sweeps, Max C.N.=1.14" font "Courier,14"
set terminal pdf
set output "triangle4d_3.pdf"
plot 'triangle4d_3.dat' with lines
set terminal gif
set output "triangle4d_3.gif"
plot 'triangle4d_3.dat' with lines

#set title "4 Sweeps, Max C.N.=1.03" font "Courier,14"
set terminal pdf
set output "triangle4d_4.pdf"
plot 'triangle4d_4.dat' with lines
set terminal gif
set output "triangle4d_4.gif"
plot 'triangle4d_4.dat' with lines
