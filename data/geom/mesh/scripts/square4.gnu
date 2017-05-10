set size square
#set size ratio 1
set noborder
set noxtics
set noytics
set nokey

#set title "Original, Mean C.N.=1.155" font "Courier,12"
set terminal pdf
set output "square4.pdf"
plot 'square4.dat' with lines
set terminal gif
set output "square4.gif"
plot 'square4.dat' with lines

#set title "1 Sweep, Mean C.N.=1.106" font "Courier,12"
set terminal pdf
set output "square4_1.pdf"
plot 'square4_1.dat' with lines
set terminal gif
set output "square4_1.gif"
plot 'square4_1.dat' with lines

#set title "5 Sweeps, Mean C.N.=1.035" font "Courier,12"
set terminal pdf
set output "square4_5.pdf"
plot 'square4_5.dat' with lines
set terminal gif
set output "square4_5.gif"
plot 'square4_5.dat' with lines

#set title "10 Sweeps, Mean C.N.=1.014" font "Courier,12"
set terminal pdf
set output "square4_10.pdf"
plot 'square4_10.dat' with lines
set terminal gif
set output "square4_10.gif"
plot 'square4_10.dat' with lines

#set title "100 Sweeps, Mean C.N.=1.008" font "Courier,12"
set terminal pdf
set output "square4_100.pdf"
plot 'square4_100.dat' with lines
set terminal gif
set output "square4_100.gif"
plot 'square4_100.dat' with lines

