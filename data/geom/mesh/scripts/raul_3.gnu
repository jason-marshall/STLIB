set size square
#set size ratio sqrt(3)/2
#set terminal pdf
set noborder
set noxtics
set noytics
set noztics
set nokey

#set title "Original, M.C.N.=3.55" font "Courier,14"
#set view 60,30,1,1
#set output "raul_3.pdf"
splot 'raul_3.dat' with lines

#set title "Distorted, M.C.N.=3.6e+14" font "Courier,14"
#set output "brick2d.pdf"
#plot 'brick2d.dat' with lines

pause -1 "Hit return to exit."