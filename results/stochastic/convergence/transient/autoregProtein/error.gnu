# Plot the errors.

set title "Auto-regulatory Network"
set xlabel "Execution time (s)"
set ylabel "Error"
set key top

set logscale
set terminal jpeg
set output "error.jpg"
#plot [1:10000] [0.01:1]
plot [1:1000]\
     'traditional/error.txt' title "Traditional" with linespoints,\
     'multi16/error.txt' title "Multi-time 16" with linespoints

#     'multi2/error.txt' title "Multi-time 2" with linespoints,\
     'multi4/error.txt' title "Multi-time 4" with linespoints,\
     'multi8/error.txt' title "Multi-time 8" with linespoints,\
     'multi32/error.txt' title "Multi-time 32" with linespoints
#set terminal postscript eps 22 color
#set output "error.eps"
#replot
