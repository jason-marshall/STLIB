# decayingDimerizing.gnu

set title "Execution times for a decaying-dimerizing problem."
set xlabel "Number of reactions"
set ylabel "Fraction of serial execution time"
set key top

set logscale x
set terminal jpeg
set size 0.75
set output "decayingDimerizing.jpg"
plot [20:40000] [0:3] \
'decayingDimerizing.txt' using 1:2 title "Serial, 2 Processes" with linespoints, \
'decayingDimerizing.txt' using 1:3 title "Concurrent, 1 Thread" with linespoints, \
'decayingDimerizing.txt' using 1:4 title "Concurrent, 2 Threads" with linespoints 
set terminal postscript eps 22 color
set size 1
set output "decayingDimerizing.eps"
replot

