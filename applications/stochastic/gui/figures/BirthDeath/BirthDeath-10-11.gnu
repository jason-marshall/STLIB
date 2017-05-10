set size 0.5, 0.5
set title ""
set xlabel "Time"
set ylabel "Population"
set key top
set terminal jpeg
set output "BirthDeath-10-11-Populations.jpg"
plot \
"BirthDeath-10-11.dat" using 1:3 title "X" with lines

set title ""
set xlabel "Time"
set ylabel "X"
set key off
set terminal jpeg
set output "BirthDeath-10-11-X.jpg"
plot \
"BirthDeath-10-11.dat" using 1:3 title "X" with lines

set ylabel "Reaction Counts"
set key left
set output "BirthDeath-10-11-Reactions.jpg"
plot \
"BirthDeath-10-11.dat" using 1:4 title "Birth" with lines,\
"BirthDeath-10-11.dat" using 1:5 title "Death" with lines
set ylabel "Birth"
set key off
set output "BirthDeath-10-11-Birth.jpg"
plot \
"BirthDeath-10-11.dat" using 1:4 title "Birth" with lines

set ylabel "Death"
set key off
set output "BirthDeath-10-11-Death.jpg"
plot \
"BirthDeath-10-11.dat" using 1:5 title "Death" with lines

