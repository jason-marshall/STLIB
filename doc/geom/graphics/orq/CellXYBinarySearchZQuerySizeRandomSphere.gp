# CellXYBinarySearchZQuerySizeRandomSphere.gp

set size 0.7,0.7
set key right bottom
set title 'Cell Array with Binary Searching, Random Points on a Sphere'
set xlabel 'Query Size'
set ylabel 'Execution Time (s)'
set terminal postscript eps color
set logscale xy
set xtics 0.0025,4,2
set nomxtics
set ytics 0.00001,10,0.1
set nomytics
set output 'CellXYBinarySearchZQuerySizeRandomSphereTotal.eps'
plot [0.00138107/1.1:2*1.1]\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 1:3 title "0.01      " with linespoints,\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 1:4 title "0.01414" with linespoints,\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 1:5 title "0.02      " with linespoints,\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 1:6 title "0.02828" with linespoints,\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 1:7 title "0.04      " with linespoints,\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 1:8 title "0.05657" with linespoints,\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 1:9 title "0.08      " with linespoints,\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 1:10 title "Sequential Scan" with linespoints

set key right top
set xlabel 'Number of Reported Records'
set ylabel 'Time Per Reported Record (microsec)'
set xtics 1,10,100000
set ytics 1,2,1024
set output 'CellXYBinarySearchZQuerySizeRandomSphereScaled.eps'
plot [1.26/1.1:376226.*1.1]\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 2:($3/$2*10e6) title "0.01      " with linespoints,\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 2:($4/$2*10e6) title "0.01414" with linespoints,\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 2:($5/$2*10e6) title "0.02      " with linespoints,\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 2:($6/$2*10e6) title "0.02828" with linespoints,\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 2:($7/$2*10e6) title "0.04      " with linespoints,\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 2:($8/$2*10e6) title "0.05657" with linespoints,\
  'CellXYBinarySearchZQuerySizeRandomSphere.dat'\
  using 2:($9/$2*10e6) title "0.08      " with linespoints
