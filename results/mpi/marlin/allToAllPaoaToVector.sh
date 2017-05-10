#!/bin/sh
### PBS ###
#PBS -N "allToAll"
#PBS -lwalltime=00:10:00
#PBS -lnodes=32:ppn=16
#PBS -j oe
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 1 > /HOMES/$USER/Dev/stlib/results/mpi/marlin/001.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 2 > /HOMES/$USER/Dev/stlib/results/mpi/marlin/002.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 4 > /HOMES/$USER/Dev/stlib/results/mpi/marlin/004.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 8 > /HOMES/$USER/Dev/stlib/results/mpi/marlin/008.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 16 > /HOMES/$USER/Dev/stlib/results/mpi/marlin/016.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 32 > /HOMES/$USER/Dev/stlib/results/mpi/marlin/032.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 64 > /HOMES/$USER/Dev/stlib/results/mpi/marlin/064.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 128 > /HOMES/$USER/Dev/stlib/results/mpi/marlin/128.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 256 > /HOMES/$USER/Dev/stlib/results/mpi/marlin/256.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 512 > /HOMES/$USER/Dev/stlib/results/mpi/marlin/512.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 1 -n > /HOMES/$USER/Dev/stlib/results/mpi/marlin/n001.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 2 -n > /HOMES/$USER/Dev/stlib/results/mpi/marlin/n002.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 4 -n > /HOMES/$USER/Dev/stlib/results/mpi/marlin/n004.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 8 -n > /HOMES/$USER/Dev/stlib/results/mpi/marlin/n008.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 16 -n > /HOMES/$USER/Dev/stlib/results/mpi/marlin/n016.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 32 -n > /HOMES/$USER/Dev/stlib/results/mpi/marlin/n032.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 64 -n > /HOMES/$USER/Dev/stlib/results/mpi/marlin/n064.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 128 -n > /HOMES/$USER/Dev/stlib/results/mpi/marlin/n128.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 256 -n > /HOMES/$USER/Dev/stlib/results/mpi/marlin/n256.txt
mpirun -np 512 /HOMES/$USER/Dev/stlib/test/performance/release/mpi/allToAllPaoaToVector -o 10000 -s 512 -n > /HOMES/$USER/Dev/stlib/results/mpi/marlin/n512.txt
