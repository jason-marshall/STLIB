#! /usr/bin/env python

# qsubScripts.py
# Write qsub scripts.

import sys
import os.path

if len(sys.argv) != 1:
    print("Usage:")
    print("python qsubScripts.py\n")
    raise "Bad command line arguments. Exiting..."

processesPerNode = 16
workingDir = os.path.abspath('.')
stlibDir = '~/Dev'

# For each cell type.
for cell in ("u", "m"):
    # The number of processes is 1 << shift.
    for shift in range(10):
        numProcesses = 1 << shift
        numMinutes = 2
        numNodes = (numProcesses + processesPerNode - 1) / processesPerNode;

        f = open(cell + "_%03d" % numProcesses + ".sh", "w")
        f.write("""#!/bin/sh
### PBS ###
#PBS -N "%s_%03d"
#PBS -lwalltime=00:%02d:00
#PBS -lnodes=%d:ppn=16
#PBS -j oe
cd %s
mpirun -np %d %s/stlib/test/performance/release/sfc/mpi/partition -c=%s -o=1048576 > %s_%03d.txt
""" % (cell, numProcesses, numMinutes, numNodes, workingDir,
       numProcesses, stlibDir, cell, cell, numProcesses))
        f.close()

    f = open("submit_" + cell + ".sh", "w")
    for shift in range(10):
        numProcesses = 1 << shift
        f.write("qsub " + cell + "_%03d.sh\n" % numProcesses)
    f.close()
