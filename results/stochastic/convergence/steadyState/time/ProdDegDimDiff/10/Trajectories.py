"""Generate trajectories for each solver."""

import sys, os, subprocess

def exitOnError():
    print('''Usage:
python Trajectories.py directory numberOfTests''')
    sys.exit(1)

if len(sys.argv) != 3:
    print('Error! Wrong number of arguments.')
    exitOnError()
directory = sys.argv[1]
numberOfTests = int(sys.argv[2])
solver = open(directory + '/solver.txt', 'r').read().rstrip()
model = open('model.txt', 'r').read()
state = open('initialState.txt', 'r').read()
trajectories = [int(_x) for _x in 
                open(directory + '/trajectories.txt', 'r').read().split()]
command = '../../../../../../../applications/stochastic/solvers/' + solver\
          + ' -p'
output = open(directory + '/output.txt', 'w')
# The total number of tests.
output.write(str(len(trajectories) * numberOfTests) + '\n')
for count in trajectories:
    print(count)
    for i in range(numberOfTests):
        # Run the simulation.
        process = subprocess.Popen(command,
                                   bufsize=-1,
                                   universal_newlines=True,
                                   shell=True,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
        # The input.
        process.stdin.write(model + state + str(count) + '\n0\n')
        process.stdin.flush()
        # Read the output.
        lines = process.stdout.readlines()
        # Write the execution time.
        output.write(lines[-1])
        # Write each of the histograms.
        for species in range(10):
            begin = 5 + 7 * species
            output.write(''.join(lines[begin:begin+7]))
        # Record the MT state for the next simulation.
        state = lines[-6]
