"""Generate trajectories for each solver."""

import sys, os, subprocess

def exitOnError():
    print('''Usage:
python Trajectories.py directory numberOfTests''')
    sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Error! Wrong number of arguments.')
        exitOnError()
    directory = sys.argv[1]
    numberOfTests = int(sys.argv[2])
    solver = open(directory + '/solver.txt', 'r').read().rstrip()
    model = open(directory + '/model.txt', 'r').read()
    state = open('initialState.txt', 'r').read()
    trajectories = [int(_x) for _x in 
                    open(directory + '/trajectories.txt', 'r').read().split()]
    command = '../../../../../applications/stochastic/solvers/' + solver + ' -p'
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
            #print(model + state + str(count) + '\n0\n')
            #process.wait()
            # Read the output.
            lines = process.stdout.readlines()
            # Save the histograms.
            time = lines[-1].rstrip()
            output.write(time + '\n')
            numberOfHistograms = int(lines[2])
            for j in range(numberOfHistograms):
                n = 14 + 4 * j
                output.write(''.join(lines[n:n+4]))
            # Record the MT state.
            n = 14 + 4 * numberOfHistograms
            state = lines[n]
