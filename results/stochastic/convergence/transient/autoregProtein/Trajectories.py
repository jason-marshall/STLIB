"""Generate trajectories for each solver."""

import sys, os, subprocess
#sys.path.append('../../../../../applications/stochastic/state')
#from Mt19937 import generateState

def exitOnError():
    print('''Usage:
python Trajectories.py directory numberOfTrajectories''')
    sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Error! Wrong number of arguments.')
        exitOnError()
    directory = sys.argv[1]
    size = int(sys.argv[2])
    solver = open(directory + '/solver.txt', 'r').read().rstrip()
    model = open(directory + '/model.txt', 'r').read()
    state = open(directory + '/state.txt', 'r').read()
    output = open(directory + '/output.txt', 'a')
    for i in range(size):
        # Run the simulation.
        command = '../../../../../applications/stochastic/solvers/' + solver +\
            ' -p'
        process = subprocess.Popen(command,
                                   #bufsize=-1,
                                   #universal_newlines=True,
                                   shell=True,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
        # The input.
        process.stdin.write(model + state + '1\n0\n')
        process.stdin.flush()
        process.wait()
        # Read the output.
        lines = process.stdout.readlines()
        # Save the histogram. For now I assume there is only one.
        lowerBound = float(lines[14])
        width = float(lines[15])
        bins = [float(_x) for _x in lines[16].split()]
        time = lines[-1].rstrip()
        output.write(time + ' ')
        for i in range(len(bins)):
            if bins[i] != 0:
                output.write('%s %s ' % (lowerBound + width * i, bins[i]))
        output.write('\n')
        # Record the MT state.
        state = lines[18]
    # Save the ending MT state.
    open(directory + '/state.txt', 'w').write(state)
