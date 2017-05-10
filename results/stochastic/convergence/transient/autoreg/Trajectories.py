"""Generate trajectories for each solver."""

import sys, os, subprocess

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
        # Save the histograms.
        time = lines[-1].rstrip()
        output.write(time + '\n')
        numberOfHistograms = int(lines[2])
        for j in range(numberOfHistograms):
            n = 14 + 4 * j
            lowerBound = float(lines[n])
            width = float(lines[n+1])
            bins = [float(_x) for _x in lines[n+2].split()]
            for i in range(len(bins)):
                if bins[i] != 0:
                    output.write('%s %s ' % (lowerBound + width * i, bins[i]))
            output.write('\n')
        # Record the MT state.
        n = 14 + 4 * numberOfHistograms
        state = lines[n]
    # Save the ending MT state.
    open(directory + '/state.txt', 'w').write(state)
