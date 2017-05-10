"""Generate states for the Mersenne Twister 19937."""

import sys

def generateState(seed):
    """Generate state for the Mersenne Twister 19937. Return a tuple of the
    state and the new seed. I copied this from Mt19937.py."""
    state = [0] * 625
    state[0] = seed & 0xffffffff
    for i in range(1, 624):
        state[i] = (1812433253 * (state[i-1] ^ (state[i-1] >> 30)) + i)
        state[i] &= 0xffffffff
    seed = (1812433253 * (state[623] ^ (state[623] >> 30)) + i)
    seed &= 0xffffffff
    # The position.
    state[-1] = 625
    # Return the array, the position, and the new seed.
    return (state, seed)

def exitOnError():
    print('''Usage:
python GenerateStates.py numberOfStates outputBaseName''')
    sys.exit(1)

if __name__ == '__main__':
    from math import log10

    if len(sys.argv) != 3:
        print('Error! Wrong number of arguments.')
        exitOnError()
    size = int(sys.argv[1])
    base = sys.argv[2]
    digits = int(log10(size))
    format = '%0' + str(digits) + 'd.txt'
    seed = 0
    for i in range(size):
        state, seed = generateState(seed)
        file = open(base + format % i, 'w')
        file.write(' '.join([str(_x) for _x in state]) + '\n')
