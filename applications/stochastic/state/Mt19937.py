"""Implements functions for the Mersenne Twister 19937."""

def generateState(seed):
    """Generate state for the Mersenne Twister 19937. Return a tuple of the
    state and the new seed. This function is adapted from 
    DiscreteUniformGeneratorMt19937::generateState()."""
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
