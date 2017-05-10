# make.py

import MT19937
import Death, Immigration, Birth, DecayingDimerizing
import math

def deathConstant(size, initialAmount):
    file = open('Death' + str(size) + 'Constant.txt', 'w')
    file.write('%s' % Death.output(size, initialAmount, lambda n: 1))
    # number of frames
    file.write('1\n')
    # list of frame times
    file.write('1000\n')
    # maximum allowed steps
    file.write('0\n')
    # list of MT 19937 state
    file.write('%s' % MT19937.output())
    # number of trajectories
    file.write('1\n')

def deathGeometric(size, initialAmount, factor):
    file = open('Death' + str(size) + 'Geometric' + str(factor) + '.txt', 'w')
    file.write('%s' % Death.output(size, initialAmount, 
                                   lambda n: factor**(float(n)/max(size-1,1))))
    # number of frames
    file.write('1\n')
    # list of frame times
    file.write('1000\n')
    # maximum allowed steps
    file.write('0\n')
    # list of MT 19937 state
    file.write('%s' % MT19937.output())
    # number of trajectories
    file.write('1\n')

def death():
    # Adjust the number of reactions and the initial amounts so the same number
    # of reactions fire.
    deathConstant(1, 1000000)
    deathConstant(10, 100000)
    deathConstant(100, 10000)
    deathConstant(1000, 1000)
    deathConstant(10000, 100)

    deathGeometric(1, 1000000, 10)
    deathGeometric(10, 100000, 10)
    deathGeometric(100, 10000, 10)
    deathGeometric(1000, 1000, 10)
    deathGeometric(10000, 100, 10)

    deathGeometric(1, 1000000, 100)
    deathGeometric(10, 100000, 100)
    deathGeometric(100, 10000, 100)
    deathGeometric(1000, 1000, 100)
    deathGeometric(10000, 100, 100)

    deathGeometric(1, 1000000, 1000)
    deathGeometric(10, 100000, 1000)
    deathGeometric(100, 10000, 1000)
    deathGeometric(1000, 1000, 1000)
    deathGeometric(10000, 100, 1000)

def immigrationConstant(size):
    file = open('Immigration' + str(size) + 'Constant.txt', 'w')
    file.write('%s' % Immigration.output(size, 0, 
                                         lambda n: 1. / size))
    # number of frames
    file.write('1\n')
    # list of frame times
    file.write('1000000\n')
    # maximum allowed steps
    file.write('0\n')
    # list of MT 19937 state
    file.write('%s' % MT19937.output())
    # number of trajectories
    file.write('1\n')

def immigrationGeometric(size, factor):
    file = open('Immigration' + str(size) + 'Geometric' + str(factor) +
                '.txt', 'w')
    sum = 0
    for n in range(size):
        sum += factor**(float(n)/max(size-1,1))
    file.write('%s' % Immigration.output(size, 0, 
                                         lambda n: factor**(float(n)/max(size-1,1)) / sum))
    # number of frames
    file.write('1\n')
    # list of frame times
    file.write('1000000\n')
    # maximum allowed steps
    file.write('0\n')
    # list of MT 19937 state
    file.write('%s' % MT19937.output())
    # number of trajectories
    file.write('1\n')

def immigration():
    # Adjust the number of reactions and the initial amounts so the same number
    # of reactions fire.
    immigrationConstant(1)
    immigrationConstant(10)
    immigrationConstant(100)
    immigrationConstant(1000)
    immigrationConstant(10000)

    immigrationGeometric(1, 10)
    immigrationGeometric(10, 10)
    immigrationGeometric(100, 10)
    immigrationGeometric(1000, 10)
    immigrationGeometric(10000, 10)

    immigrationGeometric(1, 100)
    immigrationGeometric(10, 100)
    immigrationGeometric(100, 100)
    immigrationGeometric(1000, 100)
    immigrationGeometric(10000, 100)

    immigrationGeometric(1, 1000)
    immigrationGeometric(10, 1000)
    immigrationGeometric(100, 1000)
    immigrationGeometric(1000, 1000)
    immigrationGeometric(10000, 1000)

def birthConstant(size):
    # dx/dt = 

    # n is the size.
    # p is the propensity factor.
    # t is the time.
    # n * 2^(p t) = 2^t
    # n = 2^(t (1 - p))
    # log_2 n = t (1 - p)
    # (1/t) log_2 n = (1 - p)
    # p = 1 - (1/t) log_2 n 
    #p = 1 - (1./15.) * math.log(size) / math.log(2)

    # n 2^t = 2^15
    # n = 2^(15 - t)
    # log_2 n = 15 - t
    # t = 15 - log_2 n

    file = open('Birth' + str(size) + 'Constant.txt', 'w')
    # Propensity factors sum to unity.
    file.write('%s' % Birth.output(size, 1, lambda n: 1))
    # number of frames
    file.write('1\n')
    # list of frame times
    file.write('%f\n' % (15 - math.log(size) / math.log(2)))
    # maximum allowed steps
    file.write('0\n')
    # list of MT 19937 state
    file.write('%s' % MT19937.output())
    # number of trajectories
    file.write('1\n')

def birthGeometric(size, factor):
    file = open('Birth' + str(size) + 'Geometric' + str(factor) + '.txt', 'w')
    # Propensity factors sum to unity.
    sum = 0
    for n in range(size):
        sum += factor**(float(n)/max(size-1,1))
    file.write('%s' % Birth.output(size, 1, 
                                   lambda n: factor**(float(n)/max(size-1,1)) / sum))
    # number of frames
    file.write('1\n')
    # list of frame times
    file.write('1000\n')
    # maximum allowed steps
    file.write('0\n')
    # list of MT 19937 state
    file.write('%s' % MT19937.output())
    # number of trajectories
    file.write('1\n')

def birth():
    birthConstant(1)
    birthConstant(10)
    birthConstant(100)
    birthConstant(1000)
    birthConstant(10000)

    birthGeometric(1, 10)
    birthGeometric(10, 10)
    birthGeometric(100, 10)
    birthGeometric(1000, 10)
    birthGeometric(10000, 10)

    birthGeometric(1, 100)
    birthGeometric(10, 100)
    birthGeometric(100, 100)
    birthGeometric(1000, 100)
    birthGeometric(10000, 100)

    birthGeometric(1, 1000)
    birthGeometric(10, 1000)
    birthGeometric(100, 1000)
    birthGeometric(1000, 1000)
    birthGeometric(10000, 1000)

def decayingDimerizingConstant(size):
    file = open('DecayingDimerizing' + str(size) + '.txt', 'w')
    file.write('%s' % DecayingDimerizing.output(size))
    # number of frames
    file.write('1\n')
    # list of frame times
    file.write('30\n')
    # maximum allowed steps
    file.write('0\n')
    # list of MT 19937 state
    file.write('%s' % MT19937.output())
    # number of trajectories
    file.write('1\n')

def decayingDimerizing():
    decayingDimerizingConstant(1)
    decayingDimerizingConstant(10)
    decayingDimerizingConstant(100)
    decayingDimerizingConstant(1000)
    decayingDimerizingConstant(10000)

if __name__ == '__main__':
    death()
    immigration()
    birth()
    decayingDimerizing()
