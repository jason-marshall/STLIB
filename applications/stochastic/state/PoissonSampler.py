"""Test the trajectory tree method in the context of empirically determining
the Poisson distribution by generating exponential deviates."""

import sys, math, random, copy
from Histogram import Histogram, histogramDistance

def logGammaStirling(x):
    """A few terms in Stirling's approximation of the gamma function."""
    return 0.5 * math.log(2 * math.pi) + (x - 0.5)*math.log(x) - x + 1./(12*x)\
        - 1./(160*x**3) + 1./(1260*x**5)
#print(math.exp(logGammaStirling(1)), 1)
#print(math.exp(logGammaStirling(2)), 1)
#print(math.exp(logGammaStirling(3)), 2)

def poissonPmf(mean, i):
    """Return the value of the Poisson PMF for specified mean and event.
    pmf = exp(-mean) mean^i / i!
    Calculate the value by returning the exponent of the logorithm of its value.
    log(pmf) = -mean + i log(mean) - log(Gamma(i+1))"""
    return math.exp(-mean+i*math.log(mean)-logGammaStirling(i+1))
#print(poissonPmf(10,0), math.exp(-10)*10**0/1)
#print(poissonPmf(10,1), math.exp(-10)*10**1/1)
#print(poissonPmf(10,2), math.exp(-10)*10**2/2)

def splittingTimes(mean, factor, initial=None):
    if not initial:
        initial = factor
    n = 1
    difference = initial
    times = []
    while difference < mean:
        times.append(mean - difference)
        difference *= factor
        n *= 2
    times.reverse()
    return times
#print(splittingTimes(mean, 2, 1))
#print(splittingTimes(mean, 2, 8))
#print(splittingTimes(mean, 3, 4))
#print(splittingTimes(mean, 4, 10000))
    
def generateSingle(h, mean):
    n = -1
    s = 0
    while s < mean:
        n += 1
        s += random.expovariate(1.)
    h.accumulate(n, 1)
    return n+1

def generate(h, mean, times):
    count = 0
    # (index, count, time)
    p = [(0,0,0)]
    while p:
        # Splitting.
        q = []
        for x in p:
            count += 1
            t = x[2] + random.expovariate(1.)
            multiplicity = 1
            i = x[0]
            while i < len(times) and t > times[i]:
                multiplicity *= 2
                i += 1
            for m in range(multiplicity):
                q.append((i, x[1]+1, t))
        p = q
        # Recording.
        q = []
        for x in p:
            if x[2] < mean:
                q.append(x)
            else:
                h.accumulate(x[1] - 1, 1)
        p = q
    return count

def generateFill(h, mean, difference):
    count = 0
    # (time, count, weight)
    p = [(0.,0,1.)]
    while p:
        # Advance.
        count += len(p)
        p = [(x[0]+random.expovariate(1.), x[1]+1, x[2]) for x in p]
        # Recording.
        q = []
        for x in p:
            if x[0] < mean:
                q.append(x)
            else:
                h.accumulate(x[1] - 1, x[2])
        p = q
        # Splitting.
        p.sort()
        q = []
        for i in range(len(p)):
            d = difference
            if i-1 >= 0:
                d = min(d, p[i][0] - p[i-1][0])
            if i+1 < len(p):
                d = min(d, p[i+1][0] - p[i][0])
            if d >= difference:
                for j in range(2):
                    q.append((p[i][0], p[i][1], 0.5*p[i][2]))
            else:
                q.append(p[i])
        p = q
    return count

def generateFillBias(h, mean, difference):
    """The bias seems to hurt performance."""
    count = 0
    # (time, count, weight)
    p = [(0.,0,1.)]
    while p:
        # Advance and split.
        count += len(p)
        p.sort()
        q = []
        for i in range(len(p)):
            d = difference
            if i-1 >= 0:
                d = min(d, p[i][0] - p[i-1][0])
            if i+1 < len(p):
                d = min(d, p[i+1][0] - p[i][0])
            x = p[i]
            if d >= difference:
                lo = 2
                while lo > 1:
                    lo = random.expovariate(1.)
                hi = 0
                while hi <= 1:
                    hi = random.expovariate(1.)
                q.append((x[0]+lo, x[1]+1, 0.5*x[2]))
                q.append((x[0]+hi, x[1]+1, 0.5*x[2]))
            else:
                q.append((x[0]+random.expovariate(1.), x[1]+1, x[2]))
        p = q
        # Recording.
        q = []
        for x in p:
            if x[0] < mean:
                q.append(x)
            else:
                h.accumulate(x[1] - 1, x[2])
        p = q
    return count

def generateFillTimes(h, mean, difference, times):
    times = copy.copy(times)
    count = 0
    # (time, count, weight)
    p = [(0.,0,1.)]
    while p:
        # Advance.
        count += len(p)
        p = [(x[0]+random.expovariate(1.), x[1]+1, x[2]) for x in p]
        # Recording.
        q = []
        for x in p:
            if x[0] < mean:
                q.append(x)
            else:
                h.accumulate(x[1] - 1, x[2])
        p = q
        # Splitting.
        p.sort()
        # If the maximum emperical time has stepped over a splitting time.
        if times and p and p[-1][0] > times[0]:
            if False and len(times) == 1:
                print(len(p))
                print(p)
            q = []
            for i in range(len(p)):
                d = difference
                if i-1 >= 0:
                    d = min(d, p[i][0] - p[i-1][0])
                if i+1 < len(p):
                    d = min(d, p[i+1][0] - p[i][0])
                if d >= difference:
                    for j in range(2):
                        q.append((p[i][0], p[i][1], 0.5*p[i][2]))
                else:
                    q.append(p[i])
            p = q
            del times[0]
    return count

def generateFillVariable(h, mean, splits):
    """splits is a list of pairs of times and differences."""
    recorded = []
    count = 0
    # (time, count, weight, index)
    p = [(0.,0,1.,0)]
    while p:
        # Advance.
        count += len(p)
        p = [(x[0]+random.expovariate(1.), x[1]+1, x[2], x[3]) for x in p]
        # Recording.
        q = []
        for x in p:
            if x[0] < mean:
                q.append(x)
            else:
                h.accumulate(x[1] - 1, x[2])
                recorded.append((x[1]-1, x[2]))
        p = q
        # Splitting.
        p.sort()
        q = []
        for i in range(len(p)):
            x = p[i]
            # If we have crossed a splitting time.
            #print(x)
            #print(splits)
            if x[3] < len(splits) and x[0] > splits[x[3]][0]:
                difference = splits[x[3]][1]
                d = difference
                if i-1 >= 0:
                    d = min(d, x[0] - p[i-1][0])
                if i+1 < len(p):
                    d = min(d, p[i+1][0] - x[0])
                if d >= difference:
                    for j in range(2):
                        q.append((x[0], x[1], 0.5*x[2], x[3]+1))
                else:
                    q.append((x[0], x[1], x[2], x[3]+1))
            else:
                q.append(x)
        p = q
    #print(recorded)
    return count


def generateExponential(h, mean):
    """Use the exponential distribution for the first event."""
    # exp(s - mean) > 1e-16
    # s - mean > log(1e-16)
    # s > mean + log(1e-16)
    threshold = mean + math.log(1e-16)
    n = 0
    s = 0
    while s < mean:
        # If the probablity is not negligible.
        if s > threshold:
            # P(t > mean - s)
            h.accumulate(n, math.exp(s - mean))
        n += 1
        s += random.expovariate(1.)
    return n

def complementaryErlang2(x):
    return math.exp(-x)*(1 + x)

def generateErlang2(h, mean):
    """Use the Erlang distribution with k = 2 for the first two events."""
    # CONTINUE Use exponential for first event.
    # CDF = 1 - exp(-x)(1 + x)
    # exp(-x)(1 + x) > 1e-16
    # -x + log(1+x) > log(1e-16)
    # x < 40
    # mean - s < 40
    # s > mean - 40
    p = math.exp(- mean)
    if p > 1e-16:
        h.accumulate(0, p)
    p = complementaryErlang2(mean)
    if p > 1e-16:
        h.accumulate(1, p)
    threshold = mean - 40
    n = 2
    s = 0
    r = random.expovariate(1.)
    while s + r < mean:
        x2 = mean - s
        x1 = x2 - r
        # If the probablity is not negligible.
        if s + r > threshold:
            # P(t > mean - s and t < mean - s + r)
            h.accumulate(n, complementaryErlang2(x1)*
                         (1 - complementaryErlang2(x2)))
        n += 1
        s += r
        r = random.expovariate(1.)
    return n-2

def complementaryErlang3(x):
    return math.exp(-x)*(1 + x + 0.5*x**2)

def generateErlang3(h, mean):
    """Use the Erlang distribution with k = 3 for the first three events."""
    # CDF = 1 - exp(-x)(1 + x + x^2/2)
    # exp(-x)(1 + x + x^2/2) > 1e-16
    # -x + log(1+x+x^2/2) > log(1e-16)
    # x < 44
    # mean - s < 44
    # s > mean - 44
    p = math.exp(- mean)
    if p > 1e-16:
        h.accumulate(0, p)
    p = complementaryErlang2(mean)
    if p > 1e-16:
        h.accumulate(1, p)
    p = complementaryErlang3(mean)
    if p > 1e-16:
        h.accumulate(2, p)
    threshold = mean - 44
    n = 3
    s = 0
    r = random.expovariate(1.)
    while s + r < mean:
        x2 = mean - s
        x1 = x2 - r
        # If the probablity is not negligible.
        if s + r > threshold:
            # P(t > mean - s and t < mean - s + r)
            h.accumulate(n, complementaryErlang3(x1)*
                         (1 - complementaryErlang3(x2)))
        n += 1
        s += r
        r = random.expovariate(1.)
    return n-3

# Table[1./n!, {n, 0, 19}] // CForm
InverseFactorial =\
    [1.,1.,0.5,0.16666666666666666,0.041666666666666664,
     0.008333333333333333,0.001388888888888889,0.0001984126984126984,
     0.0000248015873015873,2.7557319223985893e-6,2.755731922398589e-7,
     2.505210838544172e-8,2.08767569878681e-9,1.6059043836821613e-10,
     1.1470745597729725e-11,7.647163731819816e-13,4.779477332387385e-14,
     2.8114572543455206e-15,1.5619206968586225e-16,8.22063524662433e-18]

# Table[t /. FindRoot[Exp[-t] (Sum[t^i/i!, {i, 0, k - 1}]) == 1*^-16, {t, 40}], {k, 1, 20}] // CForm
Thresholds =\
    [36.841361487904734,40.56870918918821,43.75093834832743,
     46.64209125895275,49.341017193401534,51.899551658175255,
     54.349219528219265,56.711013295457896,58.99980237638767,
     61.226603512293565,63.39986383309912,65.52623797381753,
     67.61108475952383,69.65879862046755,71.67303867372961,
     73.65689180840498,75.61299172955044,77.54360774440592,
     79.45071223214435,81.33603276390895]

def complementaryErlang(n, x):
    return math.exp(-x)*sum([InverseFactorial[i]*x**i for i in range(n)])

def generateErlang(k, h, mean):
    """Use the Erlang distribution for the first k events."""
    for i in range(k):
        p = complementaryErlang(i, mean)
        if p > 1e-16:
            h.accumulate(i, p)
    threshold = mean - Thresholds[k-1]
    n = k
    s = 0
    r = random.expovariate(1.)
    while s + r < mean:
        x2 = mean - s
        x1 = x2 - r
        # If the probablity is not negligible.
        if s + r > threshold:
            # P(t > mean - s and t < mean - s + r)
            h.accumulate(n, complementaryErlang(k, x1)*
                         (1 - complementaryErlang(k, x2)))
        n += 1
        s += r
        r = random.expovariate(1.)
    return n-k

if __name__ == '__main__':
    mean = 100
    # Standard deviations from the mean.
    stdDev = 5
    #stdDev = 10
    lower = max(0, int(mean - stdDev * math.sqrt(mean)))
    upper = int(mean + stdDev * math.sqrt(mean))
    size = upper - lower + 1

    pmf = Histogram(size, 1)
    pmf.lowerBound = float(lower)
    for i in range(size):
        pmf.histograms[0][i] = poissonPmf(mean, lower + i)

    #print(pmf)
    tests = 20
    h = Histogram(size, 1)

    parameters = range(1,21)
    for k in parameters:
        count = 0
        for i in range(tests):
            h.clear()
            h.setCurrent(0)
            while (histogramDistance(h, pmf) > 0.1):
                count += generateErlang(k, h, mean)
        print(k, count / float(tests))

    sys.exit(0)

    count = 0
    for i in range(tests):
        h.clear()
        while (histogramDistance(h, pmf) > 0.1):
            count += generateErlang3(h, mean)
    print(count / float(tests))


    count = 0
    for i in range(tests):
        h.clear()
        while (histogramDistance(h, pmf) > 0.1):
            count += generateErlang2(h, mean)
    print(count / float(tests))

    count = 0
    for i in range(tests):
        h.clear()
        while (histogramDistance(h, pmf) > 0.1):
            count += generateExponential(h, mean)
    print(count / float(tests))

    count = 0
    for i in range(tests):
        h.clear()
        while (histogramDistance(h, pmf) > 0.1):
            count += generateSingle(h, mean)
    print(count / float(tests))

    #print(histogramDistance(h, pmf))
    #p = h.getProbabilities()
    #print('{' + ','.join(['%f' % _x for _x in p]) + '}')
    #p = pmf.getProbabilities()
    #print('{' + ','.join(['%f' % _x for _x in p]) + '}')


    count = 0
    #splits = [(100-x,math.sqrt(x)) for x in [1,4,16]]
    splits = [(100-x,1) for x in [1,2,4,8,16]]
    splits.reverse()
    for i in range(tests):
        h.clear()
        while (histogramDistance(h, pmf) > 0.1):
            count += generateFillVariable(h, mean, splits)
    print(count / float(tests), splits)

    sys.exit(0)

    count = 0
    differences = [1,2,4,8]
    factors = [2,4]
    initials = range(1,2)
    for difference in differences:
        for factor in factors:
            for initial in initials:
                times = splittingTimes(mean, factor, initial)
                for i in range(tests):
                    h.clear()
                    while (histogramDistance(h, pmf) > 0.1):
                        count += generateFillTimes(h, mean, difference, times)
                print(count / float(tests), difference, factor, initial)


    count = 0
    differences = [2,3,4,10]
    differences.reverse()
    for difference in differences:
        for i in range(tests):
            h.clear()
            while (histogramDistance(h, pmf) > 0.1):
                count += generateFill(h, mean, difference)
        print(count / float(tests), difference)


    count = 0
    #timesList = [range(95,100), range(90,100,2), range(80,100,4)]
    #timesList = [range(90,100), range(10,100,10)]
    timesList = []
    for times in timesList:
        for i in range(tests):
            h.clear()
            while (histogramDistance(h, pmf) > 0.1):
                count += generate(h, mean, times)
        print(count / float(tests), times)


    #factors = range(2,5)
    #initials = range(1,5)
    factors = range(2,5)
    initials = range(1,2)
    for factor in factors:
        for initial in initials:
            times = splittingTimes(mean, factor, initial)
            count = 0
            for i in range(tests):
                h.clear()
                while (histogramDistance(h, pmf) > 0.1):
                    count += generate(h, mean, times)
            print(factor, initial, count / float(tests), times)


    factors = [1.5, 1.6, 1.7, 1.8, 1.9]
    factors.reverse()
    for factor in factors:
        times = splittingTimes(mean, factor)
        count = 0
        for i in range(tests):
            h.clear()
            while (histogramDistance(h, pmf) > 0.1):
                count += generate(h, mean, times)
        print(factor, count / float(tests), times)

