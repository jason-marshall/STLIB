"""Statistical functions."""

import numpy

def mean(listOfArrays):
    result = numpy.zeros(listOfArrays[0].shape)
    for x in listOfArrays:
        result += x
    result /= len(listOfArrays)
    return result

def meanStdDev(listOfArrays):
    n = len(listOfArrays)
    mean = numpy.zeros(listOfArrays[0].shape)
    stdDev = numpy.zeros(listOfArrays[0].shape)
    if n > 1:
        for x in listOfArrays:
            y = numpy.array(x, numpy.float64)
            mean += y
            stdDev += y * y
        mean /= n
        stdDev -= mean * mean * n
        stdDev = numpy.sqrt(stdDev / (n - 1))
    else:
        for x in listOfArrays:
            mean += x
        mean /= n
    return mean, stdDev

def main():
    data = [numpy.array([1, 2, 3]), numpy.array([2, 3, 5])]
    print('Data:')
    print(data)
    print('Mean:')
    print(mean(data))
    print('Mean and standard deviation:')
    print(meanStdDev(data))

if __name__ == '__main__':
    main()
