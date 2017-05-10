#! /usr/bin/env python

import argparse
from random import random

# Define the arguments and options.
parser = argparse.ArgumentParser(description='Generate a set of uniformly-distributed points.')
parser.add_argument('size', type=int, help='The number of points.')
parser.add_argument('output', type=argparse.FileType('w'),
                    help='The output file.')
args = parser.parse_args()

assert args.size > 0

args.output.write('%s\n' % args.size)
for i in range(args.size):
    args.output.write('%s %s %s\n' % (random(), random(), random()))
