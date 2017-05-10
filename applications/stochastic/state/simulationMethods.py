"""A list of the simulation methods."""

# [['category',
#   [['method', hasGeneric, hasCustom, hasPython,
#     [['options', 'executable', 'parameterName', 'parameterValue',
#       'parameterToolTip'],
#      ...]], ...]]]
# The first method is the default. The first option for each method is the
# default.

_homogeneousTimeSeriesUniform_direct =\
    ['Direct', True, True, False,
     [['2-D search', 'HomogeneousDirect2DSearch', '', '', ''],
      ['2-D search, sorted', 'HomogeneousDirect2DSearchSorted', '', '', ''],
      ['2-D search, bubble sort', 'HomogeneousDirect2DSearchBubbleSort', '', '', ''],
      ['Composition rejection', 'HomogeneousDirectCompositionRejection', '', '',
       ''],
      ['Binary search, full CDF', 'HomogeneousDirectBinarySearch', '', '', ''],
      ['Binary search, sorted CDF', 'HomogeneousDirectBinarySearchSorted', '',
       '', ''],
      ['Binary search, recursive CDF',
       'HomogeneousDirectBinarySearchRecursiveCdf', '', '', ''],
      ['Linear search', 'HomogeneousDirectLinearSearch', '', '', ''],
      ['Linear search, delayed update',
       'HomogeneousDirectLinearSearchDelayedUpdate', '', '', ''],
      ['Linear search, sorted', 'HomogeneousDirectLinearSearchSorted', '', '',
       ''],
      ['Linear search, bubble sort', 'HomogeneousDirectLinearSearchBubbleSort',
       '', '', '']]
     ]
_homogeneousTimeSeriesUniform_next =\
    ['Next Reaction', True, True, False,
     [['Hashing', 'HomogeneousNextReactionHashing', '', '', ''],
      ['Binary heap, pointer', 'HomogeneousNextReactionBinaryHeapPointer', '',
       '', ''],
      ['Binary heap, pair', 'HomogeneousNextReactionBinaryHeapPair', '', '',
       ''],
      ['Partition', 'HomogeneousNextReactionPartitionCostAdaptive', '', '', ''],
      ['Linear search', 'HomogeneousNextReactionLinearSearchUnrolled', '', '',
       '']]
     ]
_homogeneousTimeSeriesUniform_first =\
    ['First Reaction', True, True, False,
     [['Simple', 'HomogeneousFirstReaction', '', '', ''],
      ['Reaction influence', 'HomogeneousFirstReactionInfluence', '', '', ''],
      ['Absolute time', 'HomogeneousFirstReactionAbsoluteTime', '', '', '']]
     ]
_homogeneousTimeSeriesUniform_tau =\
    ['Tau-Leaping', True, True, False,
     [['Runge-Kutta, 4th order', 'HomogeneousTauLeapingRungeKutta4',
       'Error:', '0.01', 'The allowed error in a step.'],
      ['Midpoint', 'HomogeneousTauLeapingMidpoint', 'Error:', '0.01',
       'The allowed error in a step.'],
      ['Forward', 'HomogeneousTauLeapingForward', 'Error:', '0.01',
       'The allowed error in a step.'],
      ['Runge-Kutta, 4th order, no correction',
       'HomogeneousTauLeapingRungeKutta4NoCorrection', 'Error:', '0.01',
       'The allowed error in a step.'],
      ['Midpoint, no correction', 'HomogeneousTauLeapingMidpointNoCorrection',
       'Error:', '0.01', 'The allowed error in a step.'],
      ['Forward, no correction', 'HomogeneousTauLeapingForwardNoCorrection',
       'Error:', '0.01', 'The allowed error in a step.'],
      ['Runge-Kutta, 4th order, fixed step size',
       'HomogeneousTauLeapingFixedRungeKutta4', 'Step size:', '0.01',
       'The time step.'],
      ['Midpoint, fixed step size', 'HomogeneousTauLeapingFixedMidpoint',
       'Step size:', '0.01', 'The time step.'],
      ['Forward, fixed step size', 'HomogeneousTauLeapingFixedForward',
       'Step size:', '0.01', 'The time step.']]
     ]
_homogeneousTimeSeriesUniform_tauImplicit =\
    ['Implicit Tau-Leaping', True, False, False,
     [['Euler, fixed step size', 'HomogeneousTauLeapingImplicitFixedEuler',
       'Step size:', '0.01', 'The time step.']]
     ]
_homogeneousTimeSeriesUniform_hybrid =\
    ['Hybrid Direct/Tau-Leaping', True, True, False,
     [['Runge-Kutta, 4th order', 'HomogeneousHybridDirectTauLeapingRungeKutta4',
       'Error:', '0.01', 'The allowed error in a step.'],
      ['Midpoint', 'HomogeneousHybridDirectTauLeapingMidpoint', 'Error:',
       '0.01', 'The allowed error in a step.'],
      ['Forward', 'HomogeneousHybridDirectTauLeapingForward', 'Error:', '0.01',
       'The allowed error in a step.']]
     ]
_homogeneousTimeSeriesUniform =\
    ['Time Series, Uniform',
     [_homogeneousTimeSeriesUniform_direct, _homogeneousTimeSeriesUniform_next,
      _homogeneousTimeSeriesUniform_first, _homogeneousTimeSeriesUniform_tau,
      _homogeneousTimeSeriesUniform_tauImplicit,
      _homogeneousTimeSeriesUniform_hybrid]
     ]

_homogeneousTimeSeriesAllReactions =\
    ['Time Series, All Reactions',
     [['Direct', True, True, False,
       [['2-D search', 'HomogeneousDirectAllReactions2DSearch', '', '', '']]
       ]]
     ]

_homogeneousTimeSeriesDeterministic =\
    ['Time Series, Deterministic',
     [['ODE, Integrate Reactions', True, True, False,
       [['Runge-Kutta, Cash-Karp', 'HomogeneousOdeReactionRungeKuttaCashKarp',
         'Error:', '1e-8', 'The allowed error in a step.'],
        ['Runge-Kutta, Cash-Karp, fixed step size',
         'HomogeneousOdeReactionFixedRungeKuttaCashKarp', 'Step size:', '0.01',
         'The time step.'],
        ['Runge-Kutta, 4th order, fixed step size',
         'HomogeneousOdeReactionFixedRungeKutta4', 'Step size:', '0.01',
         'The time step.'],
        ['Midpoint, fixed step size', 'HomogeneousOdeReactionFixedMidpoint',
         'Step size:', '0.01', 'The time step.'],
        ['Forward, fixed step size', 'HomogeneousOdeReactionFixedForward',
         'Step size:', '0.01', 'The time step.']
        ]],
      ['Mathematica', False, False, False,
       [['NDSolve', 'MathematicaNDSolve', '', '', '']]
       ]
      ]
     ]

_homogeneousHistogramsTransient =\
    ['Histograms, Transient Behavior',
     [['Direct', True, True, False,
       [['Standard', 'HomogeneousHistogramsDirect2DSearch', '', '', ''],
        ['Tree Exp. Last',
         'HomogeneousHistogramsTransientDirectTreeExponentialLast',
         '', '', ''],
        ['Tree Exp. Limit',
         'HomogeneousHistogramsTransientDirectTreeExponentialLimit',
         '', '', ''],
        ['Tree Hypoexp. Limit',
         'HomogeneousHistogramsTransientDirectTreeHypoexponentialLimit',
         '# of parameters:', '10', 'The number of parameters in the hypoexponential distribution.'],
        ['Tree Normal Approx.',
         'HomogeneousHistogramsTransientDirectTreeNormalApproximation',
         'Allowed error:', '0.1', 'The allowed error in the normal approximation of the hypoexponental distribution.']
#        ,
#        ['Multi-time', 'HomogeneousHistogramsMultiTimeDirect2DSearch', 'Multiplicity:', '16'],
#        ['Tree', 'HomogeneousHistogramsDirectTree', 'Multiplicity:', '8']
        ]]
      ]
     ]

_homogeneousHistogramsSteadyState =\
    ['Histograms, Steady State',
     [['Direct', True, True, False,
       [['Elapsed time', 'HomogeneousHistogramsAverageElapsedTime', '', '', ''],
        ['Time steps', 'HomogeneousHistogramsAverage', '', '', ''],
        ['All possible steps', 'HomogeneousHistogramsAverageAps', '', '', '']]]
      ]
     ]

_statisticsTransient =\
    ['Statistics, Transient Behavior',
     [['Import Solution', False, False, False,
       [['', '', '', '', '']
        ]]
      ]
     ]

_statisticsSteadyState =\
    ['Statistics, Steady State',
     [['Import Solution', False, False, False,
       [['', '', '', '', '']
        ]]
      ]
     ]

_inhomogeneousTimeSeriesUniform =\
    ['Time Series, Uniform',
     [['Direct', False, True, False,
       [['Constant propensities', 'InhomogeneousTimeSeriesUniformDirect',
         '', '', '']]]
      ]
     ]

_inhomogeneousTimeSeriesAllReactions =\
    ['Time Series, All Reactions',
     [['Direct', False, True, False,
       [['Constant propensities', 'InhomogeneousTimeSeriesAllReactionsDirect',
         '', '', '']]]
      ]
     ]

_inhomogeneousTimeSeriesDeterministic =\
    ['Time Series, Deterministic',
     [['ODE, Integrate Reactions', False, True, False,
       [['Runge-Kutta, Cash-Karp', 'InhomogeneousOdeReactionRungeKuttaCashKarp',
         'Error:', '1e-8', 'The allowed error in a step.']
        ]]
      ]
     ]

_inhomogeneousHistogramsTransient =\
    ['Histograms, Transient Behavior',
     [['Direct', False, True, False,
       [['Constant propensities', 'InhomogeneousHistogramsTransientDirect',
         '', '', '']
        ]]
      ]
     ]

_inhomogeneousHistogramsSteadyState =\
    ['Histograms, Steady State',
     [['Direct', False, True, False,
       [['Time steps', 'InhomogeneousHistogramsSteadyStateDirectTimeSteps',
         '', '', '']]]
      ]
     ]

_eventsTimeSeriesUniform =\
    ['Time Series, Uniform',
     [['Direct', False, False, True,
       [['Constant propensities', '',
         '', '', '']]],
      ['First Reaction', False, False, True,
       [['Constant propensities', '',
         '', '', '']]]
      ]
     ]

_homogeneous =\
    [_homogeneousTimeSeriesUniform, _homogeneousTimeSeriesAllReactions,
     _homogeneousTimeSeriesDeterministic, _homogeneousHistogramsTransient,
     _homogeneousHistogramsSteadyState, _statisticsTransient,
     _statisticsSteadyState]
_inhomogeneous =\
    [_inhomogeneousTimeSeriesUniform, _inhomogeneousTimeSeriesAllReactions,
     _inhomogeneousTimeSeriesDeterministic, _inhomogeneousHistogramsTransient,
     _inhomogeneousHistogramsSteadyState, _statisticsTransient,
     _statisticsSteadyState]
_events =\
    [_eventsTimeSeriesUniform, _statisticsTransient, _statisticsSteadyState]

_solvers = [_homogeneous, _inhomogeneous, _events]

timeDependence = ['Time Homogeneous', 'Time Inhomogeneous', 'Use Events']

categories = [[x[0] for x in t] for t in _solvers]

methods = [[[x[0] for x in y[1]] for y in t] for t in _solvers]
hasGeneric = [[[x[1] for x in y[1]] for y in t] for t in _solvers]
hasCustom = [[[x[2] for x in y[1]] for y in t] for t in _solvers]
hasPython = [[[x[3] for x in y[1]] for y in t] for t in _solvers]

options = [[[[x[0] for x in y[4]] for y in z[1]] for z in t] for t in _solvers]
names = [[[[x[1] for x in y[4]] for y in z[1]] for z in t] for t in _solvers]
parameterNames1 = [[[[x[2] for x in y[4]] for y in z[1]] for z in t] for t in _solvers]
parameterValues1 = [[[[x[3] for x in y[4]] for y in z[1]] for z in t] for t in _solvers]
parameterToolTips1 = [[[[x[4] for x in y[4]] for y in z[1]] for z in t] for t in _solvers]

numberOfTimeDependence = len(timeDependence)
numberOfCategories = [len(category) for category in categories]
numberOfMethods = [[len(x) for x in method] for method in methods]
numberOfOptions = [[[len(x) for x in y] for y in option] for option in options]

def usesFrames(timeDependenceIndex, categoryIndex):
    return categories[timeDependenceIndex][categoryIndex] in \
        ('Time Series, Uniform', 'Time Series, Deterministic',
         'Histograms, Transient Behavior')

def usesStatistics(timeDependenceIndex, categoryIndex):
    return categories[timeDependenceIndex][categoryIndex] in \
        ('Statistics, Transient Behavior', 'Statistics, Steady State')

def isStochastic(timeDependenceIndex, categoryIndex):
    return categories[timeDependenceIndex][categoryIndex] !=\
        'Time Series, Deterministic'

def isDiscrete(timeDependenceIndex, categoryIndex):
    return categories[timeDependenceIndex][categoryIndex] !=\
        'Time Series, Deterministic'

def supportsEvents(timeDependenceIndex):
    return timeDependenceIndex == 2
    
if __name__ == '__main__':
    print('\ntimeDependence')
    print(timeDependence)
    print('\ncategories')
    print(categories)

    print('\nmethods')
    print(methods)
    print('\nhasGeneric')
    print(hasGeneric)
    print('\nhasCustom')
    print(hasCustom)
    print('\nhasPython')
    print(hasPython)

    print('\noptions')
    print(options)
    print('\nnames')
    print(names)
    print('\nparameterNames1')
    print(parameterNames1)
    print('\nparameterValues1')
    print(parameterValues1)
    print('\nparameterToolTips1')
    print(parameterToolTips1)

    print('\nnumberOfTimeDependence')
    print(numberOfTimeDependence)
    print('\nnumberOfCategories')
    print(numberOfCategories)
    print('\nnumberOfMethods')
    print(numberOfMethods)
    print('\nnumberOfOptions')
    print(numberOfOptions)

