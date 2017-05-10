"""The state of the model, method, and simulation output."""

import sys
import re
import csv
import copy
import threading
import subprocess
import os.path
import random
import shutil
import tempfile
import wx

from xml.sax import parse

# If we are running the unit tests.
if __name__ == '__main__':
    sys.path.insert(1, '..')
    resourcePath = '../'
else:
    from resourcePath import resourcePath
    resourcePath = os.path.abspath(resourcePath)

from Model import Model, duplicateModel
from Method import Method
from TimeSeriesFrames import TimeSeriesFrames
from TimeSeriesAllReactions import TimeSeriesAllReactions
from HistogramFrames import HistogramFrames
from HistogramAverage import HistogramAverage
from StatisticsFrames import StatisticsFrames
from StatisticsAverage import StatisticsAverage
from Histogram import Histogram
import simulationMethods
import Mt19937
from Utilities import getNewIntegerString, getUniqueName
import convert
from simulation.FirstReaction import FirstReaction
from simulation.Direct import Direct

from io.ContentHandler import ContentHandler
from io.ContentHandlerSbml import ContentHandlerSbml
import io.gnuplot
from io.XmlWriter import XmlWriter
from io.MathematicaWriter import MathematicaWriter
from gui.Preferences import Preferences

def fixSpacesInPath(path):
    if sys.platform in ('win32', 'win64'):
        return '\\'.join([re.search(' ', x) and '"' + x + '"' or x for x in
                          path.split('\\')])
    else:
        return re.sub(' ', '\ ', path)

def killProcess(process):
    """Kill the specified process."""
    # This does not work for win32 either. Both methods stop the process but do
    # not kill it.
    if False:
        import os
        os.popen('TASKKILL /PID ' + str(process.pid) + ' /F')
    if sys.platform in ('win32', 'win64'):
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(1, False, process.pid)
        ctypes.windll.kernel32.TerminateProcess(handle, -1)
        ctypes.windll.kernel32.CloseHandle(handle)
    else:
        import os
        import signal
        os.kill(process.pid, signal.SIGKILL)

class RecordingThreadPythonTimeSeriesFrames(threading.Thread):
    """Run the simulation. Call functions to record the trajectories, record
    the MT state, and increment the progress gauge."""
    def __init__(self, main, state, simulator, output, numberOfTrajectories):
        threading.Thread.__init__(self)
        # The main application window.
        self.main = main
        # The class that records the state.
        self.state = state
        # The simulation class.
        self.simulator = simulator
        # The simulation output is collected in an instance of TimeSeriesFrames.
        assert isinstance(output, TimeSeriesFrames)
        self.output = output
        # The number of trajectories to generate.
        assert isinstance(numberOfTrajectories, int)
        self.numberOfTrajectories = numberOfTrajectories
        self.halt = False

    def run(self):
        oldSize = len(self.output.populations)
        try:
            for i in range(self.numberOfTrajectories):
                # Generate a trajectory.
                self.simulator.initialize()
                self.simulator.generateTrajectory()
                # Record the trajectory.
                self.output.populations.append(\
                    copy.copy(self.simulator.populations))
                self.output.reactionCounts.append(\
                    copy.copy(self.simulator.reactionCounts))
                # Increment the progress gauge.
                self.main.incrementProgressGauge(1)
                if self.halt:
                    break
            # Record the Mersenne twister state.
            self.state.listOfMt19937States[0] = list(random.getstate()[1])
        except Exception, error:
            wx.MessageBox(str(error), 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            self.state.numberOfTrajectories = 0
            self.state.threads = []
            self.main.updateSimulations()
            self.main.launcher.isRunning = False
            self.main.launcher.update()
        # Record the number of successfully generated trajectories for the
        # completion message.
        self.state.successfulTrajectories += len(self.output.populations) -\
                                             oldSize


class RecordingThread(threading.Thread):
    """Base class for threads that read solver output."""
    def __init__(self, main, state, index, input, output):
        threading.Thread.__init__(self)
        self.main = main
        self.state = state
        # The thread index is in the range [0..numberOfThreads).
        self.index = index
        # The input stream to the solver.
        self.input = input
        # The output stream from the solver.
        self.output = output

    def handleException(self, error):
        wx.MessageBox(str(error), 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
        self.state.numberOfTrajectories = 0
        # Delete this thread.
        if self in self.state.threads:
            index = self.state.threads.index(self)
            self.state.processes.pop(index)
            self.state.threads.pop(index)
        # If no other threads are still recording.
        if not self.state.threads:
            # Reset the launcher.
            self.main.updateSimulations()
            self.main.launcher.isRunning = False
            self.main.launcher.update()

class RecordingThreadTimeSeriesFrames(RecordingThread):
    """Read populations, reaction counts, and Mersenne twister state from 
    the pipe. Call functions to record the trajectories, record the MT state,
    and increment the progress gauge."""
    def __init__(self, main, state, index, input, output):
        RecordingThread.__init__(self, main, state, index, input, output)

    def run(self):
        try:
            isFirst = True
            # Run until we have generated enough trajectories.
            while True:
                # Send the number of trajectories to generate.
                numberOfTrajectories = self.state.\
                                       getNumberOfTrajectoriesToLaunch()
                self.input.write('%d\n' % numberOfTrajectories)
                self.input.flush()
                # Stop if there are no more trajectories to generate.
                if numberOfTrajectories == 0:
                    break
                if isFirst:
                    # Skip the blank line.
                    self.output.readline()
                    isFirst = False
                # Record the generated trajectories.
                self.state.readTimeSeriesFrames(self.output, self.index)
                # Increment the progress gauge.
                self.main.incrementProgressGauge(numberOfTrajectories)
        except Exception, error:
            self.handleException(error)

class RecordingThreadTrajectoryAll(RecordingThread):
    """Read populations, reaction counts, and Mersenne twister state from 
    the pipe. Call functions to record the trajectories, record the MT state,
    and increment the progress gauge."""
    def __init__(self, main, state, index, input, output):
        RecordingThread.__init__(self, main, state, index, input, output)

    def run(self):
        try:
            isFirst = True
            # Run until we have generated enough trajectories.
            while True:
                # Send the number of trajectories to generate.
                numberOfTrajectories = self.state.\
                                       getNumberOfTrajectoriesToLaunch()
                self.input.write('%d\n' % numberOfTrajectories)
                self.input.flush()
                # Stop if there are no more trajectories to generate.
                if numberOfTrajectories == 0:
                    break
                if isFirst:
                    # Skip the blank line.
                    self.output.readline()
                    isFirst = False
                # Record the generated trajectories.
                self.state.readTimeSeriesAllReactions(self.output, self.index)
                # Increment the progress gauge.
                self.main.incrementProgressGauge(numberOfTrajectories)
        except Exception, error:
            self.handleException(error)

class RecordingThreadHistogramFrames(RecordingThread):
    """Record the solver output."""
    def __init__(self, main, state, index, input, output):
        RecordingThread.__init__(self, main, state, index, input, output)

    def run(self):
        try:
            # Run until we have generated enough trajectories.
            while True:
                # Send the number of trajectories to generate.
                numberOfTrajectories = self.state.\
                                       getNumberOfTrajectoriesToLaunch()
                self.input.write('%d\n' % numberOfTrajectories)
                self.input.flush()
                # Stop if there are no more trajectories to generate.
                if numberOfTrajectories == 0:
                    break
                # CONTINUE: Sometime this fails. Determine why.
                # Read the number of completed trajectories in this task.
                numberCompleted = int(self.output.readline())
                assert numberCompleted == numberOfTrajectories
                # Increment the progress gauge.
                self.main.incrementProgressGauge(numberOfTrajectories)
            # Skip the blank line.
            self.output.readline()
            # Record the histograms from the generated trajectories.
            self.state.readHistogramFrames(self.output, self.index)
        except Exception, error:
            self.handleException(error)

class RecordingThreadHistogramAverage(RecordingThread):
    """Record the solver output."""
    def __init__(self, main, state, index, input, output):
        RecordingThread.__init__(self, main, state, index, input, output)

    def run(self):
        try:
            # Run until we have generated enough trajectories.
            while True:
                # Send the number of trajectories to generate.
                numberOfTrajectories = self.state.\
                                       getNumberOfTrajectoriesToLaunch()
                self.input.write('%d\n' % numberOfTrajectories)
                self.input.flush()
                # Stop if there are no more trajectories to generate.
                if numberOfTrajectories == 0:
                    break
                # CONTINUE: Sometime this fails. Determine why.
                # Read the number of completed trajectories in this task.
                numberCompleted = int(self.output.readline())
                assert numberCompleted == numberOfTrajectories
                # Increment the progress gauge.
                self.main.incrementProgressGauge(numberOfTrajectories)
            # Skip the blank line.
            self.output.readline()
            # Record the histograms from the generated trajectories.
            self.state.readHistogramAverage(self.output, self.index)
        except Exception, error:
            self.handleException(error)

class State:
    """The state of the models, methods, and simulation output.

    self.output holds the simulation output. self.solvers
    holds the names of the compiled custom executables. The output 
    dictionary is used to determine if the user can edit a model or 
    simulation parameters. The solvers dictionary exploits this to keep 
    custom executable in sync with the model and simulation parameters.
    When a custom solver is compiled, an empty data structure is 
    added if there is not already accumulated output. When a set of 
    simulation output is deleted, the corresponding custom executable is
    deleted (if it exists)."""
    
    def __init__(self):
        """Constructor."""
        self.preferences = Preferences()
        self.processes = []
        self.threads = []
        self.lock = threading.Lock()
        self.errorMessages = []
        self.clear()
        # The initial seed used to generate MT 19937 states.
        self.seed = 2**31
        self.listOfMt19937States = []
        self.lock = threading.Lock()
        self.customSolversDirectory = tempfile.mkdtemp()
        self.version = '1.6'

    def __del__(self):
        if os.access(self.customSolversDirectory, os.F_OK):
            shutil.rmtree(self.customSolversDirectory)

    def clear(self):
        """Clear the data structure."""
        # Dictionary of models. The keys are the model identifiers.
        self.models = {}
        self.methods = {}
        # Dictionary of output. The keys are tuples of model 
        # identifiers and simulation parameter identifiers.
        self.output = {}
        # Dictionary of compiled solvers. The keys are tuples of model 
        # identifiers and simulation parameter identifiers.
        self.solvers = {}

    def generateNewMt19937State(self):
        """Generate a new MT 19937 state."""
        # Generate the state and get a new seed. The state is an array of 
        # 624 32-bit unsigned integers and a position in the array.
        state, self.seed = Mt19937.generateState(self.seed)
        # Add it to the list of states.
        self.listOfMt19937States.append(state)

    def seedMt19937(self, seed):
        # The seed is a 32-bit unsigned integer.
        assert 0 <= seed and seed < 2**32
        # Record the seed.
        self.seed = seed
        # Clear the current states. New states will be generated when 
        # simulations are launched.
        self.listOfMt19937States = []

    def insertNewModel(self):
        """Insert a new model. Return the new model identifier."""
        # Make an empty model with a unique identifier.
        model = Model()
        model.id = getNewIntegerString(self.models.keys())
        # Add the model to the dictionary.
        self.models[model.id] = model
        # Return the new model's key.
        return model.id

    def insertCloneModel(self, id):
        """Insert a clone of the specified model. Return the new model 
        identifier."""
        assert id in self.models
        # Make an empty model with a unique identifier.
        model = copy.deepcopy(self.models[id])
        model.id = getUniqueName(id, self.models.keys())
        # Add the model to the dictionary.
        self.models[model.id] = model
        # Return the new model's key.
        return model.id

    def insertDuplicatedModel(self, id, multiplicity, useScaling):
        """Insert a duplicated version of the specified model. Return the
        new model identifier."""
        assert id in self.models
        # Make a duplicated model with a unique identifier.
        model = duplicateModel(self.models[id], multiplicity, useScaling)
        model.id = getUniqueName(id, self.models.keys())
        # Add the model to the dictionary.
        self.models[model.id] = model
        # Return the new model's key.
        return model.id

    def insertNewMethod(self):
        """Insert new simulation method. Return the new identifier."""
        # Make an empty model with a unique identifier.
        m = Method()
        m.id = getNewIntegerString(self.methods.keys())
        # Add the method to the dictionary.
        self.methods[m.id] = m
        # Return the new key.
        return m.id

    def insertCloneMethod(self, id):
        """Insert a clone of the specified method. Return the new identifier."""
        assert id in self.methods
        # Make an empty method with a unique identifier.
        m = copy.deepcopy(self.methods[id])
        m.id = getUniqueName(id, self.methods.keys())
        # Add to the dictionary.
        self.methods[m.id] = m
        # Return the new key.
        return m.id

    def changeModelId(self, old, new):
        # Change in the set of models.
        self.models[new] = self.models[old]
        self.models[new].id = new
        del self.models[old]
        # Change in the simulation output.
        for (modelId, methodId) in self.output.keys():
            if modelId == old:
                self.output[(new, methodId)] =\
                    self.output[(old, methodId)]
                del self.output[(old, methodId)]
        # Change in the set of solvers.
        for (modelId, methodId) in self.solvers.keys():
            if modelId == old:
                self.solvers[(new, methodId)] =\
                    self.solvers[(old, methodId)]
                del self.solvers[(old, methodId)]

    def changeMethodId(self, old, new):
        # Change in the set of simulation parameters.
        self.methods[new] = self.methods[old]
        self.methods[new].id = new
        del self.methods[old]
        # Change in the simulation output.
        for (modelId, methodId) in self.output.keys():
            if methodId == old:
                self.output[(modelId, new)] =\
                    self.output[(modelId, old)]
                del self.output[(modelId, old)]
        # Change in the set of solvers.
        for (modelId, methodId) in self.solvers.keys():
            if methodId == old:
                self.solvers[(modelId, new)] =\
                    self.solvers[(modelId, old)]
                del self.solvers[(modelId, old)]

    def doesModelHaveDependentOutput(self, id):
        for (modelId, methodId) in self.output.keys():
            if id == modelId:
                return True
        return False

    def doesMethodHaveDependentOutput(self, id):
        for (modelId, methodId) in self.output.keys():
            if id == methodId:
                return True
        return False

    def deleteOutput(self, modelId, methodId):
        del self.output[(modelId, methodId)]
        if (modelId, methodId) in self.solvers:
            del self.solvers[(modelId, methodId)]

    def deleteAllOutput(self):
        self.output = {}
        self.solvers = {}

    def saveCustomExecutable(self, modelId, methodId, fileName):
        """Save a custom executable."""
        if sys.platform in ('win32', 'win64'):
            suffix = '.exe'
        else:
            suffix = ''
        shutil.copy(os.path.join(self.customSolversDirectory,
                                 self.solvers[(modelId, methodId)] +
                                 suffix), fileName)

    def saveGenericExecutable(self, methodId, fileName):
        """Save a generic executable."""
        if sys.platform in ('win32', 'win64'):
            suffix = '.exe'
        else:
            suffix = ''
        m = self.methods[methodId]
        shutil.copy(os.path.join(
            resourcePath, 'solvers', 
            simulationMethods.names[m.timeDependence][m.category][m.method][m.options] +
            suffix), fileName)

    def exportMathematica(self, modelId, methodId, recordedSpecies,
                          recordedReactions, outputFile):
        """Export the specified model and simulation parameters to a 
        Mathematica notebook."""
        # If necessary, start a new output container for this model and method.
        self.ensureOutput(modelId, methodId, recordedSpecies, recordedReactions)
        writer = MathematicaWriter(outputFile)
        self.models[modelId].writeMathematica\
            (writer, self.methods[methodId], recordedSpecies, recordedReactions)

    # Edit menu.

    # CONTINUE Perhaps move the implementation.
    def editMethod(self, id, timeDependence, category, method, options, 
                   startTime, equilibrationTime, recordingTime, maximumSteps,
                   numberOfFrames, numberOfBins, multiplicity, solverParameter):
        """Edit the method. Return None if there are no errors. Otherwise
        return an error message."""
        assert id in self.methods
        m = Method()
        m.id = id
        m.timeDependence = timeDependence
        m.category = category
        m.method = method
        m.options = options
        m.startTime = startTime
        m.equilibrationTime = equilibrationTime
        m.recordingTime = recordingTime
        m.maximumSteps = maximumSteps
        m.numberOfFrames = numberOfFrames
        m.numberOfBins = numberOfBins
        m.multiplicity = multiplicity
        m.solverParameter = solverParameter
        errorMessage = m.hasErrors()
        if not errorMessage:
            self.methods[id] = m
        return errorMessage

    # Simulation menu callbacks.

    def clearOutput(self):
        """Clear the simulation output and the solvers."""
        self.output = {}
        self.solvers = {}

    def makePackedReactionsString(self, model):
        """Return the packed reactions string for the specified model."""
        data = []
        for reaction in model.reactions:
            data.append(len(reaction.reactants))
            for speciesReference in reaction.reactants:
                data.append(model.speciesIdentifiers.index(\
                        speciesReference.species))
                data.append(speciesReference.stoichiometry)
            data.append(len(reaction.products))
            for speciesReference in reaction.products:
                data.append(model.speciesIdentifiers.index(\
                        speciesReference.species))
                data.append(speciesReference.stoichiometry)
        return ' '.join([repr(x) for x in data])

    def computeFrameTimes(self, method):
        startTime = method.startTime
        equilibrationTime = method.equilibrationTime
        recordingTime = method.recordingTime
        numberOfFrames = method.numberOfFrames
        frameTimes = [0] * numberOfFrames
        if numberOfFrames == 1:
            frameTimes[0] = startTime + equilibrationTime + recordingTime
        else:
            t0 = startTime + equilibrationTime
            for i in range(numberOfFrames):
                frameTimes[i] = t0 + i * recordingTime / (numberOfFrames - 1)
        return frameTimes

    def evaluateModel(self, modelId):
        """Evaluate the parameters, the species initial amounts, and the
        reaction propensities for the mass action kinetic laws. Return
        None if successful. Otherwise return an error message."""
        return self.models[modelId].evaluate()

    def ensureOutput(self, modelId, methodId, recordedSpecies,
                     recordedReactions):
        """Ensure that there is an output container for the specified model
        and method."""
        # Do nothing if the output container already exists.
        if (modelId, methodId) in self.output:
            return

        method = self.methods[methodId]
        category = simulationMethods.categories[method.timeDependence]\
            [method.category]

        if category in ('Time Series, Uniform', 'Time Series, Deterministic'):
            self.output[(modelId, methodId)] =\
                TimeSeriesFrames(self.computeFrameTimes(method),
                                 recordedSpecies, recordedReactions)
        elif category == 'Time Series, All Reactions':
            # By definition all species and reactions are recorded.
            model = self.models[modelId]
            assert len(recordedSpecies) == len(model.speciesIdentifiers)
            assert len(recordedReactions) == len(model.reactions)
            initialTime = method.startTime + method.equilibrationTime
            finalTime = initialTime + method.recordingTime
            self.output[(modelId, methodId)] =\
                TimeSeriesAllReactions(recordedSpecies, recordedReactions,
                                       initialTime, finalTime)
        elif category == 'Histograms, Transient Behavior':
            hf = HistogramFrames(method.numberOfBins, method.multiplicity,
                                 recordedSpecies)
            hf.setFrameTimes(self.computeFrameTimes(method))
            self.output[(modelId, methodId)] = hf
        elif category == 'Histograms, Steady State':
            hf = HistogramFrames(method.numberOfBins, method.multiplicity,
                                 recordedSpecies)
            self.output[(modelId, methodId)] =\
                HistogramAverage(method.numberOfBins, method.multiplicity,
                                 recordedSpecies)
        else:
            assert False

    def launchSuiteOfSimulations(self, main, modelId, methodId, recordedSpecies,
                                 recordedReactions, numberOfProcesses,
                                 numberOfTrajectories, trajectoriesPerTask,
                                 niceIncrement, useCustomSolver):
        """Launch the suite of simulations.
        Return an error message if the solver cannot be launched. Otherwise
        return None."""
        assert numberOfProcesses >= 1
        # There should not be a running simulation.
        assert not self.threads
        assert not self.processes

        # Clear any old error messages.
        self.errorMessages = []
        # Record these for getNumberOfTrajectoriesToLaunch().
        self.numberOfTrajectories = numberOfTrajectories
        self.trajectoriesPerTask = trajectoriesPerTask

        # If necessary, create a new output container for this model and set
        # of parameters.
        self.ensureOutput(modelId, methodId, recordedSpecies,
                          recordedReactions)
        self.currentOutput = self.output[(modelId, methodId)]
        self.successfulTrajectories = 0

        timeDependence = self.methods[methodId].timeDependence
        category = self.methods[methodId].category
        method = self.methods[methodId].method
        options = self.methods[methodId].options

        if sys.platform in ('win32', 'win64'):
            suffix = '.exe'
            command = 'start /B '
            if niceIncrement < 4:
                command += '/High '
            elif niceIncrement < 8:
                command += '/Abovenormal '
            elif niceIncrement < 12:
                command += '/Normal '
            elif niceIncrement < 16:
                command += '/Belownormal '
            else:
                command += '/Low '
        else:
            suffix = ''
            command = 'nice -n ' + str(niceIncrement) + ' '

        # If there is a compiled custom solver.
        if useCustomSolver and (modelId, methodId) in self.solvers:
            solverPath = os.path.join(self.customSolversDirectory,
                                      self.solvers[(modelId, methodId)] +
                                      suffix)
            if not os.access(solverPath, os.X_OK):
                return 'Unable to access the custom solver ' \
                       + solverPath \
                       + '. Please submit a bug report to sean@caltech.edu.'
        else:
            solverPath = os.path.join(resourcePath, 'solvers', 
                                      simulationMethods.names[timeDependence][category][method][options] +
                                      suffix)
            if not os.access(solverPath, os.X_OK):
                return 'Unable to access the built-in solver ' \
                       + solverPath \
                       + '. Ensure that the solvers have been compiled. (See the Installation section of the manual.)'
        # Fix the spaces in the solver path as it will be passed to the shell.
        command += fixSpacesInPath(solverPath)
        

        for index in range(min(numberOfProcesses, numberOfTrajectories)):
            solverString =\
                self.makeSolverString(modelId, methodId, recordedSpecies,
                                      recordedReactions, False)
            if simulationMethods.isStochastic(timeDependence, category):
                solverString += self.makeMt19937String(index)
            process = subprocess.Popen(command, bufsize=-1,
                                       universal_newlines=True,
                                       shell=True, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE)
            self.processes.append(process)

            # Send the input to the process (except for the number of 
            # trajectories).
            process.stdin.write(solverString)
            # Create the recorder thread.
            # The solver either generates frames or records all reactions.
            if simulationMethods.categories[timeDependence][category] in\
                    ('Time Series, Uniform', 'Time Series, Deterministic'):
                thread = \
                    RecordingThreadTimeSeriesFrames\
                    (main, self, index, process.stdin, process.stdout)
            elif simulationMethods.categories[timeDependence][category] ==\
                    'Time Series, All Reactions':
                thread = \
                    RecordingThreadTrajectoryAll\
                    (main, self, index, process.stdin, process.stdout)
            elif simulationMethods.categories[timeDependence][category] ==\
                    'Histograms, Transient Behavior':
                thread = \
                    RecordingThreadHistogramFrames\
                    (main, self, index, process.stdin, process.stdout)
            elif simulationMethods.categories[timeDependence][category] ==\
                    'Histograms, Steady State':
                thread = \
                    RecordingThreadHistogramAverage\
                    (main, self, index, process.stdin, process.stdout)
            else:
                assert False
            thread.start()
            self.threads.append(thread)
        # No launching errors.
        return None

    def launchPythonSimulation(self, main, modelId, methodId, recordedSpecies,
                               recordedReactions, numberOfTrajectories):
        """Launch the Python simulation.
        Return an error message if the solver cannot be launched. Otherwise
        return None."""
        # There should not be a running simulation.
        assert not self.threads
        assert not self.processes

        # Clear any old error messages.
        self.errorMessages = []

        # If necessary, create a new output container for this model and set
        # of parameters.
        self.ensureOutput(modelId, methodId, recordedSpecies, recordedReactions)
        output = self.output[(modelId, methodId)]
        self.successfulTrajectories = 0

        # Create the simulator.
        method = self.methods[methodId]
        name = simulationMethods.methods[method.timeDependence]\
               [method.category][method.method]
        # Choose the solver.
        if name == 'Direct':
            solver = Direct(convert.makeModel(self.models[modelId],
                                              method), method.maximumSteps)
        elif name == 'First Reaction':
            solver = FirstReaction(convert.makeModel(self.models[modelId],
                                                     method),
                                   method.maximumSteps)
        else:
            raise Exception('Unknown method "' + name + '" encountered.')
        simulator = convert.makeTimeSeriesUniform\
                    (solver, self.models[modelId], output)

        # Seed the random number generator
        if not self.listOfMt19937States:
            self.generateNewMt19937State()
        version = random.getstate()[0]
        if version >= 3:
            random.setstate((version,
                             tuple(self.listOfMt19937States[0]), None))
        
        # Create the recorder thread.
        thread = \
            RecordingThreadPythonTimeSeriesFrames\
            (main, self, simulator, output, numberOfTrajectories)
        thread.start()
        self.threads.append(thread)
        # No launching errors.
        return None

    def hasCustomSolver(self, modelId, methodId):
        return (modelId, methodId) in self.solvers

    def compileSolver(self, modelId, methodId, recordedSpecies,
                      recordedReactions):
        """Compile the solver. If there is an error, return the error message.
        Otherwise return None. Note that the Propensities* files are placed
        in the custom solvers directory, which is a temporary directory. This 
        is necessary because for MS Windows applications that are installed,
        one encounters wierdness when altering the contents of the application
        directory. If you add a file it is virtually there, but not actually
        there."""
        # Make a resource path that we can pass to the shell.
        rp = fixSpacesInPath(resourcePath)
        if sys.platform in ('win32', 'win64'):
            process = subprocess.Popen(os.path.join(rp,
                                                    r'solvers\vc10vars32.bat'),
                                       bufsize=-1,
                                       universal_newlines=True, shell=True,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            (stdout, stderr) = process.communicate()
            # If there is an error in configuring the MSVC compiler.
            if stderr:
                return 'Unable to configure the Microsoft Visual C++ 2010 compiler.\nMake sure that it is installed.\nThe Visual C++ Express Edition 2010 is a free download, available from:\nhttp://www.microsoft.com/Express/vc/'
        else:
            # If there is an error in finding the GNU C++ compiler.
            if subprocess.call(r'g++ --version', bufsize=-1, shell=True,
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE):
                return 'Unable to find the GNU C++ compiler.\nMake sure that it is installed and the compiler is in your path.\nYou can check this with the shell command "which g++".'
        model = self.models[modelId]
        method = self.methods[methodId]
        if method.timeDependence == 0:
            # Time homogeneous.
            # Number of reactions.
            f = open(os.path.join(self.customSolversDirectory,
                                  'PropensitiesNumberOfReactions.ipp'), 'w')
            f.write(model.makePropensitiesNumberOfReactions())
            f.close()
            # Constructor.
            f = open(os.path.join(self.customSolversDirectory,
                                  'PropensitiesConstructor.ipp'), 'w')
            f.write(model.makePropensitiesConstructor())
            f.close()
            # Member functions.
            f = open(os.path.join(self.customSolversDirectory,
                                  'PropensitiesMemberFunctions.ipp'), 'w')
            f.write(model.makePropensitiesMemberFunctions(\
                    method.isDiscrete()))
            f.close()
        else:
            # Time homogeneous.
            # Compute the propensities.
            f = open(os.path.join(self.customSolversDirectory,
                                  'computePropensities.h'), 'w')
            f.write(model.makeInhomogeneousPropensities(method.isDiscrete()))
            f.close()
        # Compile the solver.
        sourceName = simulationMethods.names[method.timeDependence]\
            [method.category][method.method][method.options]
        assert not (modelId, methodId) in self.solvers
        executableName = getUniqueName(sourceName, self.solvers.values())
        if sys.platform in ('win32', 'win64'):
            suffix = '.exe'
        else:
            suffix = ''
        output = os.path.join(self.customSolversDirectory,
                              executableName + suffix)
        source = os.path.join(resourcePath, 'solvers', sourceName + '.cc')
        if sys.platform in ('win32', 'win64'):
            # The arguments to cl need to be enclosed in quotes in case
            # the path has spaces.
            command = os.path.join(rp, r'solvers\vc10vars32.bat') + '&&cl /I"' + os.path.join(resourcePath, 'src') + r'" /I"' + os.path.join(resourcePath, r'src\third-party') + r'" /I' + self.customSolversDirectory + r' /Ox /EHsc /DSTOCHASTIC_CUSTOM_PROPENSITIES /Fo' + self.customSolversDirectory + ' /Fe' + output + ' "' + source + '"'
        else:
            cxx = self.preferences.data['Compilation']['Compiler'] +\
                ' -DSTOCHASTIC_CUSTOM_PROPENSITIES '
            cxx += '-o ' + output + ' '
            cxxFlags = self.preferences.data['Compilation']['Flags'] +\
                ' -I' + os.path.join(rp, 'src') +\
                ' -I' + self.customSolversDirectory + ' '
            command = cxx + cxxFlags + source
        process = subprocess.Popen(command, bufsize=-1,
                                   universal_newlines=True, shell=True,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        (stdout, stderr) = process.communicate()
        #process.wait()
        # If there was an error
        if process.returncode != 0:
            # Return the error message.
            return command + '\n' + stderr + stdout
        else:
            # Record the executable in the dictionary of custom solvers.
            self.solvers[(modelId, methodId)] = executableName
            # Ensure there is an output container so the model and simulation
            # parameters are not modified.
            self.ensureOutput(modelId, methodId, recordedSpecies,
                              recordedReactions)
            return None


    def getNumberOfTrajectoriesToLaunch(self):
        self.lock.acquire()
        if self.numberOfTrajectories != 0:
            result = min(self.numberOfTrajectories, self.trajectoriesPerTask)
            self.numberOfTrajectories -= result
        else:
            result = 0
        self.lock.release()
        return result

    def tearDownSimulation(self):
        """Wait for the threads to finish. Clear the lists of processes and
        threads."""
        for thread in self.threads:
            thread.join()
        self.processes = []
        self.threads = []
            
    def stopPythonSimulation(self):
        # Clear the pending jobs.
        for thread in self.threads:
            thread.halt = True
        self.tearDownSimulation()

    def stopSimulation(self):
        # If it is a Python simulation that is being run.
        if self.threads and isinstance(self.threads[0],
                                       RecordingThreadPythonTimeSeriesFrames):
            self.stopPythonSimulation()
        else:
            # Clear the pending jobs.
            self.numberOfTrajectories = 0
            self.tearDownSimulation()

    def killSimulation(self):
        """Kill the processes. The threads will terminate by themselves."""
        # If it is a Python simulation that is being run.
        if self.threads and isinstance(self.threads[0],
                                       RecordingThreadPythonTimeSeriesFrames):
            # CONTINUE: Support killing, and not just stopping.
            self.stopPythonSimulation()
        else:
            # Clear the pending jobs.
            self.numberOfTrajectories = 0
            for process in self.processes:
                killProcess(process)
            self.tearDownSimulation()

    def makeMt19937String(self, processIndex):
        """
        Write the list of the MT 19937 state, terminated by a newline.
        """
        # Ensure there are sufficient RNG states.
        while len(self.listOfMt19937States) <= processIndex:
            self.generateNewMt19937State()
        # List of MT 19937 state.
        return ' '.join([str(x) for x in 
                         self.listOfMt19937States[processIndex]]) + '\n'

    def makeSolverString(self, modelId, methodId, recordedSpecies,
                         recordedReactions, arePrintingInformation):
        """Make the solver string for either a frames solver or an all 
        reactions solver."""
        method = self.methods[methodId]
        category = simulationMethods.categories[method.timeDependence]\
            [method.category]
        if arePrintingInformation:
            firstLine = '1\n'
        else:
            firstLine = '0\n'
        if category in ('Time Series, Uniform', 'Time Series, Deterministic'):
            return firstLine + self.makeSolverStringTimeSeriesFrames\
        (modelId, methodId, recordedSpecies, recordedReactions)
        elif category == 'Time Series, All Reactions':
            return firstLine + self.makeSolverStringTimeSeriesAllReactions\
                (modelId, methodId, recordedSpecies, recordedReactions)
        elif category == 'Histograms, Transient Behavior':
            return firstLine + self.makeSolverStringHistogramFrames\
                (modelId, methodId, recordedSpecies, recordedReactions)
        elif category == 'Histograms, Steady State':
            return firstLine + self.makeSolverStringHistogramAverage\
                (modelId, methodId, recordedSpecies, recordedReactions)
        else:
            assert False

    def makeSolverStringCommon(self, modelId, methodId, recordedSpecies,
                               recordedReactions):
        """
        <number of species>
        <number of reactions>
        <list of initial amounts>
        <packed reactions>
        <list of propensity factors>
        <number of species to record>
        <list of species to record>
        <number of reactions to record>
        <list of reactions to record>
        <maximum allowed steps>
        <number of solver parameters>
        <list of solver parameters>
        <starting time>
        """
        lines = []
        model = self.models[modelId]
        # Number of species.
        numberOfSpecies = len(model.species)
        lines.append('%d\n' % numberOfSpecies)
        # Number of reactions.
        numberOfReactions = len(model.reactions)
        lines.append('%d\n' % numberOfReactions)
        # List of initial amounts.
        data = []
        for i in range(numberOfSpecies):
            id = model.speciesIdentifiers[i]
            data.append(model.species[id].initialAmountValue)
        lines.append('%s\n' % ' '.join([str(x) for x in data]))
        # Packed reactions.
        lines.append('%s\n' % self.makePackedReactionsString(model))
        # List of propensity factors.
        propensityFactors = []
        for reaction in model.reactions:
            if reaction.massAction:
                propensityFactors.append(repr(reaction.propensityFactor))
            else:
                propensityFactors.append('0')
        lines.append('%s\n' % ' '.join(propensityFactors))
        # Number of species to record.
        lines.append('%d\n' % len(recordedSpecies))
        # List of species to record.
        lines.append('%s\n' % ' '.join([str(i) for i in recordedSpecies]))
        # Number of reactions to record.
        lines.append('%d\n' % len(recordedReactions))
        # List of reactions to record.
        lines.append('%s\n' % ' '.join([str(i) for i in recordedReactions]))
        # The maximum allowed steps.
        m = self.methods[methodId]
        if m.maximumSteps:
            lines.append('%r\n' % m.maximumSteps)
        else:
            # Don't place any limit on the maximum number of steps.
            lines.append('0\n')
        # If this method uses the first parameter.
        if simulationMethods.parameterNames1[m.timeDependence][m.category]\
                [m.method][m.options]:
            lines.append('1\n%s\n' % repr(m.solverParameter))
        else:
            lines.append('0\n\n')
        # Starting time.
        lines.append('%r\n' % m.startTime)
        return lines

    def makeSolverStringTimeSeriesFrames(self, modelId, methodId,
                                         recordedSpecies, recordedReactions):
        """
        Write everything except the number of trajectories.
        Use str() for integer types and repr() for floating point types.
        The common fields plus the following category specific fields:

        <number of frames>
        <list of frame times>
        """
        # The common fields.
        lines = self.makeSolverStringCommon(modelId, methodId, recordedSpecies,
                                            recordedReactions)
        # Number of frames.
        m = self.methods[methodId]
        lines.append('%d\n' % m.numberOfFrames)
        # List of frame times.
        lines.append('%s\n' % ' '.join([repr(x) for x in 
                                        self.computeFrameTimes(m)]))
        return ''.join(lines)

    def makeSolverStringTimeSeriesAllReactions\
            (self, modelId, methodId, recordedSpecies, recordedReactions):
        """
        Write everything except the number of trajectories.
        Use str() for integer types and repr() for floating point types.
        The common fields plus the following category specific fields:

        <equilibration time>
        <recording time>
        """
        # The common fields.
        lines = self.makeSolverStringCommon(modelId, methodId, recordedSpecies,
                                            recordedReactions)
        # Equilibration and recording times.
        lines.append('%r\n%r\n' % (self.methods[methodId].equilibrationTime,
                                   self.methods[methodId].recordingTime))
        return ''.join(lines)

    def makeSolverStringHistogramFrames(self, modelId, methodId,
                                        recordedSpecies, recordedReactions):
        """
        Write everything except the number of trajectories.
        Use str() for integer types and repr() for floating point types.
        The common fields plus the following category specific fields:

        <number of frames>
        <list of frame times>
        <number of bins in histograms>
        <histogram multiplicity>
        """
        # The common fields.
        lines = self.makeSolverStringCommon(modelId, methodId, recordedSpecies,
                                            recordedReactions)
        # Number of frames.
        m = self.methods[methodId]
        lines.append('%d\n' % m.numberOfFrames)
        # List of frame times.
        lines.append('%s\n' % ' '.join([repr(x) for x in 
                                        self.computeFrameTimes(m)]))
        # Number of bins in histograms.
        lines.append('%d\n' % m.numberOfBins)
        # Histogram multiplicity.
        lines.append('%d\n' % m.multiplicity)
        return ''.join(lines)


    def makeSolverStringHistogramAverage(self, modelId, methodId,
                                         recordedSpecies, recordedReactions):
        """
        Write everything except the number of trajectories.
        Use str() for integer types and repr() for floating point types.
        The common fields plus the following category specific fields:

        <equilibration time>
        <recording time>
        <number of bins in histograms>
        <histogram multiplicity>
        """
        # The common fields.
        lines = self.makeSolverStringCommon(modelId, methodId, recordedSpecies,
                                            recordedReactions)
        # Equilibration and recording times.
        m = self.methods[methodId]
        lines.append('%r\n%r\n' % (m.equilibrationTime, m.recordingTime))
        # Number of bins in histograms.
        lines.append('%d\n' % m.numberOfBins)
        # Histogram multiplicity.
        lines.append('%d\n' % m.multiplicity)
        return ''.join(lines)


    def exportJob(self, outputFile, modelId, methodId, recordedSpecies,
                  recordedReactions, numberOfTrajectories, processIndex = 0):
        """Add the list of MT 19937 state and the number of trajectories to
        the output of makeSolverString()."""
        # The output file.
        solverString = self.makeSolverString(modelId, methodId,
                                             recordedSpecies, recordedReactions,
                                             True)
        parameters = self.methods[methodId]
        if simulationMethods.isStochastic(parameters.timeDependence,
                                          parameters.category):
            solverString += self.makeMt19937String(processIndex)
        outputFile.write(solverString)
        # Number of trajectories.
        outputFile.write('%d\n' % numberOfTrajectories)
        outputFile.close()

    def readStatisticsFrames(self, inputFile, output):
        """
        Read the mean and standard deviation information from the file. 
        """
        # Mean and standard deviation for each recorded species.
        n = len(output.recordedSpecies)
        output.statistics = []
        for f in output.frameTimes:
            data = map(float, inputFile.readline().rstrip().split())
            if len(data) != 2 * n:
                raise Exception('Expected a list of %s numbers in the row '\
                                'for time frame %s.' % (2 * n, f))
            output.statistics.append([(data[2*i], data[2*i+1]) for i in
                                      range(n)])
        # Verify that the output is valid.
        error = output.hasErrors()
        if error:
            raise Exception('The imported solution is not valid:\n' + error)

    def readStatisticsAverage(self, inputFile, output):
        """
        Read the mean and standard deviation information from the file. 
        """
        # Mean and standard deviation for each recorded species.
        n = len(output.recordedSpecies)
        data = map(float, inputFile.readline().rstrip().split())
        if len(data) != 2 * n:
            raise Exception('Expected a list of %s numbers.' % 2 * n)
        output.statistics = [(data[2*i], data[2*i+1]) for i in range(n)]
        # Verify that the output is valid.
        error = output.hasErrors()
        if error:
            raise Exception('The imported solution is not valid:\n' + error)

    def readTimeSeriesFrames(self, inputFile, processIndex):
        """
        Read trajectories from the file. Append them to the list of
        trajectories. Record the new RNG state for the specified process index.
        Below is the file format.

        <number of trajectories>
        for each trajectory:
          <list of initial MT 19937 state>
          if successful:
            <blank line>
            <list of populations>
            <list of reaction counts>
          else:
            <error message>
        <list of final MT 19937 state>

        This function uses the following member variables:
        self.currentOutput
        self.listOfMt19937States
        """
        frameTimes = self.currentOutput.frameTimes
        # Number of trajectories.
        numberOfTrajectories = int(inputFile.readline())
        for i in range(numberOfTrajectories):
            # CONTINUE: I ignore the initial RNG state for now.
            inputFile.readline()
            errorMessage = inputFile.readline().rstrip()
            if errorMessage == '':
                # List of populations.
                populations = map(float, inputFile.readline().rstrip().split())
                # List of reaction counts.
                reactionCounts = map(float,
                                     inputFile.readline().rstrip().split())
                # Acquire the lock so that we can modify self.currentOutput.
                self.lock.acquire()
                self.currentOutput.appendPopulations(populations)
                self.currentOutput.appendReactionCounts(reactionCounts)
                self.successfulTrajectories += 1
                self.lock.release()
            else:
                self.lock.acquire()
                self.errorMessages.append(errorMessage)
                self.lock.release()
        # List of final MT 19937 state.
        state = map(int, inputFile.readline().rstrip().split())
        # If this is a stochastic method.
        if state:
            assert len(state) == 625
            # Record the RNG state. No need for a lock. The threads modify
            # different list elements.
            self.listOfMt19937States[processIndex] = state
        # CONTINUE: Verify that the output is valid.

    def readTimeSeriesAllReactions(self, inputFile, processIndex):
        """
        Read trajectories from the file. Append them to the list of
        trajectories. Record the new RNG state for the specified process index.
        Below is the file format.

        <number of trajectories>
        for each trajectory:
          <list of initial MT 19937 state>
          if successful:
            <blank line>
            <list of initial amounts>
            <list of reaction indices>
            <list of reaction times>
          else:
            <error message>
        <list of final MT 19937 state>

        This function uses the following member variables:
        self.currentOutput
        self.listOfMt19937States
        """
        # Number of trajectories.
        numberOfTrajectories = int(inputFile.readline())
        for i in range(numberOfTrajectories):
            # CONTINUE: I ignore the initial RNG state for now.
            inputFile.readline()
            errorMessage = inputFile.readline().rstrip()
            if errorMessage == '':
                # List of initial populations.
                initial = map(float, inputFile.readline().rstrip().split())
                # List of reaction indices.
                indices = map(int, inputFile.readline().rstrip().split())
                # List of reaction times.
                times = map(float, inputFile.readline().rstrip().split())
                # Acquire the lock so that we can modify self.currentOutput.
                self.lock.acquire()
                self.currentOutput.appendInitialPopulations(initial)
                self.currentOutput.appendIndices(indices)
                self.currentOutput.appendTimes(times)
                self.successfulTrajectories += 1
                self.lock.release()
            else:
                self.lock.acquire()
                self.errorMessages.append(errorMessage)
                self.lock.release()
        # List of final MT 19937 state.
        state = map(int, inputFile.readline().rstrip().split())
        # If this is a stochastic method.
        if state:
            assert len(state) == 625
            # Record the RNG state. No need for a lock, each thread writes
            # to a different element of the list.
            self.listOfMt19937States[processIndex] = state
        # CONTINUE: Verify that the output is valid.

    def readHistogramFrames(self, inputFile, processIndex):
        """
        Read trajectories from the file. Append them to the list of
        trajectories. Record the new RNG state for the specified process index.
        Below is the file format.

        if successful:
          <blank line>
          <total number of trajectories>
          <histogram multiplicity>
          for each frame:
            for each recorded species:
              <cardinality>
              <sum of weights>
              <mean>
              <summed second centered moment>
              <lower bound>
              <bin width>
              for each histogram:
                <list of weighted probabilities>
        else:
          <error message>
        <list of final MT 19937 state>

        This function uses the following member variables:
        self.currentOutput
        self.listOfMt19937States
        """
        errorMessage = inputFile.readline().rstrip()
        if errorMessage == '':
            # Total number of trajectories.
            numberOfTrajectories = int(inputFile.readline())
            # The histogram multiplicity.
            multiplicity = int(inputFile.readline())
            assert multiplicity >= 1
            x = Histogram()
            # Acquire the lock so that we can modify self.currentOutput.
            self.lock.acquire()
            self.currentOutput.numberOfTrajectories += numberOfTrajectories
            self.successfulTrajectories += numberOfTrajectories
            # For each time frame.
            for frame in self.currentOutput.histograms:
                # For each recorded species.
                for h in frame:
                    x.read(inputFile, multiplicity)
                    h.merge(x)
            self.lock.release()
        else:
            self.lock.acquire()
            self.errorMessages.append(errorMessage)
            self.lock.release()
        # List of final MT 19937 state.
        state = map(int, inputFile.readline().rstrip().split())
        assert len(state) == 625
        # Record the RNG state.
        self.listOfMt19937States[processIndex] = state

    def readHistogramAverage(self, inputFile, processIndex):
        """
        Read trajectories from the file. Append them to the list of
        trajectories. Record the new RNG state for the specified process index.
        Below is the file format.

        if successful:
          <blank line>
          <total number of trajectories>
          <histogram multiplicity>
          for each recorded species:
            <cardinality>
            <sum of weights>
            <mean>
            <summed second centered moment>
            <lower bound>
            <bin width>
            for each histogram:
              <list of weighted probabilities>
        else:
          <error message>
        <list of final MT 19937 state>

        This function uses the following member variables:
        self.currentOutput
        self.listOfMt19937States
        """
        errorMessage = inputFile.readline().rstrip()
        if errorMessage == '':
            # Total number of trajectories.
            numberOfTrajectories = int(inputFile.readline())
            # The histogram multiplicity.
            multiplicity = int(inputFile.readline())
            assert multiplicity >= 1
            x = Histogram()
            # Acquire the lock so that we can modify self.currentOutput.
            self.lock.acquire()
            self.currentOutput.numberOfTrajectories += numberOfTrajectories
            self.successfulTrajectories += numberOfTrajectories
            # For each recorded species.
            for h in self.currentOutput.histograms:
                x.read(inputFile, multiplicity)
                h.merge(x)
            self.lock.release()
        else:
            self.lock.acquire()
            self.errorMessages.append(errorMessage)
            self.lock.release()
        # List of final MT 19937 state.
        state = map(int, inputFile.readline().rstrip().split())
        assert len(state) == 625
        # Record the RNG state.
        self.listOfMt19937States[processIndex] = state

    def importStatistics(self, fileName, modelId, methodId, recordedSpecies):
        # We know the output class from the selected method.
        method = self.methods[methodId]
        category = simulationMethods.categories[method.timeDependence]\
            [method.category]
        # Open the file.
        inputFile = open(fileName, 'r')
        # Call the appropriate importer.
        if category == 'Statistics, Transient Behavior':
            output = StatisticsFrames(recordedSpecies)
            output.setFrameTimes(self.computeFrameTimes(method))
            self.output[(modelId, methodId)] = output
            self.readStatisticsFrames(inputFile, output)
        elif category == 'Statistics, Steady State':
            output = StatisticsAverage(recordedSpecies)
            self.output[(modelId, methodId)] = output
            self.readStatisticsAverage(inputFile, output)
        else:
            # CONTINUE Errors intead of assertions.
            assert False
        inputFile.close()

    def importSuiteOfTrajectories(self, listOfFileNames, modelId, methodId):
        # Ensure there are sufficient RNG states.
        while len(self.listOfMt19937States) < len(listOfFileNames):
            self.listOfMt19937States.append(None)

        # We know the output class from the selected method.
        method = self.methods[methodId]
        category = simulationMethods.categories[method.timeDependence]\
            [method.category]

        # Record for readTrajectories().
        self.currentOutput = self.output[(modelId, methodId)]
        self.successfulTrajectories = 0
        
        # Import each file.
        processIndex = 0
        for fileName in listOfFileNames:
            # CONTINUE Errors intead of assertions.
            # Open the file.
            inputFile = open(fileName, 'r')
            if category in ('Histograms, Transient Behavior',
                            'Histograms, Steady State'):
                # The number of trajectories in the task.
                inputFile.readline()
            # CONTINUE: Check for consistency.
            # Skip the dictionary of information.
            inputFile.readline()
            # Call the appropriate importer.
            if category in ('Time Series, Uniform',
                            'Time Series, Deterministic'):
                self.readTimeSeriesFrames(inputFile, processIndex)
            elif category == 'Time Series, All Reactions':
                self.readTimeSeriesAllReactions(inputFile, processIndex)
            elif category == 'Histograms, Transient Behavior':
                self.readHistogramFrames(inputFile, processIndex)
            elif category == 'Histograms, Steady State':
                self.readHistogramAverage(inputFile, processIndex)
            else:
                assert False
            inputFile.close()
            processIndex += 1

    # CONTINUE HERE: REMOVE
    def importRecorded(self, inputFile):
        """Return the recorded species and the recorded reactions."""
        # Ignore output until we reach the recorded species.
        # number of species
        inputFile.readline()
        # number of reactions
        inputFile.readline()
        # number of species to record
        n = int(inputFile.readline())
        # list of species to record
        recordedSpecies = map(int, inputFile.readline().split())
        assert n == len(recordedSpecies)
        # number of reactions to record
        n = int(inputFile.readline())
        # list of reactions to record
        recordedReactions = map(int, inputFile.readline().split())
        assert n == len(recordedReactions)
        return recordedSpecies, recordedReactions

    # Export menu callbacks.

    # File I/O.

    def read(self, fileName):
        handler = ContentHandler()
        parse(open(fileName, 'r'), handler)
        self.models = handler.models
        self.methods = handler.methods
        self.output = handler.output
        if handler.seed is not None:
            self.seed = handler.seed
        self.listOfMt19937States = []
        for state in handler.listOfMt19937States:
            self.listOfMt19937States.append(state)

        # CONTINUE: Deprecated.
        # In versions prior to 1.0, for time series data that recorded all 
        # reaction events, the initial populations were not 
        # stored because there was no equilibration time. Thus we may need to
        # add these initial populations.
        for key in self.output:
            output = self.output[key]
            if output.__class__.__name__ == 'TimeSeriesAllReactions' and\
                    not output.initialPopulations:
                model = self.models[key[0]]
                # Evaluate the model to compute the initial amounts.
                error = model.evaluate()
                if error:
                    handler.errors += error
                    continue
                initialPopulations = [model.species[id].initialAmountValue
                                      for id in model.speciesIdentifiers]
                for i in range(len(output.indices)):
                    output.appendInitialPopulations(initialPopulations)

        return handler.errors

    def importSbmlModel(self, fileName):
        """Import the SBML model. Return a tuple of the model identifier
        and an error string."""
        # Import the model.
        handler = ContentHandlerSbml()
        parse(open(fileName, 'r'), handler)
        model = handler.model
        # Check that the identifier is distinct.
        if model.id in self.models:
            # If not, change it.
            model.id = getNewIntegerString(self.models.keys())
        # Add the reverse reactions, if any.
        model.addReverseReactions()
        # Try converting custom propensity function to mass action equivalents.
        model.convertCustomToMassAction()
        # Add the model to the dictionary.
        self.models[model.id] = model
        # Build the error message.
        messages = []
        if handler.errors:
            messages.append('Errors:')
            messages.append(handler.errors)
        if handler.warnings:
            messages.append('Warnings:')
            messages.append(handler.warnings)
        # Return the new model's key and the error message.
        return (model.id, '\n'.join(messages))

    def importTextModel(self, fileName):
        """Import the model. If successful, return its identifier. Otherwise
        return None."""
        from io.readModelText import readModelText
        # Import the model.
        model = readModelText(open(fileName, 'r'))
        # If reading the model was successful.
        if model:
            # Get a distinct identifier.
            model.id = getNewIntegerString(self.models.keys())
            # Add the model to the dictionary.
            self.models[model.id] = model
            # Return the new model's key.
            return model.id
        else:
            return None

    def write(self, outputFile):
        writer = XmlWriter(outputFile)
        writer.beginDocument()
        writer.beginElement('cain', {'version':self.version})
        if self.models:
            writer.beginElement('listOfModels')
            for model in self.models.values():
                model.writeXml(writer)
            writer.endElement()
        if self.methods:
            writer.beginElement('listOfMethods')
            for p in self.methods.values():
                p.writeXml(writer)
            writer.endElement()
        if self.output:
            writer.beginElement('listOfOutput')
            for key in self.output:
                self.output[key].writeXml(writer, key[0], key[1])
            writer.endElement()

        # Random
        writer.beginElement('random', {'seed':str(self.seed)})
        for state in self.listOfMt19937States:
            writer.writeElement('stateMT19937', data=
                                ' '.join([str(x) for x in state]))
        writer.endElement() # random

        writer.endElement() # cain
        writer.endDocument()

    def writeSbml(self, modelId, outputFile, version):
        assert modelId in self.models
        writer = XmlWriter(outputFile)
        writer.beginDocument()
        self.models[modelId].writeSbml(writer, version)
        writer.endDocument()

    def exportGnuplot(self, modelId, methodId, baseName, fileName):
        """Choose the right function for the output class."""
        output = self.output[(modelId, methodId)]
        if output.__class__.__name__ == 'TimeSeriesFrames':
            self.exportGnuplotDataTimeSeriesFrames(modelId, methodId, fileName)
            self.exportGnuplotScriptTrajectory(modelId, methodId, baseName)
        elif output.__class__.__name__ == 'TimeSeriesAllReactions':
            self.exportGnuplotDataTimeSeriesAllReactions(modelId, methodId,
                                                         fileName)
            self.exportGnuplotScriptTrajectory(modelId, methodId, baseName)
        elif output.__class__.__name__ == 'HistogramFrames':
            self.exportGnuplotDataHistogramFrames(modelId, methodId, baseName)
            self.exportGnuplotScriptHistogramFrames(modelId, methodId, baseName)
        elif output.__class__.__name__ == 'HistogramAverage':
            self.exportGnuplotDataHistogramAverage(modelId, methodId, baseName)
            self.exportGnuplotScriptHistogramAverage(modelId, methodId,
                                                     baseName)
        else:
            assert False

    def exportGnuplotDataTimeSeriesFrames(self, modelId, methodId, fileName):
        model = self.models[modelId]
        output = self.output[(modelId, methodId)]
        # Construct the writer.
        writer = csv.writer(open(fileName, 'w'), dialect="gnuplot")
        # Write the comment header.
        header = ['#Time', 'Reaction Count']
        for i in output.recordedSpecies:
            header.append(model.speciesIdentifiers[i])
        for i in output.recordedReactions:
            header.append(model.reactions[i].id)
        writer.writerow(header)
        for i in range(len(output.populations)):
            # Write each data row.
            for j in range(len(output.frameTimes)):
                row = [output.frameTimes[j]]
                count = sum(output.reactionCounts[i][j])
                row.append(count)
                row.extend(output.populations[i][j])
                row.extend(output.reactionCounts[i][j])
                writer.writerow(row)
            # Blank line between frames.
            writer.writerow([])

    def exportGnuplotDataTimeSeriesAllReactions(self, modelId, methodId,
                                                fileName):
        from TrajectoryCalculator import TrajectoryCalculator
        model = self.models[modelId]
        output = self.output[(modelId, methodId)]
        # Construct the writer.
        writer = csv.writer(open(fileName, 'w'), dialect="gnuplot")
        # Convert all reaction trajectories to frame trajectories.
        trajectoryCalculator = TrajectoryCalculator(model)
        # Write the comment header.
        header = ['#Time', 'Reaction Count']
        for id in model.speciesIdentifiers:
            header.append(id)
        for reaction in model.reactions:
            header.append(reaction.id)
        writer.writerow(header)
        for index in range(len(output.indices)):
            # Convert to frames. Include the start and end times.
            times, populations, reactionCounts =\
                trajectoryCalculator.makeFramesAtReactionEvents\
                (output, index, True, True)
            # Double to get a step function.
            # Write each data row.
            for i in range(len(times)):
                row = [times[i]]
                count = sum(reactionCounts[i])
                row.append(count)
                row.extend(populations[i])
                row.extend(reactionCounts[i])
                writer.writerow(row)
                if i != len(times) - 1:
                    row[0] = times[i+1]
                    writer.writerow(row)
            # Blank line between frames.
            writer.writerow([])

    def exportGnuplotDataHistogramFrames(self, modelId, methodId, baseName):
        model = self.models[modelId]
        output = self.output[(modelId, methodId)]
        # For each frame.
        for frameIndex in range(len(output.frameTimes)):
            # For each recorded species.
            for recordedIndex in range(len(output.recordedSpecies)):
                # The species identifier.
                speciesIndex = output.recordedSpecies[recordedIndex]
                speciesId = model.speciesIdentifiers[speciesIndex]
                # The file name.
                fileName = baseName + '-' + str(frameIndex) + '-' +\
                    speciesId + '.dat'
                # Construct the CSV writer.
                writer = csv.writer(open(fileName, 'w'), dialect="gnuplot")
                # The histogram for this frame and species.
                h = output.histograms[frameIndex][recordedIndex]
                # The PMF.
                p = h.getPmf()
                # For each bin.
                for i in range(len(p)):
                    # The center of the bin and the PMF.
                    writer.writerow([h.lowerBound + (i + 0.5) * h.getWidth(),
                                     p[i]])

    def exportGnuplotDataHistogramAverage(self, modelId, methodId, baseName):
        model = self.models[modelId]
        output = self.output[(modelId, methodId)]
        # For each recorded species.
        for recordedIndex in range(len(output.recordedSpecies)):
            # The species identifier.
            speciesIndex = output.recordedSpecies[recordedIndex]
            speciesId = model.speciesIdentifiers[speciesIndex]
            # The file name.
            fileName = baseName + '-' + speciesId + '.dat'
            # Construct the CSV writer.
            writer = csv.writer(open(fileName, 'w'), dialect="gnuplot")
            # The histogram for this frame and species.
            h = output.histograms[recordedIndex]
            # The PMF.
            p = h.getPmf()
            # For each bin.
            for i in range(len(p)):
                # The center of the bin and the PMF.
                writer.writerow([h.lowerBound + (i + 0.5) * h.getWidth(), p[i]])

    def exportGnuplotScriptTrajectory(self, modelId, methodId, baseName):
        assert modelId in self.models
        model = self.models[modelId]
        output = self.output[(modelId, methodId)]
        f = open(baseName + '.gnu', 'w')
        dataName = baseName + '.dat'
        # The plot size.
        f.write('set size %s, %s\n' %
                (self.preferences.data['gnuplot']['X Scale'],
                 self.preferences.data['gnuplot']['Y Scale']))

        # If any species were recorded.
        if output.recordedSpecies:
            # Plot the populations together.
            offset = 3
            f.write('set title "' + model.id + '"\n')
            f.write('set xlabel "Time"\n')
            f.write('set ylabel "Population"\n')
            f.write('set key top\n')
            f.write('set terminal jpeg\n')
            f.write('set output "' + baseName + '-Populations.jpg"\n')
            f.write('plot \\\n')
            style = self.preferences.data['gnuplot']['Style']
            for n in range(len(output.recordedSpecies)):
                index = output.recordedSpecies[n]
                identifier = model.speciesIdentifiers[index]
                f.write('"' + dataName + '" using 1:' + str(n + offset) + 
                        ' title "' + identifier + 
                        '" with ' + style)
                if n != len(output.recordedSpecies) - 1:
                    f.write(',\\\n')
                else:
                    f.write('\n')
            f.write('\n')
        # Plot each recorded species separately.
        for n in range(len(output.recordedSpecies)):
            index = output.recordedSpecies[n]
            identifier = model.speciesIdentifiers[index]
            f.write('set title "' + model.id + '"\n')
            f.write('set xlabel "Time"\n')
            f.write('set ylabel "' + identifier + '"\n')
            f.write('set key off\n')
            f.write('set terminal jpeg\n')
            f.write('set output "' + baseName + '-' + identifier + '.jpg"\n')
            f.write('plot \\\n')
            f.write('"' + dataName + '" using 1:' + str(n + offset) + 
                    ' title "' + identifier + '" with ' + style + '\n\n')

        # If any reactions were recorded.
        if output.recordedReactions:
            # Plot the reaction counts.
            offset = 3 + len(output.recordedSpecies)
            f.write('set ylabel "Reaction Counts"\n')
            f.write('set key left\n')
            f.write('set output "' + baseName + '-Reactions.jpg"\n')
            f.write('plot \\\n')
            for n in range(len(output.recordedReactions)):
                index = output.recordedReactions[n]
                identifier = model.reactions[index].id
                f.write('"' + dataName + '" using 1:' + str(n + offset) + 
                        ' title "' + identifier + '" with ' + style)
                if n != len(model.reactions) - 1:
                    f.write(',\\\n')
                else:
                    f.write('\n')
        # Plot the the individual reactions.
        for n in range(len(output.recordedReactions)):
            index = output.recordedReactions[n]
            identifier = model.reactions[index].id
            f.write('set ylabel "' + identifier + '"\n')
            f.write('set key off\n')
            f.write('set output "' + baseName + '-' + identifier + '.jpg"\n')
            f.write('plot \\\n')
            f.write('"' + dataName + '" using 1:' + str(n + offset) + 
                    ' title "' + identifier + '" with ' + style + '\n\n')
        f.close()

    def exportGnuplotScriptHistogramFrames(self, modelId, methodId, baseName):
        model = self.models[modelId]
        output = self.output[(modelId, methodId)]
        f = open(baseName + '.gnu', 'w')
        # The common plotting commands.
        f.write('set size %s, %s\n' %
                (self.preferences.data['gnuplot']['X Scale'],
                 self.preferences.data['gnuplot']['Y Scale']))
        f.write('set xlabel "Population"\n')
        f.write('set ylabel "Probability"\n')
        f.write('set key off\n')
        f.write('set style fill solid 0.5\n')
        f.write('set terminal jpeg\n\n')
        # For each frame.
        for frameIndex in range(len(output.frameTimes)):
            # For each recorded species.
            for recordedIndex in range(len(output.recordedSpecies)):
                # The species identifier.
                speciesIndex = output.recordedSpecies[recordedIndex]
                speciesId = model.speciesIdentifiers[speciesIndex]
                # The file names.
                suffix = '-' + str(frameIndex) + '-' + speciesId
                dataName = baseName + suffix + '.dat'
                outputName = baseName + suffix + '.jpg'
                # The commands for this histogram.
                f.write('set title "' + model.id + ', Time = ' + 
                        str(output.frameTimes[frameIndex]) + '"\n')
                f.write('set output "' + outputName + '"\n')
                f.write('plot "' + dataName + '" using 1:2 with boxes\n\n')

    def exportGnuplotScriptHistogramAverage(self, modelId, methodId, baseName):
        model = self.models[modelId]
        output = self.output[(modelId, methodId)]
        f = open(baseName + '.gnu', 'w')
        # The common plotting commands.
        f.write('set size %s, %s\n' %
                (self.preferences.data['gnuplot']['X Scale'],
                 self.preferences.data['gnuplot']['Y Scale']))
        f.write('set xlabel "Population"\n')
        f.write('set ylabel "Probability"\n')
        f.write('set key off\n')
        f.write('set style fill solid 0.5\n')
        f.write('set terminal jpeg\n\n')
        f.write('set title "' + model.id + '"\n')
        # For each recorded species.
        for recordedIndex in range(len(output.recordedSpecies)):
            # The species identifier.
            speciesIndex = output.recordedSpecies[recordedIndex]
            speciesId = model.speciesIdentifiers[speciesIndex]
            # The file names.
            suffix = '-' + speciesId
            dataName = baseName + suffix + '.dat'
            outputName = baseName + suffix + '.jpg'
            # The commands for this histogram.
            f.write('set output "' + outputName + '"\n')
            f.write('plot "' + dataName + '" using 1:2 with boxes\n\n')

    # Validity.

    def hasErrorsInModel(self, modelId, methodId):
        """Return None if the model is valid. Otherwise return an error
        message."""
        if methodId:
            isDiscrete = self.methods[methodId].isDiscrete()
        else:
            isDiscrete = False
        return self.models[modelId].hasErrors(isDiscrete)

    def hasErrorsInMethod(self, id):
        """Return None if the methods are valid. Otherwise
        return an error message."""
        return self.methods[id].hasErrors()

def main():
    state = State()
    state.read('../examples/cain/DecayingDimerizing.xml')
    state.write(open('tmp.xml', 'w'))

if __name__ == '__main__':
    main()
