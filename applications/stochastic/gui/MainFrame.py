"""The main frame."""

# If we are running the unit tests.
import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')

import wx
import wx.html

import math
import os
import os.path
import re
import time
import threading
import webbrowser
import platform
import tempfile
import urllib2

from About import About
from IdentifierListEditor import IdentifierListEditor
from ModelEditor import ModelEditor, EVT_SPECIES_OR_REACTIONS_MODIFIED
from Record import Record
from MethodEditor import MethodEditor
from Launcher import Launcher
from TrajectoriesList import TrajectoriesList
from PlotTimeSeries import closeAll
from resourcePath import resourcePath
from DuplicateDialog import DuplicateDialog
from StateModified import EVT_STATE_MODIFIED
from io.SpeciesTextParser import SpeciesTextParser
from io.ReactionTextParser import ReactionTextParser
from io.TimeEventTextParser import TimeEventTextParser
from io.TriggerEventTextParser import TriggerEventTextParser
from io.ParameterTextParser import ParameterTextParser
from io.CompartmentTextParser import CompartmentTextParser
from state.State import State
import state.simulationMethods as simulationMethods
from messages import UpdateVersionFrame, ScrolledMessageFrame,\
     CompilationError, CompilingMessage, truncatedMessageBox,\
     truncatedErrorBox, openWrite

class CompileSolverThread(threading.Thread):
    """Thread for compiling a solver."""
    def __init__(self, application, compilationArguments,
                 successFunction, successArguments,
                 failureFunction, failureArguments):
        threading.Thread.__init__(self)
        self.application = application
        self.compilationArguments = compilationArguments
        self.successFunction = successFunction
        self.successArguments = successArguments
        self.failureFunction = failureFunction
        self.failureArguments = failureArguments

    def run(self):
        error = self.application.state.compileSolver(*self.compilationArguments)
        wx.CallAfter(self.application.destroyCompilingMessage)
        if error:
            wx.CallAfter(self.application.showCompilationErrors, error)
            wx.CallAfter(self.failureFunction, *self.failureArguments)
        else:
            wx.CallAfter(self.successFunction, *self.successArguments)

def computePartition(x, n, i):
    """Compute the i_th fair partition of x into n parts.

    x is the number to partition.
    n is the number of partitions.
    i is the partition index.
    """
    p = x // n
    if i < x % n:
        p += 1
    return p

class MainFrame(wx.Frame):
    """The main frame."""
    
    def __init__(self, parent=None):
        """Constructor."""
        # Data.
        self.lock = threading.Lock()
        self.title = 'Cain'
        self.filename = ''
        self.state = State()
        self.isModified = False
        # Dictionary with model identifiers as the keys. The value type is 
        # a list of the species table, reaction table, time events table,
        # trigger events table, parameters table, and compartments table.
        self.modelTables = {}
        # Widgets.
        # Note: A small screen is typically 1280x800, but may be as small
        # as 1024x600 for a netbook.
        wx.Frame.__init__(self, parent, -1, self.title, size=(1280, 750))
        self.initializeStatusBar()
        self.createMenuBar()
        self.createToolBar()

        self.splitter = wx.SplitterWindow(self,
                                          style=wx.SP_NOBORDER|wx.SP_3DSASH)
        panel = wx.Panel(self.splitter)
        self.modelsList = \
            IdentifierListEditor(panel, 'Models',
                                 insert=self.modelInsert,
                                 clone=self.modelClone, 
                                 duplicate=self.modelDuplicate,
                                 edit=self.modelEdit,
                                 delete=self.modelDelete,
                                 toolTip='The list of models. Use + to add a new model. You must select a model before launching a simulation.')
        self.Bind(EVT_STATE_MODIFIED, self.onStateModified, self.modelsList)

        self.methodsList = \
            IdentifierListEditor(panel, 'Methods',
                                 insert=self.methodInsert,
                                 clone=self.methodClone,
                                 duplicate=None,
                                 edit=self.methodEdit,
                                 delete=self.methodDelete,
                                 toolTip='The list of methods. Use + to add a new method. You must select a method before launching a simulation.')
        self.Bind(EVT_STATE_MODIFIED, self.onStateModified, self.methodsList)

        # I need to construct the launcher before the method editor because
        # the latter updates the former when a new method is selected.
        self.launcher = Launcher(panel, self, self.directLaunch,
                                 self.launchCustomSimulations, 
                                 self.stopSimulation, self.killSimulation, 
                                 self.saveExecutable, self.exportJobs,
                                 self.exportMathematica,
                                 self.importSolution)
        self.methodEditor = MethodEditor(panel, self)
        self.Bind(EVT_STATE_MODIFIED, self.onStateModified, self.methodEditor)
        self.record = Record(panel)
        self.trajectoriesList = TrajectoriesList(panel, self)
        self.Bind(EVT_STATE_MODIFIED, self.onStateModified,
                  self.trajectoriesList)
        self.modelEditor = ModelEditor(self.splitter)
        self.Bind(EVT_SPECIES_OR_REACTIONS_MODIFIED,
                  self.onSpeciesOrReactionsModified, self.modelEditor)
        self.Bind(EVT_STATE_MODIFIED, self.onStateModified, self.modelEditor)

        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.onModelSelected, 
                  self.modelsList.list)
        self.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.onModelDeselected, 
                  self.modelsList.list)

        self.Bind(wx.EVT_LIST_ITEM_SELECTED, 
                  self.onMethodSelected, 
                  self.methodsList.list)
        self.Bind(wx.EVT_LIST_ITEM_DESELECTED, 
                  self.onMethodDeselected, 
                  self.methodsList.list)

        self.Bind(wx.EVT_CLOSE, self.onCloseWindow)

        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(self.modelsList, 2, wx.EXPAND)
        row.Add(self.methodsList, 2, wx.EXPAND)
        row.Add(self.methodEditor, 0, wx.EXPAND)
        row.Add(self.record, 3, wx.EXPAND)
        row.Add(self.launcher, 0, wx.EXPAND)
        row.Add(self.trajectoriesList, 4, wx.EXPAND)
        panel.SetSizer(row)
        self.splitter.SetMinimumPaneSize(20)
        # Give equal expanding space to the top and bottom.
        self.splitter.SetSashGravity(0.5)
        self.splitter.SplitHorizontally(panel, self.modelEditor)
        self.clearModel()
        self.clearMethod()
        self.updateTrajectories()
        # Set up the help system.
        wx.FileSystem.AddHandler(wx.ZipFSHandler())
        # No bookmarks.
        self.help = wx.html.HtmlHelpController\
            (wx.html.HF_TOOLBAR | wx.html.HF_CONTENTS | wx.html.HF_INDEX |
             wx.html.HF_SEARCH | wx.html.HF_PRINT)
        cwd = os.getcwd()
        # CONTINUE REMOVE
        #self.help.SetTempDir(os.path.join(cwd, 'help'))
        self.help.SetTempDir(tempfile.mkdtemp())
        self.help.AddBook(os.path.join(cwd, 'help', 'cain.htb'))
        # CONTINUE REMOVE
        if False:
            wx.MessageBox('Unable to open help file. Working directory = '
                          + cwd,
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
        wx.CallAfter(self.setSashes)
        # In initializing the widgets, the state is set as being modified.
        # Reset to an unmodified state.
        wx.CallAfter(self.setNotModified)
        # Check the application version.
        wx.CallAfter(self.checkVersion)

    def setSashes(self):
        height = self.GetSize()[1]
        self.splitter.SetSashPosition(3*height//10)

    def checkVersion(self):
        try:
            url = 'http://cain.sourceforge.net/version.txt'
            currentVersion = urllib2.urlopen(url, timeout=2).read()
            c = currentVersion.split('.')
            t = self.state.version.split('.')
            if int(t[0]) < int(c[0]) or \
               (int(t[0]) == int(c[0]) and int(t[1]) < int(c[1])):
                UpdateVersionFrame(self).Show()
        except:
            pass

    def setNotModified(self):
        self.isModified = False

    def initializeStatusBar(self):
        self.statusBar = self.CreateStatusBar()

    def menuData(self):
        return [("&File", (
                    ("&New", "New simulation", self.onNew),
                    ("&Open", "Open a file", self.onOpen),
                    ("&Save", "Save a file", self.onSave),
                    ("Save &As", "Save as", self.onSaveAs),
                    ("", "", ""),
                    ("&Import SBML", "Import SBML model", self.onImportSbml),
                    ("Import &Text Model", "Import text model", self.onImportTextModel),
                    ("&Export SBML", "Export SBML model", self.onExportSbml),
                    ("&Export CMDL", "Export CMDL model", self.onExportCmdl),
                    ("", "", ""),
                    ("&About...", "Show about window", self.onAbout),
                    ("&Quit", "Quit", self.onCloseWindow))),
                ("&Help", (
                    ("&Help", "Documentation", self.onHelp),))]

    def createMenuBar(self):
        menuBar = wx.MenuBar()
        for eachMenuData in self.menuData():
            menuLabel = eachMenuData[0]
            menuItems = eachMenuData[1]
            menuBar.Append(self.createMenu(menuItems), menuLabel)
        self.SetMenuBar(menuBar)

    def createToolBar(self):
        toolBar = self.CreateToolBar()
        # The default bitmap size is 16 by 15 pixels.

        # CONTINUE: The long help strings don't show in the status bar on OS X.
        # CONTINUE: Try wx.Bitmap(os.path.join(resourcePath, 'gui/icons/16x16/filenew.png'))
        # CONTINUE I could use AddLabelTool instead.
        # File tools.
        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/filenew.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        tool = toolBar.AddTool(-1, bmp, shortHelpString='New',
                                longHelpString='Clear all data and start a new problem.')
#        tool = toolBar.AddLabelTool(-1, 'New', bmp, shortHelp='New',
#                                     longHelp='Clear all data and start a new problem.')
        self.Bind(wx.EVT_MENU, self.onNew, tool)

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/fileopen.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        tool = toolBar.AddTool(-1, bmp, shortHelpString='Open',
                                longHelpString='Open a file.')
        self.Bind(wx.EVT_MENU, self.onOpen, tool)

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/filesave.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        tool = toolBar.AddTool(-1, bmp, shortHelpString='Save',
                                longHelpString='Save the models, methods, and simulation output.')
        self.Bind(wx.EVT_MENU, self.onSave, tool)

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/filesaveas.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        tool = toolBar.AddTool(-1, bmp, shortHelpString='Save as',
                                longHelpString='Save the models, methods, and simulation output with a new file name.')
        self.Bind(wx.EVT_MENU, self.onSaveAs, tool)

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/exit.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        tool = toolBar.AddTool(-1, bmp, shortHelpString='Quit',
                                longHelpString='Quit Cain.')
        self.Bind(wx.EVT_MENU, self.onCloseWindow, tool)

        toolBar.AddSeparator()
        
        # Random number generator.

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/dice.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        tool = toolBar.AddTool(-1, bmp, shortHelpString='Seed the RNG',
                                longHelpString='Seed the Mersenne Twister')
        self.Bind(wx.EVT_MENU, self.onSeed, tool)

        toolBar.AddSeparator()

        # Help tools.
        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/preferences-system.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        tool = toolBar.AddTool(-1, bmp, shortHelpString='Preferences',
                                longHelpString='Open the preferences dialog.')
        self.Bind(wx.EVT_MENU, self.onPreferences, tool)

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/help.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        tool = toolBar.AddTool(-1, bmp, shortHelpString='Help',
                                longHelpString='Open the help browser.')
        self.Bind(wx.EVT_MENU, self.onHelp, tool)

        toolBar.Realize()

    def createMenu(self, menuData):
        menu = wx.Menu()
        for eachItem in menuData:
            if len(eachItem) == 2:
                label = eachItem[0]
                subMenu = self.createMenu(eachItem[1])
                menu.AppendMenu(wx.NewId(), label, subMenu)
            else:
                self.createMenuItem(menu, *eachItem)
        return menu

    def createMenuItem(self, menu, label, status, handler, kind=wx.ITEM_NORMAL):
        if label:
            menuItem = menu.Append(-1, label, status, kind)
            self.Bind(wx.EVT_MENU, handler, menuItem)
        else:
            menu.AppendSeparator()

    # File menu callbacks.

    def clear(self):
        # Kill any running simulation.
        if self.launcher.isRunning:
            if wx.MessageBox('There is a running simulation. Do you want to continue and kill the simulation?', 'Warning', wx.YES|wx.NO) == wx.YES:
                self.killSimulation()
            else:
                return
        self.filename = ''
        self.SetTitle(self.title + ' -- ' + self.filename)
        self.state.clear()
        self.clearModelList()
        self.clearMethodList()
        self.updateTrajectories()
        self.isModified = False

    def saveChanges(self, message):
        """If there are modifications check to see if the user wants to save
        them. Return true if there are no changes, they want to discard the
        changes, or if they successfully save the changes. Otherwise (if
        they hit cancel or are unable to save the changes) return false."""
        # If there are unsaved changes.
        if self.isModified:
            # Check to see if they want to save the changes.
            dialog = wx.MessageDialog(self, message, 'Save changes?',
                                      wx.YES_NO|wx.CANCEL)
            result = dialog.ShowModal()
            dialog.Destroy()
            if result == wx.ID_CANCEL:
                return False
            if result == wx.ID_YES:
                # If the state is not successfully be saved.
                if not self.onSave(None):
                    return False
        return True

    def onNew(self, event):
        """Make an empty model."""
        if self.saveChanges('Do you want to save your changes before clearing the state?'):
            self.clear()

    def onOpen(self, event):
        if not self.saveChanges('Do you want to save your changes before opening a new file?'):
            return
        wildcardXml = "XML files (*.xml)|*.xml|" + \
            "All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Open a file...", os.getcwd(),
                               style=wx.OPEN, wildcard=wildcardXml)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            self.readFile(dialog.GetPath())
        dialog.Destroy()

    def onSave(self, event):
        """Return true if the file is saved."""
        if not self.filename:
            return self.onSaveAs(event)
        else:
            if not self.syncSelectedModelAndMethod\
                    ('Error! Correct before saving.') or\
                    not self.parseModelsAndMethods\
                    ('Error! Correct before saving.'):
                return False
            return self.saveFile()

    def onSaveAs(self, event):
        """Return true if the file is saved."""
        if not self.syncSelectedModelAndMethod\
                ('Error! Correct before saving.') or\
                not self.parseModelsAndMethods\
                ('Error! Correct before saving.'):
            return False
        wildcardXml = "XML files (*.xml)|*.xml|" + \
            "All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Save as...", os.getcwd(),
                               style=wx.SAVE|wx.OVERWRITE_PROMPT,
                               wildcard=wildcardXml)
        result = False
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            filename = dialog.GetPath()
            if not os.path.splitext(filename)[1]:
                filename = filename + '.xml'
            self.filename = filename
            result = self.saveFile()
            if result:
                self.SetTitle(self.title + ' -- ' + self.filename)
        dialog.Destroy()
        return result

    def onImportSbml(self, event):
        wildcardXml = "XML files (*.xml)|*.xml|" + \
            "All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Import SBML model...", os.getcwd(),
                               style=wx.OPEN, wildcard=wildcardXml)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            self.importSbml(dialog.GetPath())
        dialog.Destroy()

    def onImportTextModel(self, event):
        wildcard = "Text files (*.txt)|*.txt|" + \
            "All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Import text model...", os.getcwd(),
                               style=wx.OPEN, wildcard=wildcard)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            self.importTextModel(dialog.GetPath())
        dialog.Destroy()

    def onExportSbml(self, event):
        id = self.modelsList.getSelectedText()
        if id == None:
            wx.MessageBox('No model is selected.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        self.syncModel(id)
        if not self.parseModel(id, None, 'Error! Cannot export model.'):
            return
        wildcardXml = "XML files (*.xml)|*.xml|" + "All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Export SBML model...", os.getcwd(),
                               style=wx.SAVE|wx.OVERWRITE_PROMPT,
                               wildcard=wildcardXml)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            filename = dialog.GetPath()
            if not os.path.splitext(filename)[1]:
                filename = filename + '.xml'
            self.exportSbml(id, filename)
        dialog.Destroy()

    def onExportCmdl(self, event):
        id = self.modelsList.getSelectedText()
        if id == None:
            wx.MessageBox('No model is selected.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        self.syncModel(id)
        if not self.parseModel(id, None, 'Error! Cannot export model.'):
            return
        wildcardXml = "CMDL files (*.cmdl)|*.cmdl|" + "All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Export CMDL model...", os.getcwd(),
                               style=wx.SAVE|wx.OVERWRITE_PROMPT,
                               wildcard=wildcardXml)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            filename = dialog.GetPath()
            if not os.path.splitext(filename)[1]:
                filename = filename + '.cmdl'
            self.exportCmdl(id, filename)
        dialog.Destroy()

    def onAbout(self, event): 
        dialog = About()
        dialog.ShowModal()
        dialog.Destroy()

    def onCloseWindow(self, event):
        """Close all of the matplotlib figures before destroying this window."""
        if self.saveChanges('Do you want to save your changes before quitting?'):
            # Close the children of the simulation output panel.
            self.trajectoriesList.tearDown()
            # Close the plotting windows.
            closeAll()
            self.Destroy()

    def onStateModified(self, event):
        self.isModified = True

    # Random number menu callbacks.

    def onSeed(self, event):
        # Get the seed.
        s = wx.GetTextFromUser('Enter a 32-bit unsigned integer seed.',
                               'Mersenne Twister Seed', '0', self)
        # Check the validity.
        if not s:
            return
        try:
            seed = int(s)
        except:
            wx.MessageBox(s + ' is not an integer.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        if not (0 <= seed and seed < 2**32):
            wx.MessageBox(s + ' is not a 32-bit unsigned integer.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        # Set the seed.
        self.state.seedMt19937(seed)

    # Help menu callbacks.

    def onPreferences(self, event):
        self.state.preferences.openDialog()

    def onHelp(self, event):
        self.displayHelp()

    def displayHelp(self, x=None):
        if x:
            self.help.Display(x)
        else:
            self.help.DisplayContents()
        # Call this here because the window needs to be created first.
        self.help.GetHelpWindow().Bind(wx.html.EVT_HTML_LINK_CLICKED,
                                       self.onLinkClicked)

    def onLinkClicked(self, event):
        """Open external resources in a browser."""
        href = event.GetLinkInfo().GetHref()
        if len(href) > 4 and href[0:4] == 'http':
            version = platform.python_version_tuple()
            if 10 * int(version[0]) + int(version[1]) >= 25:
                # Open the page in a new tab.
                # This function is new in python 2.5.
                webbrowser.open_new_tab(href)
            else:
                # Open the page in a new window.
                webbrowser.open_new(href)
        else:
            event.Skip()
        
    # Simulation functions.

    def cacheModel(self):
        """Get the selected model. Return True if a model is selected."""
        # Store for use in finishSimulations().
        self.modelId = self.modelsList.getSelectedText()
        if not self.modelId:
            wx.MessageBox('No model selected.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return False
        self.syncModel(self.modelId)
        return True

    def cacheModelAndMethod(self):
        """Get the selected model and method. Return True if they are valid."""
        if not self.cacheModel():
            return False
        # Store for use in finishSimulations().
        self.methodId = self.methodsList.getSelectedText()
        if not self.methodId:
            wx.MessageBox('No method is selected.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return False
        if not self.syncMethod\
                (self.methodId, 'Error! Bad simulation parameters.') or\
                not self.parseModel(self.modelId, self.methodId,
                                    'Error! Bad model.')\
                or not self.parseMethods\
                (self.methodId, 'Error! Bad simulation parameters.'):
            return False
        return True

    def launchSimulations(self, useCustomSolver):
        method = self.state.methods[self.methodId]
        if simulationMethods.isStochastic(method.timeDependence,
                                          method.category):
            self.numberOfTrajectories = self.launcher.trajectories.GetValue()
            numberOfProcesses = self.launcher.cores.GetValue()
            if self.numberOfTrajectories > numberOfProcesses:
                trajectoriesPerTask =\
                    int(pow(float(self.numberOfTrajectories) /
                            numberOfProcesses,
                            self.launcher.getGranularity()))
            else:
                trajectoriesPerTask = 1
        else:
            self.numberOfTrajectories = 1
            numberOfProcesses = 1
            trajectoriesPerTask = 1
        assert numberOfProcesses >= 1
        self.launcher.gauge.SetRange(self.numberOfTrajectories)
        self.launcher.gauge.SetValue(0)
        self.trajectoryCount = 0
        self.startTime = time.time()
        recordedSpecies, recordedReactions = self.record.get()
        error = self.reportRecordedErrors(method, recordedSpecies,
                                          recordedReactions)
        if error:
            wx.MessageBox(error, 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            self.launcher.isRunning = False
            self.launcher.update()
            return
        error = self.state.launchSuiteOfSimulations\
                (self, self.modelId, self.methodId, recordedSpecies,
                 recordedReactions, numberOfProcesses,
                 self.numberOfTrajectories, trajectoriesPerTask,
                 self.launcher.getNiceIncrement(), useCustomSolver)
        if error:
            wx.MessageBox(error, 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            self.launcher.isRunning = False
            self.launcher.update()
            return

    def reportRecordedErrors(self, method, recordedSpecies, recordedReactions):
        """Return an error message if the recorded species and recorded
        reactions are not valid for the specified method. If there are no 
        errors, return None."""
        category = simulationMethods.categories[method.timeDependence]\
            [method.category]
        if category in ('Time Series, Uniform', 'Time Series, Deterministic')\
                and not (recordedSpecies or recordedReactions):
            return 'No species or reactions are being recorded.'
        # No need to check TimeSeriesAllReactions.
        if category in ('Histograms, Transient Behavior',
                        'Histograms, Steady State',
                        'Statistics, Transient Behavior',
                        'Statistics, Steady State') and not recordedSpecies:
            return 'No species are being recorded.'
        # No errors.
        return None

    def evaluateModel(self):
        """Return True if the model can be evaluated. 
        Call cacheModel() or cacheModelAndMethod() before calling this
        function."""
        error = self.state.evaluateModel(self.modelId)
        if error:
            truncatedErrorBox(error)
            return False
        return True

    def hasLaunchErrors(self):
        """Return True if there are errors that prevent a launch."""
        if not self.cacheModelAndMethod() or not self.evaluateModel():
            return True
        method = self.state.methods[self.methodId]
        # If the method is stochastic, check that the initial amounts are 
        # integers.
        if simulationMethods.isStochastic(method.timeDependence,
                                          method.category) and\
                not self.state.models[self.modelId].hasIntegerInitialAmounts():
            wx.MessageBox('For stochastic methods the initial amounts must be integer-valued.', 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return True
        # If the method is deterministic, check that they have not already
        # generated a trajectory.
        if not simulationMethods.isStochastic(method.timeDependence,
                                              method.category) and\
                (self.modelId, self.methodId) in self.state.output:
            wx.MessageBox('This is a deterministic method.\nThe trajectory has already been generated.', 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return True
        # If the method does not support events, check that the model does
        # not have events.
        model = self.state.models[self.modelId]
        if not simulationMethods.supportsEvents(method.timeDependence) and\
           (model.timeEvents or model.triggerEvents):
            wx.MessageBox('This model has events, but the method does not support them.\nUse a solver in the "Use Events" category.', 'Error!',
                          style=wx.OK|wx.ICON_EXCLAMATION)
            return True
        return False

    def launchCustomSimulations(self):
        # Check for errors and ensure that the executable has been compiled.
        if self.hasLaunchErrors():
            self.launcher.abort()
            return
        self.compileSolver(self.launchSimulations, (True,),
                           self.launcher.abort, ())

    def directLaunch(self):
        """Launch simulations with built-in mass action solvers or with
        a Python solver."""
        if self.hasLaunchErrors():
            self.launcher.abort()
            return
        method = self.state.methods[self.methodId]
        if simulationMethods.hasGeneric[method.timeDependence]\
               [method.category][method.method] and\
               self.state.models[self.modelId].hasOnlyMassActionKineticLaws():
            self.launchSimulations(False)
        elif simulationMethods.hasPython[method.timeDependence]\
                 [method.category][method.method]:
            self.launchPythonSimulations()
        elif simulationMethods.hasCustom[method.timeDependence]\
                 [method.category][method.method]:
            self.compileSolver(self.launchSimulations, (True,),
                               self.launcher.abort, ())
        else:
            wx.MessageBox(\
                'There is no suitable solver for this model and method.',
                'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            self.launcher.abort()

    def launchPythonSimulations(self):
        method = self.state.methods[self.methodId]
        self.numberOfTrajectories = self.launcher.trajectories.GetValue()
        self.launcher.gauge.SetRange(self.numberOfTrajectories)
        self.launcher.gauge.SetValue(0)
        self.trajectoryCount = 0
        self.startTime = time.time()
        recordedSpecies, recordedReactions = self.record.get()
        error = self.reportRecordedErrors(method, recordedSpecies,
                                          recordedReactions)
        if error:
            wx.MessageBox(error, 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            self.launcher.isRunning = False
            self.launcher.update()
            return
        error = self.state.launchPythonSimulation\
                (self, self.modelId, self.methodId, recordedSpecies,
                 recordedReactions, self.numberOfTrajectories)
        if error:
            wx.MessageBox(error, 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            self.launcher.isRunning = False
            self.launcher.update()
            return

    def incrementProgressGauge(self, numberOfTrajectories):
        self.lock.acquire()
        try:
            self.trajectoryCount += numberOfTrajectories
            self.launcher.gauge.SetValue(self.trajectoryCount)
            # If we have generated all of the trajectories.
            if self.trajectoryCount == self.numberOfTrajectories:
                wx.CallAfter(self.finishSimulations)
        finally:
            self.lock.release()

    def saveExecutable(self):
        """Save an executable. Compile it if necessary."""
        if not self.cacheModelAndMethod():
            return
        if not self.evaluateModel():
            return
        choice = wx.GetSingleChoiceIndex('Choose the kind of solver.',
                                         'Choose Solver',
                                         ['Custom solver for this model.',
                                          'Generic solver.'])
        if choice == 0:
            self.compileSolver(self.saveExecutableCustom, (),
                               lambda : None, ())
        elif choice == 1:
            self.saveExecutableMassAction()

    def saveExecutableCustom(self):
        method = self.state.methods[self.methodId]
        if sys.platform in ('win32', 'win64'):
            wildcard = "Executable files (*.exe)|*.exe|" + \
                       "All files (*.*)|*.*"
        else:
            wildcard = ""
        dialog = wx.FileDialog(self, "Save as...", os.getcwd(),
                               style=wx.SAVE|wx.OVERWRITE_PROMPT,
                               wildcard=wildcard)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            filename = dialog.GetPath()
            self.state.saveCustomExecutable(self.modelId, self.methodId,
                                            filename)
        dialog.Destroy()

    def saveExecutableMassAction(self):
        method = self.state.methods[self.methodId]
        if not simulationMethods.hasGeneric[method.timeDependence]\
                [method.category][method.method]:
            wx.MessageBox(\
                'This method does not have a generic mass action solver.',
                'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        if sys.platform in ('win32', 'win64'):
            wildcard = "Executable files (*.exe)|*.exe|" + \
                       "All files (*.*)|*.*"
        else:
            wildcard = ""
        dialog = wx.FileDialog(self, "Save as...", os.getcwd(),
                               style=wx.SAVE|wx.OVERWRITE_PROMPT,
                               wildcard=wildcard)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            filename = dialog.GetPath()
            self.state.saveGenericExecutable(self.methodId, filename)
        dialog.Destroy()

    def exportMathematica(self):
        """Export the selected model and simulation parameters to a 
        Mathematica notebook."""
        if not self.cacheModelAndMethod() or not self.evaluateModel():
            return

        method = self.state.methods[self.methodId]
        recordedSpecies, recordedReactions = self.record.get()
        error = self.reportRecordedErrors(method, recordedSpecies,
                                          recordedReactions)
        if error:
            wx.MessageBox(error, 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return

        wildcard = "Mathematica Notebooks (*.nb)|*.nb|" + \
            "All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Save as...", os.getcwd(),
                               style=wx.SAVE|wx.OVERWRITE_PROMPT,
                               wildcard=wildcard)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            fileName = dialog.GetPath()
            if os.path.splitext(fileName)[1] != '.nb':
                fileName += '.nb'
            outputFile = openWrite(fileName)
            if outputFile:
                self.state.exportMathematica(self.modelId, self.methodId,
                                             recordedSpecies, recordedReactions,
                                             outputFile)
                # A placeholder with zero trajectories may have been created.
                self.updateSimulations()
        dialog.Destroy()

    def compileSolver(self, successFunction, successArguments,
                      failureFunction, failureArguments):
        """Call self.cacheModelAndMethod() to define self.modelId, 
        self.methodId before calling this function."""
        method = self.state.methods[self.methodId]
        if not simulationMethods.hasCustom[method.timeDependence]\
                [method.category][method.method]:
            wx.MessageBox(\
                'This method cannot be compiled into a custom solver.',
                'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            failureFunction(*failureArguments)
            return
        # Check that the solver has not already been compiled.
        if self.state.hasCustomSolver(self.modelId, self.methodId):
            successFunction(*successArguments)
            return
        # Check the recorded species and reactions.
        recordedSpecies, recordedReactions = self.record.get()
        error = self.reportRecordedErrors(method, recordedSpecies,
                                          recordedReactions)
        if error:
            wx.MessageBox(error, 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            failureFunction(*failureArguments)
            return
        # Start compiling.
        self.compilingMessage = CompilingMessage()
        self.compilingMessage.Show()
        thread = CompileSolverThread(self, (self.modelId, self.methodId,
                                            recordedSpecies, recordedReactions),
                                     successFunction, successArguments,
                                     failureFunction, failureArguments)
        thread.start()

    def showCompilationErrors(self, errors):
        CompilationError(errors).Show()
        self.launcher.update()

    def destroyCompilingMessage(self):
        self.compilingMessage.Destroy()

    # CONTINUE: When the job is stopped or killed, it reports more simulations
    # than are actually completed.
    def endSimulationsMessage(self, title, elapsedTime):
        self.updateSimulations()
        message = 'Model: ' + self.modelId + ', Method: ' + self.methodId
        message += '\nGenerated %d trajectories in %f seconds.' % \
            (self.state.successfulTrajectories, elapsedTime)
        if self.state.errorMessages:
            message += '\n%d simulations failed.' %\
                len(self.state.errorMessages)
            for error in self.state.errorMessages:
                message += '\n%s' % error
        ScrolledMessageFrame(message, title, (600, 300)).Show()

    def stopSimulation(self):
        elapsedTime = time.time() - self.startTime
        self.state.stopSimulation()
        self.endSimulationsMessage('Stopped', elapsedTime)

    def killSimulation(self):
        # CONTINUE
        if sys.platform in ('win32', 'win64'):
            wx.MessageBox('Killing a simulation is not supported on MS Windows',
                          'Not Supported', style=wx.OK)
            return
        elapsedTime = time.time() - self.startTime
        self.state.killSimulation()
        self.endSimulationsMessage('Killed', elapsedTime)

    def finishSimulations(self):
        elapsedTime = time.time() - self.startTime
        self.state.tearDownSimulation()
        self.launcher.isRunning = False
        self.launcher.update()
        self.endSimulationsMessage('Finished', elapsedTime)

    def exportJobs(self):
        if not self.cacheModelAndMethod() or not self.evaluateModel():
            return

        method = self.state.methods[self.methodId]
        if simulationMethods.isStochastic(method.timeDependence,
                                          method.category):
            numberOfTrajectories = self.launcher.trajectories.GetValue()
            numberOfProcesses = self.launcher.cores.GetValue()
        else:
            numberOfTrajectories = 1
            numberOfProcesses = 1

        recordedSpecies, recordedReactions = self.record.get()
        error = self.reportRecordedErrors(method, recordedSpecies,
                                          recordedReactions)
        if error:
            wx.MessageBox(error, 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return

        if numberOfProcesses == 1:
            wildcard = "Text files (*.txt)|*.txt|" + \
                "All files (*.*)|*.*"
            dialog = wx.FileDialog(self, "Export job as...", os.getcwd(),
                                   style=wx.SAVE|wx.OVERWRITE_PROMPT,
                                   wildcard=wildcard)
        else:
            wildcard = "All files (*.*)|*.*"
            dialog = wx.FileDialog(self, "Select base name...",
                                   os.getcwd(), style=wx.SAVE,
                                   wildcard=wildcard)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            self.exportSuiteOfJobs(dialog.GetPath(), self.modelId,
                                   self.methodId, recordedSpecies,
                                   recordedReactions, numberOfTrajectories,
                                   numberOfProcesses)
            # A placeholder with zero trajectories may have been created.
            self.updateSimulations()
        dialog.Destroy()

    def exportSuiteOfJobs(self, fileName, modelId, methodId, recordedSpecies,
                          recordedReactions, numberOfTrajectories,
                          numberOfProcesses):
        # If necessary, start a new output container for this model and method.
        self.state.ensureOutput(modelId, methodId, recordedSpecies,
                                recordedReactions)
        if numberOfProcesses == 1:
            if not os.path.splitext(fileName)[1]:
                fileName += '.txt'
            outputFile = openWrite(fileName)
            if not outputFile:
                return
            self.state.exportJob(outputFile, modelId, methodId,
                                 recordedSpecies, recordedReactions,
                                 numberOfTrajectories)
        else:
            width = int(math.log10(numberOfProcesses - 0.1)) + 1
            format = '_%0' + str(width) + 'd.txt'
            for index in range(numberOfProcesses):
                n = computePartition(numberOfTrajectories, numberOfProcesses,
                                     index)
                if n != 0:
                    outputFile = openWrite(fileName + format % index)
                    if not outputFile:
                        return
                    self.state.exportJob(outputFile, modelId, methodId,
                                         recordedSpecies, recordedReactions, n,
                                         index)

    def importSolution(self):
        if not self.cacheModelAndMethod():
            return
        method = self.state.methods[self.methodId]
        if simulationMethods.usesStatistics(method.timeDependence,
                                            method.category):
            self.importStatistics()
        else:
            self.importTrajectories()
            
    def importStatistics(self):
        # The solution must not have been imported before.
        if (self.modelId, self.methodId) in self.state.output:
            wx.MessageBox('This solution for this model and method has already been imported.', 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            self.launcher.abort()
            return

        # Get the recorded species.
        recordedSpecies, recordedReactions = self.record.get()
        method = self.state.methods[self.methodId]
        error = self.reportRecordedErrors(method, recordedSpecies,
                                          recordedReactions)
        if error:
            wx.MessageBox(error, 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            self.launcher.abort()
            return

        wildcard = "Text files (*.txt)|*.txt|" + \
            "All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Import solution statistics...",
                               os.getcwd(), style=wx.FD_OPEN, 
                               wildcard=wildcard)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            filename = dialog.GetPath()
            try:
                self.state.importStatistics(filename, self.modelId, 
                                            self.methodId, recordedSpecies)
                self.updateSimulations()
            except Exception, exception:
                truncatedErrorBox("Problem in importing %s.\n" % filename +
                                  str(exception))
                self.deleteOutput(self.modelId, self.methodId)
        dialog.Destroy()

    def importTrajectories(self):
        # They must have exported this job.
        if not (self.modelId, self.methodId) in self.state.output:
            wx.MessageBox('This job was not exported. There is no placeholder for the output.', 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            self.launcher.abort()
            return
        # If the method is deterministic, check that they have not already
        # generated a trajectory.
        method = self.state.methods[self.methodId]
        if not simulationMethods.isStochastic(method.timeDependence,
                                              method.category) and\
                self.state.output[(self.modelId, self.methodId)].populations:
            wx.MessageBox('This is a deterministic method.\nThe trajectory has already been generated.', 'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            self.launcher.abort()
            return

        wildcard = "Text files (*.txt)|*.txt|" + \
            "All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Import trajectories...", os.getcwd(),
                               style=wx.FD_OPEN|wx.FD_MULTIPLE, 
                               wildcard=wildcard)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            try:
                self.state.importSuiteOfTrajectories(dialog.GetPaths(),
                                                     self.modelId,
                                                     self.methodId)
            except Exception, exception:
                truncatedErrorBox("Problem in importing trajectories.\n" +
                                  str(exception))
            self.updateSimulations()
        dialog.Destroy()


    def updateSimulations(self):
        self.launcher.gauge.SetValue(0)
        self.updateTrajectoriesCount(self.modelId, self.methodId)
        self.setModelPermissions(self.modelId)
        self.setMethodPermissions(self.methodId)
        self.updateRecorded(self.modelId, self.methodId)
        self.launcher.update()

    # File I/O.

    def readFile(self, filename='', useMessageBox=True):
        filename = os.path.abspath(filename)
        # Check that the file exists.
        if not os.path.isfile(filename) and useMessageBox:
            wx.MessageBox("The file %s does not exist." % filename,
                          "Error!", style=wx.OK|wx.ICON_EXCLAMATION)
            return
        # Determine if this is a Cain or an SBML file.
        try:
            if re.search('<sbml', open(filename, 'r').read()):
                self.importSbml(filename, useMessageBox)
                return
        except Exception, exception:
            if useMessageBox:
                truncatedErrorBox('Problem in determining the file type for ' +
                                  filename + '.\n' + str(exception))
            return
        # Read the Cain file.
        try:
            self.clear()
            errors = ''
            errors = self.state.read(filename)
            if errors and useMessageBox:
                truncatedErrorBox(errors)
            for id in self.state.models:
                model = self.state.models[id]
                self.modelTables[id] = [model.writeSpeciesTable(),
                                        model.writeReactionsTable(),
                                        model.writeTimeEventsTable(),
                                        model.writeTriggerEventsTable(),
                                        model.writeParametersTable(),
                                        model.writeCompartmentsTable()]
                self.modelsList.insertItem(id)
            for id in self.state.methods:
                self.methodsList.insertItem(id)
            self.clearModel()
            self.modelsList.select()
            self.clearMethod()
            self.methodsList.select()
            self.updateTrajectories()
            self.filename = filename
            self.SetTitle(self.title + ' -- ' + self.filename)
            self.isModified = False
        except Exception, exception:
            if useMessageBox:
                truncatedErrorBox("Problem in reading %s.\n" % filename +
                                  errors + '\n' + str(exception))

    def importSbml(self, filename, useMessageBox=True):
        if not os.path.isfile(filename) and useMessageBox:
            wx.MessageBox("The file %s does not exist." % filename,
                          "Error!", style=wx.OK|wx.ICON_EXCLAMATION)
            return
        try:
            errors = ''
            (id, errors) = self.state.importSbmlModel(filename)
            if errors and useMessageBox:
                truncatedErrorBox(errors)
            if id:
                self.modelTables[id] =\
                    [self.state.models[id].writeSpeciesTable(),
                     self.state.models[id].writeReactionsTable(),
                     self.state.models[id].writeTimeEventsTable(),
                     self.state.models[id].writeTriggerEventsTable(),
                     self.state.models[id].writeParametersTable(),
                     self.state.models[id].writeCompartmentsTable()]
                self.modelsList.insertItem(id)
                # Select the model.
                self.modelsList.selectLast()
            self.isModified = True
        except Exception, exception:
            if useMessageBox:
                truncatedErrorBox("Problem in importing %s.\n" % filename +
                                  errors + '\n' + str(exception))

    def importTextModel(self, filename, useMessageBox=True):
        if not os.path.isfile(filename) and useMessageBox:
            wx.MessageBox("The file %s does not exist." % filename,
                          "Error!", style=wx.OK|wx.ICON_EXCLAMATION)
            return
        try:
            id = self.state.importTextModel(filename)
            assert id
            self.modelTables[id] =\
                [self.state.models[id].writeSpeciesTable(),
                 self.state.models[id].writeReactionsTable(),
                 self.state.models[id].writeTimeEventsTable(),
                 self.state.models[id].writeTriggerEventsTable(),
                 self.state.models[id].writeParametersTable(),
                 self.state.models[id].writeCompartmentsTable()]
            self.modelsList.insertItem(id)
            # Select the model.
            self.modelsList.selectLast()
            self.isModified = True
        except Exception, exception:
            if useMessageBox:
                truncatedErrorBox("Problem in importing text model %s.\n"
                                  % filename + '\n' + str(exception))

    def exportSbml(self, id, filename):
        self.modelId = id
        if not self.evaluateModel():
            return
        outputFile = openWrite(filename)
        if outputFile:
            version = int(self.state.preferences.data['SBML']['Version'])
            self.state.writeSbml(id, outputFile, version)

    def exportCmdl(self, id, filename):
        self.modelId = id
        if not self.evaluateModel():
            return
        outputFile = openWrite(filename)
        if not outputFile:
            return
        self.state.models[id].writeCmdl(outputFile)

    # Model editor callbacks.

    def modelInsert(self):
        id = self.state.insertNewModel()
        model = self.state.models[id]
        self.modelTables[id] = [model.writeSpeciesTable(),
                                model.writeReactionsTable(),
                                model.writeTimeEventsTable(),
                                model.writeTriggerEventsTable(),
                                model.writeParametersTable(),
                                model.writeCompartmentsTable()]
        return id

    def modelClone(self, id):
        newId = self.state.insertCloneModel(id)
        model = self.state.models[newId]
        self.modelTables[newId] = [model.writeSpeciesTable(),
                                   model.writeReactionsTable(),
                                   model.writeTimeEventsTable(),
                                   model.writeTriggerEventsTable(),
                                   model.writeParametersTable(),
                                   model.writeCompartmentsTable()]
        return newId

    def modelDuplicate(self, id):
        dialog = DuplicateDialog(self)
        result = dialog.ShowModal()
        if result != wx.ID_OK:
            return None
        multiplicity = dialog.getMultiplicity()
        useScaling = dialog.useScaling()
        dialog.Destroy()
        newId = self.state.insertDuplicatedModel(id, multiplicity, useScaling)
        model = self.state.models[newId]
        self.modelTables[newId] = [model.writeSpeciesTable(),
                                   model.writeReactionsTable(),
                                   model.writeTimeEventsTable(),
                                   model.writeTriggerEventsTable(),
                                   model.writeParametersTable(),
                                   model.writeCompartmentsTable()]
        return newId

    def modelEdit(self, old, new):
        if new in self.state.models:
            wx.MessageBox("Cannot change identifier %s to %s." % (old, new),
                          "Error!", style=wx.OK|wx.ICON_EXCLAMATION)
            return False
        # Rename the model identifier in:
        # The state.
        self.state.changeModelId(old, new)
        # The model list.
        self.modelTables[new] = self.modelTables[old]
        del self.modelTables[old]
        # The trajectories list.
        self.trajectoriesList.changeModelId(old, new)
        return True

    def modelDelete(self, id):
        assert id in self.state.models
        self.clearModel()
        del self.state.models[id]
        del self.modelTables[id]

    def onModelSelected(self, event):
        id = self.modelsList.getText(event.GetIndex())
        if id:
            self.updateModel(id)
        self.launcher.update()

    def onModelDeselected(self, event):
        id = self.modelsList.getText(event.GetIndex())
        if id:
            self.modelTables[id] = self.modelEditor.getTableData()
        self.clearModel()
        self.launcher.update()

    def getSelectedModelId(self):
        return self.modelsList.getSelectedText()

    def getSelectedMethodInfo(self):
        """Return the method information as a tuple of method, hasGeneric, 
        hasCustom, and hasPython. This is the information that the launcher
        needs to enable the appropriate buttons. Return a tuple of None's if
        no method is selected."""
        methodId = self.methodsList.getSelectedText()
        if not methodId:
            return (None, None, None, None)
        m = self.state.methods[methodId]
        i, j, k = m.timeDependence, m.category, m.method
        return (simulationMethods.methods[i][j][k],
                simulationMethods.hasGeneric[i][j][k],
                simulationMethods.hasCustom[i][j][k],
                simulationMethods.hasPython[i][j][k])

    # Simulation parameters editor callbacks.

    def methodInsert(self):
        """Insert new simulation parameters. Return the new identifier."""
        self.syncSelectedMethod()
        return self.state.insertNewMethod()

    def methodClone(self, id):
        """Insert a clone of the specified simulation parameters. Return the
        new identifier. If the simulation parameters are not valid, return 
        None."""
        if not self.syncMethod(id, 'Error'):
            return None
        return self.state.insertCloneMethod(id)

    def methodEdit(self, old, new):
        if new in self.state.methods:
            wx.MessageBox("Cannot change identifier %s to %s." % (old, new),
                          "Error!", style=wx.OK|wx.ICON_EXCLAMATION)
            return False
        # Rename the identifier in:
        # The state.
        self.state.changeMethodId(old, new)
        # The trajectories list.
        self.trajectoriesList.changeMethodId(old, new)
        return True

    def methodDelete(self, id):
        sp = self.state.methods
        assert id in sp
        self.clearMethod()
        del sp[id]

    def onMethodSelected(self, event):
        id = self.methodsList.getText(event.GetIndex())
        assert id
        self.updateMethod(id)
        self.launcher.update()

    def onMethodDeselected(self, event):
        id = self.methodsList.getText(event.GetIndex())
        # Note that a method might be deselected right after being deleted.
        # Thus we need to check if the method still exists.
        if id in self.state.methods:
            self.syncMethod(id, 'Error')
            self.parseMethods(id)
        self.clearMethod()
        self.launcher.update()

    # Trajectories display callbacks.

    def updatePermissions(self):
        # Check the models and methods.
        modelId = self.modelsList.getSelectedText()
        if modelId:
            self.setModelPermissions(modelId)
        methodId = self.methodsList.getSelectedText()
        if methodId:
            self.setMethodPermissions(methodId)
        self.updateRecorded(modelId, methodId)

    def deleteOutput(self, modelId, methodId):
        self.state.deleteOutput(modelId, methodId)
        self.updatePermissions()

    def deleteAllOutput(self):
        self.state.deleteAllOutput()
        self.updatePermissions()

    # CONTINUE: Remove. This will be implemented in ExportTimeSeries.
    def exportCsv(self, modelId, methodId):
        wildcardCsv = "CSV files (*.csv)|*.csv|" + \
            "All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Save as...", os.getcwd(),
                               style=wx.SAVE|wx.OVERWRITE_PROMPT,
                               wildcard=wildcardCsv)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            filename = dialog.GetPath()
            if not os.path.splitext(filename)[1]:
                filename = filename + '.csv'
            outputFile = openWrite(filename)
            if outputFile:
                self.state.exportCsv(modelId, methodId, outputFile)
        dialog.Destroy()

    def exportGnuplot(self, modelId, methodId):
        # CONTINUE
        # Exporting statistics is not currently supported.
        method = self.state.methods[methodId]
        category = simulationMethods.categories[method.timeDependence]\
            [method.category]
        if category in ('Statistics, Transient Behavior',
                        'Statistics, Steady State'):
            wx.MessageBox('Exporting ' + category + ' is not yet supported.',
                          'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        
        wildcardGnuplot = "Gnuplot data files (*.dat)|*.dat|" + \
            "All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Save as...", os.getcwd(),
                               style=wx.SAVE|wx.OVERWRITE_PROMPT,
                               wildcard=wildcardGnuplot)
        if dialog.ShowModal() == wx.ID_OK:
            os.chdir(dialog.GetDirectory())
            filename = dialog.GetPath()
            baseName = os.path.splitext(os.path.split(filename)[1])[0]
            if not os.path.splitext(filename)[1]:
                filename = filename + '.dat'
            self.state.exportGnuplot(modelId, methodId, baseName, filename)
        dialog.Destroy()

    # Other.

    def saveFile(self):
        """Return true if the file was saved."""
        outputFile = openWrite(self.filename)
        if not outputFile:
            return False
        self.state.write(outputFile)
        self.isModified = False
        return True

    def clearModelList(self):
        self.modelsList.clear()
        self.modelTables = {}
        self.clearModel()

    def clearModel(self):
        self.modelEditor.clear()
        self.clearRecorded()

    def updateModel(self, id):
        self.modelEditor.setTableData(self.modelTables[id])
        self.setModelPermissions(id)
        self.updateRecorded(modelId=id)

    def setModelPermissions(self, id):
        if self.state.doesModelHaveDependentOutput(id):
            self.modelsList.disableDelete()
            self.modelEditor.disable()
        else:
            self.modelsList.enableDelete()
            self.modelEditor.enable()

    # CONTINUE: I should move the implementation to MethodEditor.
    def updateMethod(self, id):
        m = self.state.methods[id]
        editor = self.methodEditor
        editor.setMethod(m.timeDependence, m.category, m.method, m.options)
        editor.startTime.SetValue(str(m.startTime))
        editor.equilibrationTime.SetValue(str(m.equilibrationTime))
        editor.recordingTime.SetValue(str(m.recordingTime))
        if m.maximumSteps is not None:
            editor.maximumSteps.SetValue(str(m.maximumSteps))
        else:
            editor.maximumSteps.SetValue('')
        editor.frames.SetValue(m.numberOfFrames)
        editor.bins.SetValue(m.numberOfBins)
        editor.multiplicity.SetValue(m.multiplicity)
        if m.solverParameter is not None:
            editor.solverParameter.SetValue(str(m.solverParameter))
        else:
            editor.solverParameter.SetValue('')
        self.setMethodPermissions(id)
        self.updateRecorded(methodId=id)

    def clearMethod(self):
        editor = self.methodEditor
        editor.setMethod(0, 0, 0, 0)
        editor.startTime.SetValue('0')
        editor.equilibrationTime.SetValue('0')
        editor.recordingTime.SetValue('1')
        editor.maximumSteps.SetValue('')
        editor.frames.SetValue(11)
        editor.bins.SetValue(32)
        editor.multiplicity.SetValue(4)
        editor.solverParameter.SetValue('')
        self.methodEditor.Disable()
        self.clearRecorded()

    def clearMethodList(self):
        self.methodsList.clear()
        self.clearMethod()

    def setMethodPermissions(self, id):
        if self.state.doesMethodHaveDependentOutput(id):
            self.methodsList.disableDelete()
            # Move the focus to the launcher. Without this, if the cursor is in
            # a text control and the widget is disabled then the user can
            # still edit the text.
            self.launcher.SetFocus()
            self.methodEditor.Disable()
        else:
            self.methodsList.enableDelete()
            self.methodEditor.Enable()

    def clearRecorded(self):
        """Empty lists of species and reactions to record."""
        self.record.set([], [])

    def onSpeciesOrReactionsModified(self, event):
        self.updateRecorded()

    def updateRecorded(self, modelId=None, methodId=None):
        # Get the selected model and method if they were not passed as
        # parameters.
        if not modelId:
            modelId = self.modelsList.getSelectedText()
        if not methodId:
            methodId = self.methodsList.getSelectedText()
        # If there is not a selected model and method, clear the recorded
        # items and return.
        if not (modelId and methodId):
            self.clearRecorded()
            return
        speciesIdentifiers, reactionIdentifiers =\
            self.modelEditor.getSpeciesAndReactionIdentifiers()
        # If there is output for the model and method, display the recorded
        # items and disable input.
        if (modelId, methodId) in self.state.output:
            output = self.state.output[(modelId, methodId)]
            # Display all species and reactions.
            self.record.set(speciesIdentifiers, reactionIdentifiers)
            # Check the recorded species and reactions
            self.record.checkList(output.recordedSpecies,
                                  output.recordedReactions)
            # Disable input.
            self.record.disable()
            return
        # Otherwise display the items that can be recorded.
        timeDependenceIndex = self.methodEditor.timeDependence.GetSelection()
        categoryIndex = self.methodEditor.category.GetSelection()
        category = simulationMethods.categories[timeDependenceIndex]\
            [categoryIndex]
        if category in ('Time Series, Uniform', 'Time Series, Deterministic'):
            # Display the species and reactions.
            self.record.set(speciesIdentifiers, reactionIdentifiers)
            # Check each of the species, but not the reactions.
            self.record.checkSpecies(True)
            # Enable selection.
            self.record.enable()
        elif category == 'Time Series, All Reactions':
            # Each reaction event is recorded, so every species and reaction
            # are recorded.
            self.record.set(speciesIdentifiers, reactionIdentifiers)
            self.record.checkAll(True)
            self.record.disable()
        elif category in ('Histograms, Transient Behavior',
                          'Histograms, Steady State',
                          'Statistics, Transient Behavior',
                          'Statistics, Steady State'):
            # Only species may be recorded.
            self.record.set(speciesIdentifiers, [])
            # Check each of the species.
            self.record.checkSpecies(True)
            self.record.enable()
        else:
            assert False

    def updateTrajectories(self):
        self.trajectoriesList.clear()
        for key in self.state.output:
            self.trajectoriesList.insertItem(
                key[0], key[1], str(self.state.output[key].size()))

    def updateTrajectoriesCount(self, modelId, methodId):
        """Update the trajectories count."""
        key = (modelId, methodId)
        assert key in self.state.output
        self.trajectoriesList.update(modelId, methodId, 
                                     str(self.state.output[key].size()))

    def syncSelectedModel(self):
        id = self.modelsList.getSelectedText()
        # Do nothing if no model is selected.
        if not id:
            return
        # CONTINUE: Do nothing if the model has not been modified.
        self.syncModel(id)

    def syncModel(self, id):
        self.modelTables[id] = self.modelEditor.getTableData()

    def syncSelectedMethod(self, errorMessage = 'Error!'):
        id = self.methodsList.getSelectedText()
        # Do nothing if no parameters are selected.
        if not id:
            return True
        # CONTINUE: Do nothing if the parameters have not been modified.
        return self.syncMethod(id, errorMessage)

    # CONTINUE Perhaps move the implementation.
    def syncMethod(self, id, errorMessage):
        editor = self.methodEditor
        errors = ''
        # The time interval.
        try:
            startTime = float(editor.startTime.GetValue())
        except:
            errors += 'The start time must be a floating point value.\n'
        try:
            equilibrationTime = float(editor.equilibrationTime.GetValue())
        except:
            errors += 'The equilibration time must be a floating point value.\n'
        try:
            recordingTime = float(editor.recordingTime.GetValue())
        except:
            errors += 'The recording time must be a floating point value.\n'
        if editor.maximumSteps.GetValue():
            try:
                maximumSteps = float(editor.maximumSteps.GetValue())
            except:
                errors += 'The maximum steps must be either blank or a floating point value.\n'
        else:
            maximumSteps = None
        # The solver parameter.
        parameterValue = None
        if editor.solverParameter.GetValue():
            try:
                parameterValue = float(editor.solverParameter.GetValue())
            except:
                errors = 'The solver parameter must be a floating point value.\n'
        if not errors:
            errors = self.state.editMethod\
                (id,
                 editor.timeDependence.GetSelection(),
                 editor.category.GetSelection(),
                 editor.method.GetSelection(),
                 editor.options.GetSelection(),
                 startTime,
                 equilibrationTime,
                 recordingTime,
                 maximumSteps,
                 editor.frames.GetValue(),
                 editor.bins.GetValue(),
                 editor.multiplicity.GetValue(),
                 parameterValue)
        if errors:
            truncatedErrorBox('Invalid simulation parameters.\n' + errors)
            return False
        return True

    def syncSelectedModelAndMethod(self, errorMessage = 'Error!'):
        self.syncSelectedModel()
        return self.syncSelectedMethod(errorMessage)

    # CONTINUE: Get rid of errorMessage parameter.
    def parseModel(self, id, methodId, errorMessage = 'Error!'):
        # The identifiers that are accumulated as each component is parsed.
        identifiers = []
        # Update the model table.
        if not self.parseParametersTable(id, identifiers):
            return False
        if not self.parseCompartmentsTable(id, identifiers):
            return False
        if not self.parseSpeciesTable(id, identifiers):
            return False
        if not self.parseReactionsTable(id, identifiers):
            return False
        if not self.parseTimeEventsTable(id, identifiers):
            return False
        if not self.parseTriggerEventsTable(id, identifiers):
            return False
        error = self.state.hasErrorsInModel(id, methodId)
        if error:
            truncatedErrorBox('Model ' + id + ' is invalid.\n' + error)
            return False
        return True

    # CONTINUE: Get rid of errorMessage parameter.
    def parseMethods(self, id, errorMessage = 'Error!'):
        error = self.state.hasErrorsInMethod(id)
        if error:
            truncatedErrorBox('Invalid simulation parameters.\n' + error)
            return False
        return True

    def parseModelsAndMethods(self, errorMessage = 'Error!'):
        # Parse the models.
        for id in self.state.models:
            if not self.parseModel(id, None):
                return False
        # Parse the currently edited simulation parameters.
        id = self.methodsList.getSelectedText()
        # If a set of simulation parameters are selected.
        if id:
            return self.parseMethods(id, errorMessage)
        return True

    def parseSpeciesTable(self, id, identifiers):
        model = self.state.models[id]
        parser = SpeciesTextParser()
        model.speciesIdentifiers, model.species = \
            parser.parseTable(self.modelTables[id][0], identifiers)
        if parser.errorMessage:
            truncatedErrorBox('In model ' + id + ': ' + parser.errorMessage)
            return False
        return True

    def parseReactionsTable(self, id, identifiers):
        model = self.state.models[id]
        parser = ReactionTextParser()
        model.reactions = parser.parseTable(self.modelTables[id][1],
                                            model.speciesIdentifiers,
                                            identifiers)
        if parser.errorMessage:
            truncatedErrorBox('In model ' + id + ': ' + parser.errorMessage)
            return False
        return True

    def parseTimeEventsTable(self, id, identifiers):
        model = self.state.models[id]
        parser = TimeEventTextParser()
        model.timeEvents = parser.parseTable(self.modelTables[id][2],
                                             identifiers)
        if parser.errorMessage:
            truncatedErrorBox('In model ' + id + ': ' + parser.errorMessage)
            return False
        return True

    def parseTriggerEventsTable(self, id, identifiers):
        model = self.state.models[id]
        parser = TriggerEventTextParser()
        model.triggerEvents = parser.parseTable(self.modelTables[id][3],
                                                identifiers)
        if parser.errorMessage:
            truncatedErrorBox('In model ' + id + ': ' + parser.errorMessage)
            return False
        return True

    def parseParametersTable(self, id, identifiers):
        model = self.state.models[id]
        parser = ParameterTextParser()
        model.parameters = parser.parseTable(self.modelTables[id][4],
                                             identifiers)
        if parser.errorMessage:
            truncatedErrorBox('In model ' + id + ': ' + parser.errorMessage)
            return False
        return True

    def parseCompartmentsTable(self, id, identifiers):
        model = self.state.models[id]
        parser = CompartmentTextParser()
        model.compartments = parser.parseTable(self.modelTables[id][5],
                                               identifiers)
        if parser.errorMessage:
            truncatedErrorBox('In model ' + id + ': ' + parser.errorMessage)
            return False
        return True

def main():
    app = wx.PySimpleApp()
    #frame = MainFrame()
    #frame.Show(True)
    message = '\n'.join(['Long error message-------------------------------------------------------------------------number ' + str(n) + '.' for n in range(1,61)])
    messageFrame = ScrolledMessageFrame(message, 'Errors', (600, 300))
    messageFrame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
