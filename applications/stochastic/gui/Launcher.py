"""Implements the simulation launcher."""

# If we are running the unit tests.
if __name__ == '__main__':
    resourcePath = '../'
else:
    from resourcePath import resourcePath

import wx
import os
import os.path
import sys

def cpuCount():
    """Return the number of available cores."""
    try:
        # The multiprocessing module was introduced in Python 2.6. It has been
        # backported to 2.5 and 2.4 and is included in some distributions of
        # these.
        import multiprocessing
        return multiprocessing.cpu_count()
    except:
        # I found the following recipe at
        # http://codeliberates.blogspot.com/2008/05/detecting-cpuscores-in-python.html
        # Linux, Unix and MacOS:
        if hasattr(os, "sysconf"):
            if os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"):
                # Linux & Unix:
                count = os.sysconf("SC_NPROCESSORS_ONLN")
                if isinstance(count, int) and count > 0:
                    return count
            else: # OS X:
                return int(os.popen2("sysctl -n hw.ncpu")[1].read())
        # Windows:
        if os.environ.has_key("NUMBER_OF_PROCESSORS"):
            count = int(os.environ["NUMBER_OF_PROCESSORS"]);
            if count > 0:
                return count
        # Default.
        return 1

class LauncherButtons(wx.Panel):
    def __init__(self, parent=None):
        wx.Panel.__init__(self, parent)
        self.exportMethod = ''

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/launch.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.launch = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/launchDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.launch.SetBitmapDisabled(bmp)
        self.launch.SetToolTip(wx.ToolTip('Launch the simulations with the mass action solvers.'))

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/compile.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.compile = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/compileDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.compile.SetBitmapDisabled(bmp)
        self.compile.SetToolTip(wx.ToolTip('Launch the simulations. Compile if necessary.'))

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/stop.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.stop = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/stopDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.stop.SetBitmapDisabled(bmp)
        self.stop.SetToolTip(wx.ToolTip('Stop the simulation'))

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/cancel.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.kill = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/cancelDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.kill.SetBitmapDisabled(bmp)
        self.kill.SetToolTip(wx.ToolTip('Kill the simulation'))

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/utilities-terminal.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.saveExecutable = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                           'gui/icons/16x16/utilities-terminalDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.saveExecutable.SetBitmapDisabled(bmp)
        self.saveExecutable.SetToolTip(wx.ToolTip('Save the command line executable. Compile if necessary.'))

        self.exportJobsBitmap =\
            wx.Image(os.path.join(resourcePath,
                                  'gui/icons/16x16/filesave.png'),
                     wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.exportJobsToolTip = 'Export jobs for batch processing.'
        self.exportMathematicaBitmap =\
            wx.Image(os.path.join(resourcePath,
                                  'gui/icons/16x16/Mathematica.png'),
                     wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.exportMathematicaToolTip =\
            'Export model to a Mathematica notebook.'
        self.export = wx.BitmapButton(self, -1, self.exportJobsBitmap)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                           'gui/icons/16x16/filesaveDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.export.SetBitmapDisabled(bmp)
        self.export.SetToolTip(wx.ToolTip(self.exportJobsToolTip))

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/fileopen.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.importTrajectories = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/fileopenDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.importTrajectories.SetBitmapDisabled(bmp)
        self.importTrajectories.SetToolTip(wx.ToolTip('Import solutions.'))

        sizer = wx.GridSizer(rows=2, cols=4)
        sizer.Add(self.launch)
        sizer.Add(self.compile)
        sizer.Add(self.stop)
        sizer.Add(self.kill)
        sizer.Add(self.saveExecutable)
        sizer.Add(self.export)
        sizer.Add(self.importTrajectories)
        self.SetSizer(sizer)

    def enableLaunch(self, method, hasMassAction, hasCustom, hasPython):
        self.exportMethod = method
        # If this is a statistics group with no defined method (in which case
        # the solution may only be imported) or if they have selected the
        # Mathematica solver.
        if method in ('Import Solution', 'Mathematica'):
            self.launch.Disable()
            self.compile.Disable()
        else:
            self.launch.Enable(hasMassAction or hasPython)
            self.compile.Enable(hasCustom)
        self.stop.Disable()
        self.kill.Disable()

    def disableLaunch(self):
        self.launch.Disable()
        self.compile.Disable()
        self.stop.Disable()
        self.kill.Disable()

    def runSimulation(self):
        self.launch.Disable()
        self.compile.Disable()
        self.stop.Enable()
        self.kill.Enable()

    def enableOther(self, method, hasPython):
        self.exportMethod = method
        if method == 'Import Solution':
            # Statistics group with no solvers. One may only import solutions.
            self.saveExecutable.Disable()
            self.export.SetBitmapLabel(self.exportJobsBitmap)
            self.export.SetToolTip(wx.ToolTip(self.exportJobsToolTip))
            self.export.Disable()
            self.importTrajectories.Enable()
        elif method == 'Mathematica':
            self.saveExecutable.Disable()
            self.export.SetBitmapLabel(self.exportMathematicaBitmap)
            self.export.SetToolTip(wx.ToolTip(self.exportMathematicaToolTip))
            self.export.Enable()
            self.importTrajectories.Enable()
        elif hasPython:
            self.disableOther()
        else:
            self.saveExecutable.Enable()
            self.export.SetBitmapLabel(self.exportJobsBitmap)
            self.export.SetToolTip(wx.ToolTip(self.exportJobsToolTip))
            self.export.Enable()
            self.importTrajectories.Enable()

    def disableOther(self):
        self.saveExecutable.Disable()
        self.export.Disable()
        self.importTrajectories.Disable()

    def enable(self, method, hasMassAction, hasCustom, hasPython):
        # CONTINUE
        self.enableLaunch(method, hasMassAction, hasCustom, hasPython)
        self.enableOther(method, hasPython)

    def disable(self):
        self.disableLaunch()
        self.disableOther()

class Launcher(wx.Panel):
    """The simulation launcher."""
    
    def __init__(self, parent, application, launch, compile, stop,
                 kill, saveExecutable, exportJobs, exportMathematica,
                 importTrajectories):
        """Parameters:
        parent: The parent widget."""
        wx.Panel.__init__(self, parent)

        self.application = application
        self.launch = launch
        self.compile = compile
        self.stop = stop
        self.kill = kill
        self.saveExecutable = saveExecutable
        self.exportJobs = exportJobs
        self.exportMathematica = exportMathematica
        self.importTrajectories = importTrajectories
        self.isRunning = False
        
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Title.
        title = wx.StaticText(self, -1, 'Launcher')
        title.SetToolTip(wx.ToolTip('Launch a simulation to generate trajectories. Use the quick launch button for mass action kinetics and the compile button for custom propensity functions. Select the number of trajectories to generate and the number of cores to use.'))
        sizer.Add(title, 0, wx.ALL, 0)
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND|wx.ALL, 1)

        # Buttons.
        self.buttons = LauncherButtons(self)
        self.Bind(wx.EVT_BUTTON, self.onLaunch, self.buttons.launch)
        self.Bind(wx.EVT_BUTTON, self.onCompile, self.buttons.compile)
        self.Bind(wx.EVT_BUTTON, self.onStop, self.buttons.stop)
        self.Bind(wx.EVT_BUTTON, self.onKill, self.buttons.kill)
        self.Bind(wx.EVT_BUTTON, self.onSaveExecutable,
                  self.buttons.saveExecutable)
        self.Bind(wx.EVT_BUTTON, self.onExport, self.buttons.export)
        self.Bind(wx.EVT_BUTTON, self.onImport, self.buttons.importTrajectories)
        sizer.Add(self.buttons, 0)

        # Progress gauge.
        self.gauge = wx.Gauge(self, size=(120,12))
        self.gauge.SetToolTip(wx.ToolTip('The progress bar shows what portion of the current simulation has been completed.'))
        sizer.Add(self.gauge, 0)
        
        # Parameters.
        t = wx.StaticText(self, -1, "Trajectories:")
        t.SetToolTip(wx.ToolTip('The number of trajectories to generate.'))
        sizer.Add(t, 0)
        self.trajectories = wx.SpinCtrl(self, value='1', size=(10*12, 2*12),
                                        min=1, max=1000000, initial=1)
        sizer.Add(self.trajectories, 0)

        t = wx.StaticText(self, -1, "Cores:")
        t.SetToolTip(wx.ToolTip('The number of solver processes to run concurrently. For best performance set this to the number of cores in your computer.'))
        sizer.Add(t, 0)
        count = cpuCount()
        # Set the max core count to 1024 because they may be exporting jobs
        # to run on a cluster.
        self.cores = wx.SpinCtrl(self, value=str(count), size=(10*12, 2*12),
                                 min=1, max=1024, initial=count)
        sizer.Add(self.cores, 0)

        t = wx.StaticText(self, -1, "Granularity:")
        t.SetToolTip(wx.ToolTip('The granularity determines the number of trajectories computed in a single task.'))
        sizer.Add(t, 0)
        self.granularity = wx.Slider(self, -1, value=5, minValue=0,
                                  maxValue=10, size=(120,-1))
        sizer.Add(self.granularity, 0)

        t = wx.StaticText(self, -1, "Priority:")
        t.SetToolTip(wx.ToolTip('The priority of the solver processes. Using the lowest priority is recommended.'))
        sizer.Add(t, 0)
        self.priority = wx.Slider(self, -1, value=0, minValue=0,
                                  maxValue=20, size=(120,-1))
        sizer.Add(self.priority, 0)

        self.SetSizer(sizer)
        sizer.Fit(self)

        self.update()

    def onLaunch(self, event):
        self.isRunning = True
        self.update()
        # This allows the disable to take effect.
        wx.Yield()
        self.launch()

    def onCompile(self, event):
        self.isRunning = True
        self.update()
        # This allows the disable to take effect.
        wx.Yield()
        self.compile()

    def onStop(self, event):
        self.stop()
        self.abort()

    def onKill(self, event):
        self.kill()
        self.abort()

    def onSaveExecutable(self, event):
        self.buttons.disable()
        # This allows the disable to take effect.
        wx.Yield()
        self.saveExecutable()
        self.update()

    def onExport(self, event):
        # There is no need to disable because the dialogs are modal.
        if self.buttons.exportMethod == 'Mathematica':
            self.exportMathematica()
        else:
            self.exportJobs()

    def onImport(self, event):
        self.importTrajectories()

    def abort(self):
        self.isRunning = False
        self.update()

    def update(self):
        info = self.application.getSelectedMethodInfo()
        if self.isRunning:
            self.buttons.runSimulation()
            if self.application.getSelectedModelId() and info[0]:
                self.buttons.enableOther(info[0], info[3])
            else:
                self.buttons.disableOther()
        else:
            if self.application.getSelectedModelId() and info[0]:
                self.buttons.enable(*info)
            else:
                self.buttons.disable()

    def getNiceIncrement(self):
        """Return the nice value for the specified priority."""
        return 20 - self.priority.GetValue()

    def getGranularity(self):
        """The granularity defines the number of jobs per task. Return a 
        value between 0 and 1."""
        return float(self.granularity.GetValue()) / self.granularity.GetMax()

class LauncherFrame(wx.Frame):
    def __init__(self, parent=None):
        wx.Frame.__init__(self, parent, title='Launcher',
                          size=(200,400))
        launcher = Launcher(self, self, self.launch, self.compile,
                            self.stop, self.kill, 
                            self.saveExecutable,
                            self.exportJobs, self.exportMathematica,
                            self.importTrajectories)

    def launch(self):
        print 'launch'

    def compile(self):
        print 'compile'

    def stop(self):
        print 'stop'

    def kill(self):
        print 'kill'

    def saveExecutable(self):
        print 'saveExecutable'

    def exportJobs(self):
        print 'exportJobs'

    def importTrajectories(self):
        print 'importTrajectories'

    def exportMathematica(self):
        print 'exportMathematica'

    def getSelectedModelId(self):
        return '1'

    def getSelectedMethodInfo(self):
        return ('Direct', True, True, False)

def main():
    app = wx.PySimpleApp()
    frame = LauncherFrame()
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
