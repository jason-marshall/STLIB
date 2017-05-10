"""Trajectories list."""

import sys
import os.path
# If we are running the unit tests.
if __name__ == '__main__':
    sys.path.insert(1, '..')
    resourcePath = '../'
else:
    from resourcePath import resourcePath

from FigureNumber import FigureNumber
from Plot import Plot
from Export import Export
from TableTimeSeriesFrames import TableFrame as TfFrame
from TableTimeSeriesAllReactions import TableFrame as TarFrame
from TableHistogram import TableHistogramFrame
from TableHistogramFrames import TableErrorFrame
from TableHistogramAverage import TableErrorFrame as TableErrorAverageFrame
from TableHistogramFramesStatistics import TableHistogramFramesStatistics
from TableHistogramAverageStatistics import TableHistogramAverageStatistics
from TableStatisticsFrames import TableStatisticsFrames
from TableStatisticsAverage import TableStatisticsAverage
from HistogramDistance import HistogramDistance
from PValueMean import PValueMean
from SpeciesFrameDialog import SpeciesFrameDialog
from StateModified import StateModified

import wx

class TrajectoriesListButtons(wx.Panel):
    def __init__(self, parent=None):
        wx.Panel.__init__(self, parent)

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/cancel.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.delete = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/cancelDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.delete.SetBitmapDisabled(bmp)
        self.delete.SetToolTip(wx.ToolTip('Left click to delete selected output. Right click to delete all output.'))

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/up.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.moveUp = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/upDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.moveUp.SetBitmapDisabled(bmp)
        self.moveUp.SetToolTip(wx.ToolTip('Move up.'))

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/down.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.moveDown = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/downDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.moveDown.SetBitmapDisabled(bmp)
        self.moveDown.SetToolTip(wx.ToolTip('Move down.'))

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/plot.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.plot = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/plotDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.plot.SetBitmapDisabled(bmp)
        self.plot.SetToolTip(wx.ToolTip('Plot populations or reaction counts.'))

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/gnuplot.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.gnuplot = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/gnuplotDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.gnuplot.SetBitmapDisabled(bmp)
        self.gnuplot.SetToolTip(wx.ToolTip('Export data to gnuplot files.'))

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/x-office-spreadsheet.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.table = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                           'gui/icons/16x16/x-office-spreadsheetDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.table.SetBitmapDisabled(bmp)
        self.table.SetToolTip(wx.ToolTip('Display data in a table.'))

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/HistogramDistance.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.distance = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/HistogramDistanceDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.distance.SetBitmapDisabled(bmp)
        self.distance.SetToolTip(wx.ToolTip('Compute histogram distance.'))

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/pValue.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.pValue = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/pValueDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.pValue.SetBitmapDisabled(bmp)
        self.pValue.SetToolTip(wx.ToolTip('Compute p-value for mean equality.'))

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/filesave.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.export = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/filesaveDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.export.SetBitmapDisabled(bmp)
        self.export.SetToolTip(wx.ToolTip('Export data to a file.'))

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.delete)
        sizer.Add(self.moveUp)
        sizer.Add(self.moveDown)
        sizer.Add(self.plot)
        sizer.Add(self.gnuplot)
        sizer.Add(self.table)
        sizer.Add(self.distance)
        sizer.Add(self.pValue)
        sizer.Add(self.export)
        self.SetSizer(sizer)

    def setSelectedState(self, up, down):
        """Enable the buttons when an output is selected. up and down are
        boolean values than indicate if the item can be moved up and down."""
        self.delete.Enable()
        self.moveUp.Enable(up)
        self.moveDown.Enable(down)
        self.gnuplot.Enable()
        # CONTINUE: Give tables the same treatment as plotting.
        self.table.Enable()

    def setUnselectedState(self):
        self.delete.Disable()
        self.moveUp.Disable()
        self.moveDown.Disable()
        self.gnuplot.Disable()
        self.table.Disable()


class TrajectoriesList(wx.Panel, StateModified):
    def __init__(self, parent, application, title='Output'):
        """Parameters:
        - parent: The parent widget.
        """
        wx.Panel.__init__(self, parent)

        self.application = application
        self.title = wx.StaticText(self, -1, title)
        self.title.SetToolTip(wx.ToolTip('The list of simulation output. For each model and method the number of generated trajectories is shown. With the simulation output you can: make a plot, make a table, calculate the distance between histograms, or export to a text file.'))
        self.figureNumber = FigureNumber()
        # The dictionary of child windows. These are used to plot, compute
        # histogram distances, etc. They are stored in a dictionary. The keys 
        # are the window identifiers.
        self.children = {}

        self.buttons = TrajectoriesListButtons(self)
        self.buttons.delete.Bind(wx.EVT_LEFT_DOWN, self.onDelete)
        self.buttons.delete.Bind(wx.EVT_RIGHT_DOWN, self.onDeleteAll)
        self.buttons.moveUp.Bind(wx.EVT_BUTTON, self.onMoveUp)
        self.buttons.moveDown.Bind(wx.EVT_BUTTON, self.onMoveDown)
        self.buttons.plot.Bind(wx.EVT_BUTTON, self.onPlot)
        self.buttons.gnuplot.Bind(wx.EVT_BUTTON, self.onGnuplot)
        self.buttons.table.Bind(wx.EVT_BUTTON, self.onTable)
        self.buttons.distance.Bind(wx.EVT_BUTTON, self.onDistance)
        self.buttons.pValue.Bind(wx.EVT_BUTTON, self.onPValue)
        self.buttons.export.Bind(wx.EVT_BUTTON, self.onExport)

        # Report with single selection.
        self.list = wx.ListCtrl(self, -1, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        # The columns.
        labels = ['Model', 'Method', 'Trajectories']
        self.minimumWidth = []
        for col in range(3):
            self.list.InsertColumn(col, labels[col])
            self.minimumWidth.append(
                self.list.GetTextExtent(labels[col])[0] + 10)

        # Selection callbacks.
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.onItemSelected, self.list)
        self.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.onItemDeselected, self.list)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.title, 0, wx.ALL, 0)
        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND|wx.ALL, 1)
        sizer.Add(self.buttons, 0)
        sizer.Add(self.list, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()
        self.updateState()

    def tearDown(self):
        for child in self.children.values():
            child.Destroy()

    def updateChildren(self):
        for x in self.children.values():
            x.refresh()

    def clear(self):
        self.list.DeleteAllItems()
        self.updateState()
        self.updateChildren()
        self.processEventStateModified()

    def autosize(self):
        for col in range(3):
            self.list.SetColumnWidth(col, wx.LIST_AUTOSIZE)
            if self.list.GetColumnWidth(col) < self.minimumWidth[col]:
                self.list.SetColumnWidth(col, self.minimumWidth[col])

    def insertItem(self, model, parameters, trajectories):
        index = self.list.GetItemCount()
        self.list.InsertStringItem(index, model)
        self.list.SetStringItem(index, 1, parameters)
        self.list.SetStringItem(index, 2, trajectories)
        self.list.SetItemState(index, wx.LIST_STATE_SELECTED, 
                               wx.LIST_STATE_SELECTED)
        self.autosize()
        self.updateState()
        self.updateChildren()
        self.processEventStateModified()

    def update(self, model, parameters, trajectories):
        index = self.find(model, parameters)
        # If the item is not in the list.
        if index == -1:
            self.insertItem(model, parameters, trajectories)
        else:
            self.list.SetStringItem(index, 2, trajectories)
            self.list.SetItemState(index, wx.LIST_STATE_SELECTED, 
                                   wx.LIST_STATE_SELECTED)
        self.autosize()
        self.updateState()
        # No need to update the children if only number of trajectories changes.
        self.processEventStateModified()

    def find(self, model, parameters):
        """Find the specified item."""
        index = -1
        while True:
            index = self.list.FindItem(index, model)
            # If the search failed.
            if index == -1:
                break
            # If we found the right row.
            if self.list.GetItem(index, 0).GetText() == model and\
                    self.list.GetItem(index, 1).GetText() == parameters:
                break
            # Move to the next row to start the next search.
            index += 1
        return index

    def changeModelId(self, old, new):
        for index in range(self.list.GetItemCount()):
            item = self.list.GetItem(index, 0)
            if item.GetText() == old:
                self.list.SetStringItem(index, 0, new)

    def changeMethodId(self, old, new):
        for index in range(self.list.GetItemCount()):
            item = self.list.GetItem(index, 1)
            if item.GetText() == old:
                self.list.SetStringItem(index, 1, new)

    def getText(self, index):
        return self.list.GetItemText(index)

    def getSelectedIndex(self):
        """Return the index of the selected item, or -1 if no item is 
        selected."""
        for item in range(self.list.GetItemCount()):
            if self.list.GetItemState(item, wx.LIST_STATE_SELECTED):
                return item
        return -1

    def getSelectedText(self):
        """Return the text of the selected item, or None if no item is 
        selected."""
        index = self.getSelectedIndex()
        if index == -1:
            return None
        return self.list.GetItemText(index)

    def getInfo(self):
        """Return a tuple of the model identifier, the parameters identifier,
        and the trajectory class name. If there is an error, return a tuple of 
        None's."""
        # Get the set of trajectories to plot.
        index = self.getSelectedIndex()
        if index == -1:
            return (None, None, None)
        modelId = self.list.GetItem(index, 0).GetText()
        methodId = self.list.GetItem(index, 1).GetText()
        state = self.application.state
        output = state.output[(modelId, methodId)]
        if output.empty():
            return (None, None, None)
        # Return the tuple.
        return (modelId, methodId, output.__class__.__name__)

    def onPlot(self, event):
        frame = Plot(self, 'Plot time series or histogram data.', 
                     self.application.state, self.figureNumber)
        frame.Show()
        # Add to the dictionary.
        self.children[frame.GetId()] = frame

    def onGnuplot(self, event):
        modelId, methodId, style = self.getInfo()
        # If there was a problem, do nothing.
        if not modelId:
            return
        self.application.exportGnuplot(modelId, methodId)

    def onTable(self, event):
        modelId, methodId, style = self.getInfo()
        # If there was a problem, do nothing.
        if not modelId:
            return
        # Choose between "frames" and "all reactions".
        if style == 'TimeSeriesFrames':
            self.tableTimeSeriesFrames(modelId, methodId)
        elif style == 'TimeSeriesAllReactions':
            self.tableTimeSeriesAllReactions(modelId, methodId)
        elif style == 'HistogramFrames':
            self.tableHistogramFrames(modelId, methodId)
        elif style == 'HistogramAverage':
            self.tableHistogramAverage(modelId, methodId)
        elif style == 'StatisticsFrames':
            self.tableStatisticsFrames(modelId, methodId)
        elif style == 'StatisticsAverage':
            self.tableStatisticsAverage(modelId, methodId)
        else:
            assert False

    def tableTimeSeriesFrames(self, modelId, methodId):
        state = self.application.state
        model = state.models[modelId]
        output = state.output[(modelId, methodId)]
        # There must be recorded species or reactions.
        assert output.recordedSpecies or output.recordedReactions

        choices = []
        populations = 'Populations'
        if output.recordedSpecies:
            choices.extend([populations])
        binned = 'Binned reaction counts'
        cumulative = 'Cumulative reaction counts'
        if output.recordedReactions:
            choices.extend([binned, cumulative])

        data = wx.GetSingleChoiceIndex('Choose data to display.', 'Table',
                                       choices)
        title = ' for Model: ' + modelId + ', Method: ' +\
            methodId + '.'
        if populations in choices and data == choices.index(populations):
            style = wx.GetSingleChoiceIndex\
                ('Choose how to display the populations', 'Table',
                 ['Statistics for all frames',
                  'Statistics for the last frame',
                  'Ensemble showing all frames',
                  'Ensemble showing the last frame'])
            if style == 0:
                TfFrame(model, output, 'PopulationStatisticsAll',
                        'Population statistics' + title, self).Show()
            elif style == 1:
                TfFrame(model, output, 'PopulationStatisticsLast',
                        'Population statistics' + title, self).Show()
            elif style == 2:
                TfFrame(model, output, 'PopulationsAll',
                        'Populations' + title, self).Show()
            elif style == 3:
                TfFrame(model, output, 'PopulationsLast',
                        'Populations' + title, self).Show()
        elif binned in choices and data == choices.index(binned):
            style = wx.GetSingleChoiceIndex\
                ('Choose how to display the binned reaction counts',
                 'Table', ['Statistics', 'Ensemble'])
            if style == 0:
                TfFrame(model, output, 'ReactionStatisticsBinned',
                        'Binned reaction count statistics' + title, self).Show()
            elif style == 1:
                TfFrame(model, output, 'ReactionsBinned',
                        'Binned reaction counts' + title, self).Show()
        elif cumulative in choices and data == choices.index(cumulative):
            style = wx.GetSingleChoiceIndex\
                ('Choose how to display the cumulative reaction counts',
                 'Table',
                 ['Statistics for all frames',
                  'Statistics for the last frame',
                  'Ensemble showing all frames',
                  'Ensemble showing the last frame'])
            if style == 0:
                TfFrame(model, output, 'ReactionStatisticsAll',
                        'Cumulative reaction count statistics' + title,
                        self).Show()
            elif style == 1:
                TfFrame(model, output, 'ReactionStatisticsLast',
                        'Cumulative reaction count statistics' + title,
                        self).Show()
            elif style == 2:
                TfFrame(model, output, 'ReactionsAll',
                        'Cumulative reaction counts' + title, self).Show()
            elif style == 3:
                TfFrame(model, output, 'ReactionsLast',
                        'Cumulative reaction counts' + title, self).Show()

    def tableTimeSeriesAllReactions(self, modelId, methodId):
        data = wx.GetSingleChoiceIndex\
            ('Choose data to display.', 'Table', 
             ['Populations', 'Reaction counts'])
        state = self.application.state
        model = state.models[modelId]
        output = state.output[(modelId, methodId)]
        title = ' for Model: ' + modelId + ', Method: ' +\
            methodId + '.'
        if data == 0:
            style = wx.GetSingleChoiceIndex\
                ('Choose how to display the populations', 'Table',
                 ['Ensemble showing all reactions',
                  'Ensemble showing the final populations'])
            if style == 0:
                TarFrame(model, output, 'PopulationsAll',
                         'Populations' + title, self).Show()
            elif style == 1:
                TarFrame(model, output, 'PopulationsEnd',
                         'Populations' + title, self).Show()
        elif data == 1:
            style = wx.GetSingleChoiceIndex\
                ('Choose how to display the reaction counts',
                 'Table',
                 ['Ensemble showing all reactions',
                  'Ensemble showing the final reaction counts'])
            if style == 0:
                TarFrame(model, output, 'ReactionsAll',
                         'Reaction counts' + title, self).Show()
            elif style == 1:
                TarFrame(model, output, 'ReactionsEnd',
                         'Reaction counts' + title, self).Show()

    def tableHistogramFrames(self, modelId, methodId):
        # The model and output.
        state = self.application.state
        model = state.models[modelId]
        output = state.output[(modelId, methodId)]

        # Choose between showing the estimated error or the histogram.
        index = wx.GetSingleChoiceIndex('Choose table content.', 'Table', 
                                        ['Estimated error',
                                         'Mean and standard deviation',
                                         'Histogram bin values'])

        if index == 0:
            # Show the errors.
            title = 'Estimated errors for model: ' + modelId + ', method: ' +\
                methodId + '.'
            TableErrorFrame(model, output, title, self).Show()
        elif index == 1:
            # Show population statistics.
            title = 'Statistics for model: ' + modelId + ', method: ' +\
                methodId + '.'
            TableHistogramFramesStatistics(model, output, title, self).Show()
        elif index == 2:
            # Select a species and frame.
            dialog = SpeciesFrameDialog(self, model, output)
            result = dialog.ShowModal()
            species = dialog.getSpecies()
            frame = dialog.getFrame()
            dialog.Destroy()
            if result != wx.ID_OK:
                return
            # Show the histogram.
            speciesId =\
                model.speciesIdentifiers[output.recordedSpecies[species]]
            title = 'Histogram for model: ' + modelId + ', method: ' +\
                methodId + ', species: ' + speciesId +\
                ', frame: ' + str(output.frameTimes[frame]) + '.'
            TableHistogramFrame(output.histograms[frame][species], title,
                                self).Show()

    def tableHistogramAverage(self, modelId, methodId):
        # The model and output.
        state = self.application.state
        model = state.models[modelId]
        output = state.output[(modelId, methodId)]

        # Choose between showing the estimated error or the histogram.
        index = wx.GetSingleChoiceIndex('Choose table content.', 'Table', 
                                        ['Estimated error',
                                         'Mean and standard deviation',
                                         'Histogram bin values'])

        if index == 0:
            # Show the errors.
            title = 'Estimated errors for model: ' + modelId + ', method: ' +\
                methodId + '.'
            TableErrorAverageFrame(model, output, title, self).Show()
        elif index == 1:
            # Show population statistics.
            title = 'Statistics for model: ' + modelId + ', method: ' +\
                methodId + '.'
            TableHistogramAverageStatistics(model, output, title, self).Show()
        elif index == 2:
            # Select a species.
            choices = [model.speciesIdentifiers[_i] for _i in
                       output.recordedSpecies]
            selection = wx.GetSingleChoiceIndex('Choose a species.', 'Species', 
                                                choices)
            if selection < 0:
                return
            # Show the histogram.
            speciesId =\
                model.speciesIdentifiers[output.recordedSpecies[selection]]
            title = 'Histogram for model: ' + modelId + ', method: ' +\
                methodId + ', species: ' + speciesId + '.'
            TableHistogramFrame(output.histograms[selection], title,
                                self).Show()

    def tableStatisticsFrames(self, modelId, methodId):
        # The model and output.
        state = self.application.state
        model = state.models[modelId]
        output = state.output[(modelId, methodId)]

        # Show population statistics.
        title = 'Statistics for model: ' + modelId + ', method: ' +\
            methodId + '.'
        TableStatisticsFrames(model, output, title, self).Show()

    def tableStatisticsAverage(self, modelId, methodId):
        # The model and output.
        state = self.application.state
        model = state.models[modelId]
        output = state.output[(modelId, methodId)]

        # Show population statistics.
        title = 'Statistics for model: ' + modelId + ', method: ' +\
            methodId + '.'
        TableStatisticsAverage(model, output, title, self).Show()

    def onDistance(self, event):
        frame = HistogramDistance(self.application.state, 'Histogram distance',
                                  self)
        frame.Show()
        # Add to the dictionary.
        self.children[frame.GetId()] = frame

    def onPValue(self, event):
        frame = PValueMean(self.application.state, 'P-value for mean equality',
                           self)
        frame.Show()
        # Add to the dictionary.
        self.children[frame.GetId()] = frame

    def onExport(self, event):
        frame = Export(self, 'Export time series or histogram data.', 
                       self.application.state)
        frame.Show()
        # Add to the dictionary.
        self.children[frame.GetId()] = frame

    def onDelete(self, event):
        index = self.getSelectedIndex()
        if index != -1:
            self.application.deleteOutput\
                (self.list.GetItem(index, 0).GetText(), 
                 self.list.GetItem(index, 1).GetText())
            self.list.DeleteItem(index)
        self.autosize()
        self.updateState()
        self.updateChildren()
        self.processEventStateModified()

    def onDeleteAll(self, event):
        self.application.deleteAllOutput()
        self.clear()
        # No need to call processEventStateModified(), that is done in clear().

    def onMoveUp(self, event):
        index = self.getSelectedIndex()
        # If we can move it up.
        if index != -1 and index != 0:
            for col in range(3):
                tmp = self.list.GetItem(index - 1, col).GetText()
                self.list.SetStringItem(index - 1, col, 
                                        self.list.GetItem(index, col).GetText())
                self.list.SetStringItem(index, col, tmp)
            self.list.SetItemState(index - 1, wx.LIST_STATE_SELECTED, 
                                   wx.LIST_STATE_SELECTED)
        self.updateState()

    def onMoveDown(self, event):
        index = self.getSelectedIndex()
        # If we can move it up.
        if index != -1 and index != self.list.GetItemCount() - 1:
            for col in range(3):
                tmp = self.list.GetItem(index + 1, col).GetText()
                self.list.SetStringItem(index + 1, col, 
                                        self.list.GetItem(index, col).GetText())
                self.list.SetStringItem(index, col, tmp)
            self.list.SetItemState(index + 1, wx.LIST_STATE_SELECTED, 
                                   wx.LIST_STATE_SELECTED)
        self.updateState()

    def updateState(self):
        index = self.getSelectedIndex()
        if index == -1:
            self.buttons.setUnselectedState()
        else:
            self.buttons.setSelectedState(index != 0, 
                                          index != self.list.GetItemCount() - 1)

    def onItemSelected(self, event):
        self.updateState()

    def onItemDeselected(self, event):
        self.buttons.setUnselectedState()

#
# Test code.
#

# CONTINUE
class State:
    def __init__(self):
        self.identifiers = []

    def delete(self, model, method):
        print 'delete', model, method

    def plotPopulationStatistics(self, model, method):
        print 'plotPopulationStatistics', model, method

    def plotReactionStatistics(self, model, method):
        print 'plotReactionStatistics', model, method

    def plotPopulations(self, model, method):
        print 'plotPopulations', model, method

    def plotReactions(self, model, method):
        print 'plotReactions', model, method

    def displayStatisticsLast(self, model, method):
        print 'display statistics last', model, method

    def displayStatisticsAll(self, model, method):
        print 'display statistics all', model, method

    def displayLast(self, model, method):
        print 'display last', model, method

    def displayAll(self, model, method):
        print 'display all', model, method

    def csv(self, model, method):
        print 'csv', model, method

    def gnuplot(self, model, method):
        print 'gnuplot', model, method

class TrajectoriesListFrame(wx.Frame):
    def __init__(self, parent=None):
        wx.Frame.__init__(self, parent, title='Output',
                          size=(500,300))
        self.state = State()
        editor = TrajectoriesList(self, self)
        editor.insertItem('m1', 'p1', '100')
        editor.insertItem('m2', 'p2', '200')
        editor.insertItem('m3', 'p3', '300')

    def exportCsv(self, modelId, methodId):
        print 'exportCsv', modelId, methodId

    def exportGnuplot(self, modelId, methodId):
        print 'exportGnuplot', modelId, methodId

    def deleteOutput(self, modelId, methodId):
        print 'deleteOutput', modelId, methodId


def main():
    app = wx.PySimpleApp()
    frame = TrajectoriesListFrame()
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
