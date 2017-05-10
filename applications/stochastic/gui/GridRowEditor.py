"""Grid editor that can be used to edit species or reactions."""

import os, os.path
import sys
# If we are running the test code.
if os.path.split(os.getcwd())[1] == 'gui':
    sys.path.insert(1, '..')
    resourcePath = '../'
else:
    from resourcePath import resourcePath

import wx
import wx.grid
from wx.lib.buttons import GenBitmapToggleButton
from StateModified import StateModified

class FirstColumnModifiedEvent(wx.PyCommandEvent):
    """Event that is processed when the first column of a GridRowEditor is
    modified."""
    def __init__(self, evtType, id):
        wx.PyCommandEvent.__init__(self, evtType, id)

# Generate an event type.
EVT_FIRST_COLUMN_MODIFIED_TYPE = wx.NewEventType()
# Create a binder object.
EVT_FIRST_COLUMN_MODIFIED = wx.PyEventBinder(EVT_FIRST_COLUMN_MODIFIED_TYPE, 1)

class GridRowEditorButtons(wx.Panel):
    def __init__(self, parent=None):
        wx.Panel.__init__(self, parent)
        self.isEditable = True

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/add.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.insert = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/addDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.insert.SetBitmapDisabled(bmp)
        self.insert.SetToolTip(wx.ToolTip('Insert a row.'))

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/cancel.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.delete = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/cancelDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.delete.SetBitmapDisabled(bmp)
        self.delete.SetToolTip(wx.ToolTip('Delete selected rows.'))

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/up.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.moveUp = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/upDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.moveUp.SetBitmapDisabled(bmp)
        self.moveUp.SetToolTip(wx.ToolTip('Left click to move row up. Right click to move row to top.'))

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/down.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.moveDown = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/downDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.moveDown.SetBitmapDisabled(bmp)
        self.moveDown.SetToolTip(wx.ToolTip('Left click to move row down. Right click to move row to bottom.'))

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/sort.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.sort = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/sortDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.sort.SetBitmapDisabled(bmp)
        self.sort.SetToolTip(wx.ToolTip('Left click to sort in ascending order. Right click for descending order.'))

        bmp = wx.Image(os.path.join(resourcePath, 'gui/icons/16x16/scale.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.autoSize = wx.BitmapButton(self, -1, bmp)
        if sys.platform in ('win32', 'win64'):
            bmp = wx.Image(os.path.join(resourcePath,
                                        'gui/icons/16x16/scaleDisabled.png'),
                           wx.BITMAP_TYPE_PNG).ConvertToBitmap()
            self.autoSize.SetBitmapDisabled(bmp)
        self.autoSize.SetToolTip(wx.ToolTip('Auto size cells.'))

        bmp = wx.Image(os.path.join(resourcePath,
                                    'gui/icons/16x16/system-search.png'),
                       wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.expand = GenBitmapToggleButton(self, -1, bmp,
                                            size=self.autoSize.GetSize())
        self.expand.SetToolTip(wx.ToolTip('Show/hide optional fields.'))

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.insert)
        sizer.Add(self.delete)
        sizer.Add(self.moveUp)
        sizer.Add(self.moveDown)
        sizer.Add(self.sort)
        sizer.Add(self.autoSize)
        sizer.Add(self.expand)
        self.SetSizer(sizer)
        self.setCellSelectedState()

    def enable(self):
        self.isEditable = True
        self.insert.Enable()
        self.delete.Enable()
        self.moveUp.Enable()
        self.moveDown.Enable()
        self.sort.Enable()

    def disable(self):
        self.isEditable = False
        self.insert.Disable()
        self.delete.Disable()
        self.moveUp.Disable()
        self.moveDown.Disable()
        self.sort.Disable()

    def setCellSelectedState(self):
        self.insert.Enable(self.isEditable)
        self.delete.Disable()
        self.moveUp.Disable()
        self.moveDown.Disable()
        self.sort.Enable(self.isEditable)

    def setRowSelectedState(self, up, down):
        self.insert.Enable(self.isEditable)
        self.delete.Enable(self.isEditable)
        self.moveUp.Enable(self.isEditable and up)
        self.moveDown.Enable(self.isEditable and down)
        self.sort.Enable(self.isEditable)

    def setMultiRowSelectedState(self):
        self.insert.Enable(self.isEditable)
        self.delete.Enable(self.isEditable)
        self.moveUp.Disable()
        self.moveDown.Disable()
        self.sort.Enable(self.isEditable)

class GridRowEditorGrid(wx.grid.Grid):
    def __init__(self, parent, columnLabels, columnSizes, boolean, details):
        wx.grid.Grid.__init__(self, parent)
        self.CreateGrid(0, len(columnLabels))
        self.SetColLabelSize(18)
        for n in range(len(columnLabels)):
            self.SetColLabelValue(n, columnLabels[n])
        self.columnSizes = columnSizes
        # The list of boolean columns.
        self.boolean = boolean
        # The list of columns that are non-essential details.
        self.details = details
        # This width should be enough room for a four digit label.
        # Sufficient for less than 10,000 rows.
        self.SetRowLabelSize(50)
        # Set the minimal column widths.
        self.SetColMinimalAcceptableWidth(0)
        for i in self.details:
            self.SetColMinimalWidth(i, 0)
        # Set the column sizes.
        assert len(columnLabels) >= len(columnSizes)
        for i in range(len(columnSizes)):
            self.SetColSize(i, columnSizes[i])
        # Hide the details.
        self.hideDetails()

    def getTableData(self):
        data = []
        for row in range(self.GetNumberRows()):
            rowData = []
            for col in range(self.GetNumberCols()):
                rowData.append(self.GetCellValue(row, col))
            data.append(rowData)
        return data

    def setTableData(self, data):
        if self.GetNumberRows() < len(data):
            self.AppendRows(len(data) - self.GetNumberRows())
        elif self.GetNumberRows() > len(data):
            self.DeleteRows(len(data), self.GetNumberRows() - len(data))
        for row in range(len(data)):
            assert len(data[row]) == self.GetNumberCols()
            for col in range(len(data[row])):
                self.SetCellValue(row, col, data[row][col])

    def setBoolRendererEditor(self, pos, numRows):
        for row in range(pos, pos + numRows):
            for col, value in self.boolean:
                self.SetCellRenderer(row, col, wx.grid.GridCellBoolRenderer())
                self.SetCellEditor(row, col, wx.grid.GridCellBoolEditor())
                # value is '' for False and '1' for True.
                self.SetCellValue(row, col, value)

    def InsertRows(self, pos=0, numRows=1, updateLabels=True):
        if not wx.grid.Grid.InsertRows(self, pos, numRows, updateLabels):
            return False
        self.setBoolRendererEditor(pos, numRows)
        return True

    def AppendRows(self, numRows=1, updateLabels=True):
        pos = self.GetNumberRows()
        if not wx.grid.Grid.AppendRows(self, numRows, updateLabels):
            return False
        self.setBoolRendererEditor(pos, numRows)
        return True

    def showDetails(self):
        for i in self.details:
            if len(self.columnSizes) > i:
                self.SetColSize(i, self.columnSizes[i])
            else:
                self.SetColSize(i, 80)
        self.ForceRefresh()

    def hideDetails(self):
        for i in self.details:
            self.SetColSize(i, 0)
        self.ForceRefresh()

class GridRowEditor(wx.Panel, StateModified):
    def __init__(self, parent, title, columnLabels, columnSizes=[], boolean=[],
                 details=[], toolTip=None):
        """boolean is a list of (column, initialValue) tuples The initial
        value is '' for false and '1' for true."""
        wx.Panel.__init__(self, parent)

        if title:
            self.title = wx.StaticText(self, -1, title)
            if toolTip:
                self.title.SetToolTip(wx.ToolTip(toolTip))
        else:
            self.title = None

        self.buttons = GridRowEditorButtons(self)
        self.buttons.insert.Bind(wx.EVT_BUTTON, self.onInsert)
        self.buttons.delete.Bind(wx.EVT_BUTTON, self.onDelete)
        self.buttons.moveUp.Bind(wx.EVT_LEFT_DOWN, self.onMoveUp)
        self.buttons.moveUp.Bind(wx.EVT_RIGHT_DOWN, self.onMoveTop)
        self.buttons.moveDown.Bind(wx.EVT_LEFT_DOWN, self.onMoveDown)
        self.buttons.moveDown.Bind(wx.EVT_RIGHT_DOWN, self.onMoveBottom)
        self.buttons.sort.Bind(wx.EVT_LEFT_DOWN, self.onSortAscending)
        self.buttons.sort.Bind(wx.EVT_RIGHT_DOWN, self.onSortDescending)
        self.buttons.autoSize.Bind(wx.EVT_BUTTON, self.onAutoSize)
        self.buttons.expand.Bind(wx.EVT_BUTTON, self.onExpand)

        self.grid = GridRowEditorGrid(self, columnLabels, columnSizes,
                                      boolean, details)

        # Cell and label selection.
        self.Bind(wx.grid.EVT_GRID_SELECT_CELL, self.onSelectCell, self.grid)
        self.Bind(wx.grid.EVT_GRID_LABEL_LEFT_CLICK, self.onLabelClick,
                  self.grid)
        self.Bind(wx.grid.EVT_GRID_CELL_CHANGE, self.onCellChange, self.grid)
        # CONTINUE I would like to capture multi-row selection, but I don't
        # know how. I can't capture mouse events in the grid.

        sizer = wx.BoxSizer(wx.VERTICAL)
        if title:
            sizer.Add(self.title, 0, wx.ALL, 0)
            sizer.Add(wx.StaticLine(self), 0, wx.EXPAND|wx.ALL, 1)
        sizer.Add(self.buttons, 0)
        sizer.Add(self.grid, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

    def enable(self):
        self.buttons.isEditable = True
        self.update()
        #self.buttons.enable()
        self.grid.EnableEditing(True)

    def disable(self):
        self.buttons.disable()
        self.grid.EnableEditing(False)

    def getTableData(self):
        return self.grid.getTableData()

    def setTableData(self, data):
        self.grid.setTableData(data)
        self.update()

    def onAutoSize(self, event):
        self.grid.AutoSizeColumns(False)
        if not self.buttons.expand.GetValue():
            self.grid.hideDetails()
        self.Layout()
        
    def onExpand(self, event):
        if self.buttons.expand.GetValue():
            self.grid.showDetails()
        else:
            self.grid.hideDetails()
        self.Layout()
        
    def onInsert(self, event):
        selected = self.grid.GetSelectedRows()
        if selected:
            row = selected[0]
        else:
            row = self.grid.GetNumberRows()
        self.grid.InsertRows(row)
        self.Layout()
        self.update()
        self.processFirstColumnModifiedEvent()
        self.processEventStateModified()

    def onDelete(self, event):
        selected = self.grid.GetSelectedRows()
        selected.sort()
        selected.reverse()
        for row in selected:
            self.grid.DeleteRows(row)
        self.Layout()
        self.update()
        self.processFirstColumnModifiedEvent()
        self.processEventStateModified()

    def getSelectedRow(self):
        selected = self.grid.GetSelectedRows()
        if len(selected) != 1:
            return -1
        return selected[0]

    def isRowValid(self, row):
        return row >= 0 and row < self.grid.GetNumberRows()

    def move(self, source, target):
        if self.isRowValid(source) and self.isRowValid(target) and\
                source != target:
            # Copy the values.
            values = [self.grid.GetCellValue(source, col) for col in
                      range(self.grid.GetNumberCols())]
            # Erase the source row.
            self.grid.DeleteRows(source)
            # Insert a row and copy the values.
            self.grid.InsertRows(target)
            for col in range(len(values)):
                self.grid.SetCellValue(target, col, values[col])
            self.grid.SetGridCursor(target, 0)
            self.grid.SelectRow(target)
        self.update()
        self.processFirstColumnModifiedEvent()
        self.processEventStateModified()

    def onMoveUp(self, event):
        row = self.getSelectedRow()
        self.move(row, row - 1)

    def onMoveTop(self, event):
        row = self.getSelectedRow()
        self.move(row, 0)

    def onMoveDown(self, event):
        row = self.getSelectedRow()
        self.move(row, row + 1)

    def onMoveBottom(self, event):
        row = self.getSelectedRow()
        self.move(row, self.grid.GetNumberRows() - 1)

    def sort(self, isAscending):
        numRows = self.grid.GetNumberRows()
        numCols = self.grid.GetNumberCols()
        tableRows = {}
        for row in range(numRows):
            tableRows[self.grid.GetCellValue(row, 0)] =\
                [self.grid.GetCellValue(row, col) for col in range(numCols)]
        if len(tableRows) != numRows:
	    wx.MessageBox('The identifiers are not distinct.',
			  'Error!', style=wx.OK|wx.ICON_EXCLAMATION)
            return
        keys = tableRows.keys()
        keys.sort()
        if not isAscending:
            keys.reverse()
        row = 0
        for key in keys:
            values = tableRows[key]
            for col in range(numCols):
                self.grid.SetCellValue(row, col, values[col])
            row += 1
        self.processFirstColumnModifiedEvent()
        self.processEventStateModified()

    def onSortAscending(self, event):
        self.sort(True)

    def onSortDescending(self, event):
        self.sort(False)

    def onLabelClick(self, event):
        row = event.GetRow()
        # If a row was selected.
        if row != -1:
            self.buttons.setRowSelectedState\
                (row != 0, row != self.grid.GetNumberRows() - 1)
        # If the whole grid was selected.
        elif event.GetCol() == -1:
            self.buttons.setRowSelectedState(False, False)
        # Otherwise a column was selected.
        else:
            self.buttons.setCellSelectedState()
        event.Skip()

    def update(self):
        selected = self.grid.GetSelectedRows()
        if not selected:
            self.buttons.setCellSelectedState()
        elif len(selected) == 1:
            row = selected[0]
            self.buttons.setRowSelectedState\
                (row != 0, row != self.grid.GetNumberRows() - 1)
        else:
            self.buttons.setMultiRowSelectedState()

    def getNonEmptyCellsInFirstColumn(self):
        data = []
        for row in range(self.grid.GetNumberRows()):
            x = self.grid.GetCellValue(row, 0)
            if x:
                data.append(x)
        return data

    def onSelectCell(self, event):
        self.buttons.setCellSelectedState()
        event.Skip()

    def onCellChange(self, event):
        self.processEventStateModified()
        if event.GetCol() == 0:
            self.processFirstColumnModifiedEvent()

    def processFirstColumnModifiedEvent(self):
        # Create the event.
        event = FirstColumnModifiedEvent(EVT_FIRST_COLUMN_MODIFIED_TYPE,
                                         self.GetId())
        # Process the event.
        self.GetEventHandler().ProcessEvent(event)
        
if __name__ == '__main__':
    class GridRowEditorFrame(wx.Frame):
        def __init__(self, parent=None):
            wx.Frame.__init__(self, parent, title='Reaction Editor',
                              size=(600,300))
            self.editor = GridRowEditor(self, 'Reactions',
                                        ['ID', 'Name', 'Reactants', 'Products',
                                         'MA', 'Propensity'],
                                        [40, 80, 80, 80, 25, 80],
                                        boolean=[(4, '1')], details=[1])
            # Bind the event.
            self.Bind(EVT_FIRST_COLUMN_MODIFIED, self.firstColumnModified,
                      self.editor)

        def firstColumnModified(self, event):
            print('First column modified.',
                  self.editor.getNonEmptyCellsInFirstColumn())

    app = wx.PySimpleApp()
    frame = GridRowEditorFrame()
    frame.Show()
    app.MainLoop()
