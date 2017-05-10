"""Grid of the species to export."""

import os, os.path
# If we are running the test code.
if os.path.split(os.getcwd())[1] == 'gui':
    import sys
    sys.path.insert(1, '..')

import wx
import wx.grid
import colorsys

class ExportHistogramGrid(wx.grid.Grid):
    def __init__(self, parent):
        wx.grid.Grid.__init__(self, parent)
        columnLabels = ['Selected']
        self.CreateGrid(0, len(columnLabels))
        v = [int(_x) for _x in wx.__version__.split('.')]
        assert(len(v) >= 3)
        wxVersion = 100 * v[0] + 10 * v[1] + v[2]
        if wxVersion >= 288:
            # CONTINUE: This does not work.
            #self.SetColLabelSize(wx.grid.GRID_AUTOSIZE)
            self.SetColLabelSize(36)
        else:
            self.SetColLabelSize(36)
        for n in range(len(columnLabels)):
            self.SetColLabelValue(n, columnLabels[n])
        # CONTINUE: Determine this from the identifiers.
        self.SetRowLabelSize(160)
        self.SetRowLabelAlignment(wx.ALIGN_LEFT, wx.ALIGN_CENTRE)
        # Set the minimal column widths.
        self.SetColMinimalAcceptableWidth(0)

        # Renderers and editors.
        boolRenderer = wx.grid.GridCellBoolRenderer()
        boolEditor = wx.grid.GridCellBoolEditor()
        
        #
        # Grid column attributes.
        #
        # Selected.
        attr = wx.grid.GridCellAttr()
        attr.SetRenderer(boolRenderer)
        attr.SetEditor(boolEditor)
        self.SetColAttr(0, attr)

        # Events.
        self.Bind(wx.grid.EVT_GRID_LABEL_LEFT_CLICK, self.onLabelLeftClick)
        self.Bind(wx.grid.EVT_GRID_LABEL_RIGHT_CLICK, self.onLabelRightClick)
        self.columnLabelFunctions = [self.select]

    def onLabelLeftClick(self, event):
        self.labelClick(event, True)

    def onLabelRightClick(self, event):
        self.labelClick(event, False)

    def labelClick(self, event, isLeft):
        row, col = event.GetRow(), event.GetCol()
        # Do nothing if they did not click a column label.
        if col == -1:
            return
        self.saveEditControlValue()
        self.columnLabelFunctions[col](col, isLeft)
        # Don't skip the event. This prevents the row or column from being 
        # selected.
        # Force a refresh to render the modified cells.
        self.ForceRefresh()

    def select(self, col, isLeft):
        if isLeft:
            value = '1'
        else:
            value = ''
        for row in range(self.GetNumberRows()):
            self.SetCellValue(row, col, value)

    def setIdentifiers(self, identifiers):
        if self.GetNumberRows() < len(identifiers):
            self.InsertRows(0, len(identifiers) - self.GetNumberRows())
        elif self.GetNumberRows() > len(identifiers):
            self.DeleteRows(0, self.GetNumberRows() - len(identifiers))
        for row in range(self.GetNumberRows()):
            self.SetRowLabelValue(row, identifiers[row])
            # Selected.
            # '' is False, '1' is True.
            self.SetCellValue(row, 0, '1')
        self.AutoSizeColumns(False)
        self.Layout()

    def saveEditControlValue(self):
        """Save the values being edited.."""
        if self.IsCellEditControlShown(): 
            self.SaveEditControlValue() 
            self.HideCellEditControl() 

    def getCheckedItems(self):
        """Return the list of checked item indices."""
        return filter(lambda row: self.GetCellValue(row, 0),
                      range(self.GetNumberRows()))

    def areAnyItemsSelected(self):
        return self.GetNumberRows() != 0 and\
               reduce(lambda x, y: x or y, [self.GetCellValue(row, 0) for row
                                            in range(self.GetNumberRows())])

if __name__ == '__main__':
    class Frame(wx.Frame):
        def __init__(self, parent=None):
            wx.Frame.__init__(self, parent, title='Export', size=(600,300))
            sizer = wx.BoxSizer(wx.VERTICAL)
            grid = ExportHistogramGrid(self)
            grid.setIdentifiers(['a', 'b'])
            sizer.Add(grid, 1, wx.EXPAND)
            self.SetSizer(sizer)
            self.Fit()

    app = wx.PySimpleApp()
    Frame().Show()
    app.MainLoop()
