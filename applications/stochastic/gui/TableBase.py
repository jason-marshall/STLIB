import wx.grid

class TableBase(wx.grid.Grid):
    """Base class for tables displaying data."""
    def __init__(self, parent):
        wx.grid.Grid.__init__(self, parent)
        self.CreateGrid(1, 1)
        self.EnableEditing(False)

    def resize(self, numberOfRows, numberOfCols):
        """Resize the table."""
        if self.GetNumberRows() < numberOfRows:
            self.AppendRows(numberOfRows - self.GetNumberRows())
        elif self.GetNumberRows() > numberOfRows:
            self.DeleteRows(0, self.GetNumberRows() - numberOfRows)
        if self.GetNumberCols() < numberOfCols:
            self.AppendCols(numberOfCols - self.GetNumberCols())
        elif self.GetNumberCols() > numberOfCols:
            self.DeleteCols(0, self.GetNumberCols() - numberOfCols)

    def setColumnLabels(self, labels):
        for col in range(len(labels)):
            self.SetColLabelValue(col, labels[col])

