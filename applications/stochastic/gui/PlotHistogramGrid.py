"""Grid of the items (species or frames) to plot."""

import os, os.path
# If we are running the test code.
if os.path.split(os.getcwd())[1] == 'gui':
    import sys
    sys.path.insert(1, '..')

import wx
import wx.grid
import colorsys

styleNames = ['solid', 'dashed', 'dash dot', 'dot']
styleStrings = ('-', '--', '-.', ':')

def wxToMatplotlibColor(color):
    return (color.Red() / 255.0, color.Green() / 255.0, color.Blue() / 255.0)

class PlotHistogramGrid(wx.grid.Grid):
    def __init__(self, parent):
        wx.grid.Grid.__init__(self, parent)
        columnLabels = ['Show', 'Line\nColor', 'Line\nStyle', 'Line\nWidth',
                        'Filled', 'Fill\nColor', 'Alpha', 'Legend\nLabel']
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

        # Lower and upper bounds for sizes and widths.
        self.lowerBound = 1
        self.upperBound = 10

        # Renderers and editors.
        boolRenderer = wx.grid.GridCellBoolRenderer()
        boolEditor = wx.grid.GridCellBoolEditor()
        lineStyleEditor = wx.grid.GridCellChoiceEditor(styleNames)
        numberEditor = wx.grid.GridCellNumberEditor(self.lowerBound,
                                                    self.upperBound)
        floatEditor = wx.grid.GridCellFloatEditor()
        
        #
        # Grid column attributes.
        #
        # Show.
        attr = wx.grid.GridCellAttr()
        attr.SetRenderer(boolRenderer)
        attr.SetEditor(boolEditor)
        self.SetColAttr(0, attr)
        # Filled (out of order).
        self.SetColAttr(4, attr)
        # Line Color.
        attr = wx.grid.GridCellAttr()
        attr.SetReadOnly()
        self.SetColAttr(1, attr)
        # Fill Color (out of order).
        self.SetColAttr(5, attr)
        # Line Style.
        attr = wx.grid.GridCellAttr()
        attr.SetEditor(lineStyleEditor)
        self.SetColAttr(2, attr)
        # Line Width.
        attr = wx.grid.GridCellAttr()
        attr.SetEditor(numberEditor)
        self.SetColAttr(3, attr)
        # Alpha
        attr = wx.grid.GridCellAttr()
        attr.SetEditor(floatEditor)
        self.SetColAttr(6, attr)
        # Use the default for the legend labels.
        
        # Events.
        self.Bind(wx.grid.EVT_GRID_CELL_LEFT_CLICK, self.onCellLeftClick)
        self.Bind(wx.grid.EVT_GRID_LABEL_LEFT_CLICK, self.onLabelLeftClick)
        self.Bind(wx.grid.EVT_GRID_LABEL_RIGHT_CLICK, self.onLabelRightClick)
        self.columnLabelFunctions = [self.select, self.lineColor,
                                     self.lineStyle, self.increment,
                                     self.select, self.fillColor,
                                     self.incrementAlpha, self.legendLabels]

    def onCellLeftClick(self, event):
        row, col = event.GetRow(), event.GetCol()
        if col in (1, 5):
            color = wx.GetColourFromUser(None)
            if color.IsOk():
                self.SetCellBackgroundColour(row, col, color)
        # Let the cell be selected.
        event.Skip()

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

    def _colorByHue(self, col):
        black = wx.Colour(0, 0, 0)
        # Count the selected items.
        count = 0
        for row in range(self.GetNumberRows()):
            if self.GetCellValue(row, 0):
                count += 1
        # Color the lines for the selected items.
        n = 0.0
        for row in range(self.GetNumberRows()):
            if self.GetCellValue(row, 0):
                rgb = colorsys.hsv_to_rgb(n / count, 1, 1)
                color = wx.Colour(rgb[0] * 255, rgb[1] * 255, rgb[2] * 255)
                self.SetCellBackgroundColour(row, col, color)
                n += 1
            else:
                self.SetCellBackgroundColour(row, col, black)

    def lineColor(self, col, isLeft):
        if isLeft:
            self._colorByHue(col)
        else:
            black = wx.Colour(0, 0, 0)
            for row in range(self.GetNumberRows()):
                self.SetCellBackgroundColour(row, col, black)

    def fillColor(self, col, isLeft):
        if isLeft:
            # Color by hue.
            self._colorByHue(col)
        else:
            # Match the line colors.
            for row in range(self.GetNumberRows()):
                self.SetCellBackgroundColour\
                    (row, col, self.GetCellBackgroundColour(row, 1))

    def lineStyle(self, col, isLeft):
        if isLeft:
            # Cycle between the available line styles.
            n = 0
            for row in range(self.GetNumberRows()):
                if self.GetCellValue(row, 0):
                    self.SetCellValue(row, col, styleNames[n])
                    n = (n + 1) % len(styleNames)
        else:
            # Set all line styles to solid.
            for row in range(self.GetNumberRows()):
                self.SetCellValue(row, col, styleNames[0])

    def increment(self, col, isLeft):
        if isLeft:
            for row in range(self.GetNumberRows()):
                try:
                    value = int(self.GetCellValue(row, col)) + 1
                    if value > self.upperBound:
                        value = self.upperBound
                    self.SetCellValue(row, col, str(value))
                except:
                    print('Could not convert "' + self.GetCellValue(row, col) +
                          '" to an integer.')
        else:
            for row in range(self.GetNumberRows()):
                try:
                    value = int(self.GetCellValue(row, col)) - 1
                    if value < self.lowerBound:
                        value = self.lowerBound
                    self.SetCellValue(row, col, str(value))
                except:
                    print('Could not convert "' + self.GetCellValue(row, col) +
                          '" to an integer.')

    def incrementAlpha(self, col, isLeft):
        if isLeft:
            for row in range(self.GetNumberRows()):
                try:
                    value = float(self.GetCellValue(row, col)) + 0.1
                    if value > 1.:
                        value = 1.
                    self.SetCellValue(row, col, str(value))
                except:
                    print('Could not convert "' + self.GetCellValue(row, col) +
                          '" to a float.')
        else:
            for row in range(self.GetNumberRows()):
                try:
                    value = float(self.GetCellValue(row, col)) - 0.1
                    if value < 0.:
                        value = 0.
                    self.SetCellValue(row, col, str(value))
                except:
                    print('Could not convert "' + self.GetCellValue(row, col) +
                          '" to a float.')

    def legendLabels(self, col, isLeft):
        if isLeft:
            for row in range(self.GetNumberRows()):
                self.SetCellValue(row, col, self.GetRowLabelValue(row))
        else:
            for row in range(self.GetNumberRows()):
                self.SetCellValue(row, col, '')
        
    def setIdentifiers(self, identifiers):
        if self.GetNumberRows() < len(identifiers):
            self.InsertRows(0, len(identifiers) - self.GetNumberRows())
        elif self.GetNumberRows() > len(identifiers):
            self.DeleteRows(0, self.GetNumberRows() - len(identifiers))
        for row in range(self.GetNumberRows()):
            self.SetRowLabelValue(row, identifiers[row])
            # Show. '' is False, '1' is True.
            self.SetCellValue(row, 0, '1')
            # Line Style.
            self.SetCellValue(row, 2, styleNames[0])
            # Line Width.
            self.SetCellValue(row, 3, str(self.lowerBound))
            # Filled in initially false.
            self.SetCellValue(row, 4, '')
            # Alpha.
            self.SetCellValue(row, 6, '1')
            # Legend labels.
            self.SetCellValue(row, 7, identifiers[row])
        self.AutoSizeColumns(False)
        self.Layout()
        # Color the lines by hue.
        self.lineColor(1, True)
        # Match the fill colors.
        # The marker face and marker edge are black.
        self.fillColor(5, True)

    def saveEditControlValue(self):
        """Save the values being edited.."""
        if self.IsCellEditControlShown(): 
            self.SaveEditControlValue() 
            self.HideCellEditControl() 

    def getCheckedItems(self):
        """Return the list of checked item indices."""
        return filter(lambda row: self.GetCellValue(row, 0),
                      range(self.GetNumberRows()))

    def getLineStyles(self, row):
        """Get the line styles. If no legend label is specified, use a single
        space."""
        return {'color':wxToMatplotlibColor
                (self.GetCellBackgroundColour(row, 1)),
                'linestyle':
                    styleStrings[styleNames.index(self.GetCellValue(row, 2))],
                'linewidth':int(self.GetCellValue(row, 3)),
                'label':self.GetCellValue(row, 7) or ' '}

    def getFillStyle(self, row):
        """Get the fill style for the specified row. Return the tuple:
        (isFilled, color, alpha)."""
        isFilled = bool(self.GetCellValue(row, 4))
        color = wxToMatplotlibColor(self.GetCellBackgroundColour(row, 5))
        try:
            alpha = float(self.GetCellValue(row, 6))
            if alpha < 0.:
                alpha = 0.
            if alpha > 1.:
                alpha = 1.
        except:
            alpha = 1.
        return (isFilled, color, alpha)

    def areAnyItemsSelected(self):
        return self.GetNumberRows() != 0 and\
               reduce(lambda x, y: x or y, [self.GetCellValue(row, 0) for row
                                            in range(self.GetNumberRows())])

if __name__ == '__main__':
    class Frame(wx.Frame):
        def __init__(self, parent=None):
            wx.Frame.__init__(self, parent, title='Plot', size=(600,300))
            sizer = wx.BoxSizer(wx.VERTICAL)
            grid = PlotHistogramGrid(self)
            grid.setIdentifiers(['a', 'b'])
            sizer.Add(grid, 1, wx.EXPAND)
            self.SetSizer(sizer)
            self.Fit()

    app = wx.PySimpleApp()
    Frame().Show()
    app.MainLoop()
