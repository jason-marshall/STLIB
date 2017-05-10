"""Grid of the items (species or reactions) to plot."""

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
# Extracted from matplotlib/lines.py.
markers =  (
    ('', 'No marker'),
    ('.', 'Point'),
    (',', 'Pixel'),
    ('o', 'Circle'),
    ('v', 'Triangle down'),
    ('^', 'Triangle up'),
    ('<', 'Triangle left'),
    ('>', 'Triangle right'),
    ('1', 'Tripod down'),
    ('2', 'Tripod up'),
    ('3', 'Tripod left'),
    ('4', 'Tripod right'),
    ('s', 'Square'),
    ('p', 'Pentagon'),
    ('h', 'Hexagon1'),
    ('H', 'Hexagon2'),
    ('+', 'Plus'),
    ('x', 'x'),
    ('D', 'Diamond'),
    ('d', 'Thin diamond'),
    ('|', 'Vertical line'),
    ('_', 'Horizontal line')
    )
markerNames = [x[1] for x in list(markers)]
markerStrings = [x[0] for x in list(markers)]

def wxToMatplotlibColor(color):
    return (color.Red() / 255.0, color.Green() / 255.0, color.Blue() / 255.0)

class PlotTimeSeriesGrid(wx.grid.Grid):
    def __init__(self, parent):
        wx.grid.Grid.__init__(self, parent)
        columnLabels = ['Show', 'Std\nDev', 'Line\nColor', 'Style', 'Width',
                        'Marker\nStyle', 'Size', 'Face\nColor', 'Edge\nColor',
                        'Edge\nWidth', 'Legend\nLabel']
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
        # The Std. Dev. column may be hidden.
        self.SetColMinimalWidth(1, 0)

        # Lower and upper bounds for sizes and widths.
        self.lowerBound = 1
        self.upperBound = 10

        # Renderers and editors.
        boolRenderer = wx.grid.GridCellBoolRenderer()
        boolEditor = wx.grid.GridCellBoolEditor()
        lineStyleEditor = wx.grid.GridCellChoiceEditor(styleNames)
        markerStyleEditor = wx.grid.GridCellChoiceEditor(markerNames)
        numberEditor = wx.grid.GridCellNumberEditor(self.lowerBound,
                                                    self.upperBound)
        
        #
        # Grid column attributes.
        #
        # Show and Std Dev.
        attr = wx.grid.GridCellAttr()
        attr.SetRenderer(boolRenderer)
        attr.SetEditor(boolEditor)
        for col in range(2):
            self.SetColAttr(col, attr)
        # Line Color.
        attr = wx.grid.GridCellAttr()
        attr.SetReadOnly()
        self.SetColAttr(2, attr)
        # Line Style.
        attr = wx.grid.GridCellAttr()
        attr.SetEditor(lineStyleEditor)
        self.SetColAttr(3, attr)
        # Line Width.
        attr = wx.grid.GridCellAttr()
        attr.SetEditor(numberEditor)
        self.SetColAttr(4, attr)
        # Marker Style.
        attr = wx.grid.GridCellAttr()
        attr.SetEditor(markerStyleEditor)
        self.SetColAttr(5, attr)
        # Marker Size. The default size is 5.
        attr = wx.grid.GridCellAttr()
        attr.SetEditor(numberEditor)
        self.SetColAttr(6, attr)
        # Marker Face Color.
        attr = wx.grid.GridCellAttr()
        attr.SetReadOnly()
        self.SetColAttr(7, attr)
        # Marker Edge Color.
        attr = wx.grid.GridCellAttr()
        attr.SetReadOnly()
        self.SetColAttr(8, attr)
        # Marker Edge Width.
        attr = wx.grid.GridCellAttr()
        attr.SetEditor(numberEditor)
        self.SetColAttr(9, attr)
        # Use the default for the legend labels.

        # Events.
        self.Bind(wx.grid.EVT_GRID_CELL_LEFT_CLICK, self.onCellLeftClick)
        self.Bind(wx.grid.EVT_GRID_LABEL_LEFT_CLICK, self.onLabelLeftClick)
        self.Bind(wx.grid.EVT_GRID_LABEL_RIGHT_CLICK, self.onLabelRightClick)
        self.columnLabelFunctions = [self.select, self.select, self.lineColor,
                                     self.lineStyle, self.increment,
                                     self.markerStyle, self.increment,
                                     self.matchColor, self.matchColor,
                                     self.increment, self.legendLabels]

    def onCellLeftClick(self, event):
        row, col = event.GetRow(), event.GetCol()
        if col in (2, 7, 8):
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

    def lineColor(self, col, isLeft):
        black = wx.Colour(0, 0, 0)
        if isLeft:
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
        else:
            for row in range(self.GetNumberRows()):
                self.SetCellBackgroundColour(row, col, black)

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

    def markerStyle(self, col, isLeft):
        if isLeft:
            # Cycle between the available marker styles.
            n = 0
            for row in range(self.GetNumberRows()):
                if self.GetCellValue(row, 0):
                    # Note that the first marker style is "No marker" and the
                    # second and third (point and pixel) don't scale with the
                    # marker size.
                    # We skip these choices.
                    self.SetCellValue(row, col, markerNames[n + 3])
                    n = (n + 1) % (len(markerNames) - 3)
        else:
            # No markers.
            for row in range(self.GetNumberRows()):
                self.SetCellValue(row, col, markerNames[0])

    def matchColor(self, col, isLeft):
        if isLeft:
            # Match the line colors.
            for row in range(self.GetNumberRows()):
                self.SetCellBackgroundColour\
                    (row, col, self.GetCellBackgroundColour(row, 2))
        else:
            # Set the color to black.
            color = wx.Colour(0, 0, 0)
            for row in range(self.GetNumberRows()):
                self.SetCellBackgroundColour(row, col, color)

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
            # Show and Std Dev.
            for col in range(2):
                # '' is False, '1' is True.
                self.SetCellValue(row, col, '1')
            # Line Style.
            self.SetCellValue(row, 3, styleNames[0])
            # Line Width.
            self.SetCellValue(row, 4, str(self.lowerBound))
            # Marker Style.
            self.SetCellValue(row, 5, markerNames[0])
            # Marker Size. The default size is 5.
            self.SetCellValue(row, 6, '5')
            # Marker Edge Width.
            self.SetCellValue(row, 9, str(self.lowerBound))
            # Legend labels.
            self.SetCellValue(row, 10, identifiers[row])
        self.AutoSizeColumns(False)
        self.Layout()
        # Color the lines by hue.
        self.lineColor(2, True)
        # The marker face and marker edge are black.
        self.matchColor(7, False)
        self.matchColor(8, False)

    def saveEditControlValue(self):
        """Save the values being edited.."""
        if self.IsCellEditControlShown(): 
            self.SaveEditControlValue() 
            self.HideCellEditControl() 

    def showStdDev(self):
        self.AutoSizeColumn(1, False)
        self.ForceRefresh()

    def hideStdDev(self):
        self.SetColSize(1, 0)
        self.ForceRefresh()

    def getCheckedItems(self):
        """Return the list of checked item indices."""
        return filter(lambda row: self.GetCellValue(row, 0),
                      range(self.GetNumberRows()))

    def useMarkers(self, row):
        """Return True if the specified item should be plotted with markers."""
        return self.GetCellValue(row, 5) != markerNames[0]

    def getLegendLabel(self, row):
        """If no legend label is specified, use a single space."""
        return self.GetCellValue(row, 10) or ' '
    
    def getLineStyles(self, row):
        """Get the line styles."""
        return {'color':wxToMatplotlibColor
                (self.GetCellBackgroundColour(row, 2)),
                'linestyle':
                    styleStrings[styleNames.index(self.GetCellValue(row, 3))],
                'linewidth':int(self.GetCellValue(row, 4))}

    def getLineAndMarkerStyles(self, row):
        """Get the line and marker styles."""
        styles = self.getLineStyles(row)
        styles.update(
            {'marker':
                 markerStrings[markerNames.index(self.GetCellValue(row, 5))],
             'markersize':int(self.GetCellValue(row, 6)),
             'markerfacecolor':
                 wxToMatplotlibColor(self.GetCellBackgroundColour(row, 7)),
             'markeredgecolor':
                 wxToMatplotlibColor(self.GetCellBackgroundColour(row, 8)),
             'markeredgewidth':int(self.GetCellValue(row, 9))})
        return styles

    def areAnyItemsSelected(self):
        return self.GetNumberRows() != 0 and\
               reduce(lambda x, y: x or y, [self.GetCellValue(row, 0) for row
                                            in range(self.GetNumberRows())])

if __name__ == '__main__':
    class Frame(wx.Frame):
        def __init__(self, parent=None):
            wx.Frame.__init__(self, parent, title='Plot', size=(600,300))
            sizer = wx.BoxSizer(wx.VERTICAL)
            grid = PlotTimeSeriesGrid(self)
            grid.setIdentifiers(['a', 'b'])
            sizer.Add(grid, 1, wx.EXPAND)
            self.SetSizer(sizer)
            self.Fit()

    app = wx.PySimpleApp()
    Frame().Show()
    app.MainLoop()
