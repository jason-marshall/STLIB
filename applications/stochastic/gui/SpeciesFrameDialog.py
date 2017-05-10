"""Select a species and a frame with a dialog."""

import wx

class SpeciesFrameDialog(wx.Dialog):
    """Select a species and a frame with a dialog."""
    
    def __init__(self, parent, model, output):
	"""Construct."""
	wx.Dialog.__init__(self, parent, -1, 'Select a histogram.')

        # Species.
	selectionSizer = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(self, -1, 'Species:')
        selectionSizer.Add(label, 0, wx.ALL, 5)
	self.species = wx.Choice(self,
                                 choices=[model.speciesIdentifiers[_i]
                                          for _i in output.recordedSpecies])
	self.species.SetSelection(0)
        selectionSizer.Add(self.species, 0, wx.ALL, 5)

        # Frame.
        label = wx.StaticText(self, -1, 'Frame:')
        selectionSizer.Add(label, 0, wx.ALL, 5)
	self.frame = wx.Choice(self, choices=[str(_t) for _t in
                                              output.frameTimes])
	self.frame.SetSelection(0)
        selectionSizer.Add(self.frame, 0, wx.ALL, 5)

        # Buttons.
	okButton = wx.Button(self, wx.ID_OK, 'OK')
        okButton.SetDefault()
	cancelButton = wx.Button(self, wx.ID_CANCEL, 'Cancel')
	buttons = wx.BoxSizer(wx.HORIZONTAL)
        buttons.Add(okButton, 0, wx.ALIGN_RIGHT, 5)
        buttons.Add(cancelButton, 0, wx.ALIGN_RIGHT, 5)

	sizer = wx.BoxSizer(wx.VERTICAL)
	sizer.Add(selectionSizer, 0, wx.EXPAND | wx.ALL, border=5)
	sizer.Add(buttons, 0, wx.ALIGN_RIGHT | wx.ALIGN_BOTTOM, 5)

	self.SetSizer(sizer)
	self.Layout()
	self.Fit()

    def getSpecies(self):
        return self.species.GetCurrentSelection()

    def getFrame(self):
        return self.frame.GetCurrentSelection()

def main():
    import sys
    sys.path.insert(1, '..')
    from state.Model import Model
    from state.HistogramFrames import HistogramFrames

    model = Model()
    model.speciesIdentifiers = ['s1', 's2']

    hf = HistogramFrames(10, [0, 1])
    hf.setFrameTimes([0., 1., 2.])

    app = wx.PySimpleApp()
    frame = SpeciesFrameDialog(None, model, hf)
    result = frame.ShowModal()
    if result == wx.ID_OK:
        print 'OK'
        print frame.getSpecies()
        print frame.getFrame()
    else:
        print 'Cancel'
    frame.Destroy()
    app.MainLoop()

if __name__ == '__main__':
    main()



