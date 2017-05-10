"""Implements the duplicate dialog."""

import wx

class DuplicateDialog(wx.Dialog):
    """The duplicate dialog."""
    
    def __init__(self, parent=None):
	"""."""
	wx.Dialog.__init__(self, parent, -1, 'Duplicate the model.')

        # Multiplicity.
	multiplicitySizer = wx.BoxSizer(wx.HORIZONTAL)
        self.multiplicity = wx.SpinCtrl(self, -1, value='2', min=2, max=1000000,
                                   initial=2)
        multiplicitySizer.Add(self.multiplicity, 0, wx.ALL, 5)
        label = wx.StaticText(self, -1, 'Multiplicity.')
        multiplicitySizer.Add(label, 0, wx.ALL, 5)

        # Randomly scale propensities.
        self.scale = wx.CheckBox(self, -1, 'Randomly scale propensities.')

        # Buttons.
	okButton = wx.Button(self, wx.ID_OK, 'OK')
        okButton.SetDefault()
	cancelButton = wx.Button(self, wx.ID_CANCEL, 'Cancel')
	buttons = wx.BoxSizer(wx.HORIZONTAL)
        buttons.Add(okButton, 0, wx.ALIGN_RIGHT, 5)
        buttons.Add(cancelButton, 0, wx.ALIGN_RIGHT, 5)

	sizer = wx.BoxSizer(wx.VERTICAL)
	sizer.Add(multiplicitySizer, 0, wx.EXPAND | wx.ALL, border=5)
	sizer.Add(self.scale, 0, wx.EXPAND | wx.ALL, border=5)
	sizer.Add(buttons, 0, wx.ALIGN_RIGHT | wx.ALIGN_BOTTOM, 5)

	self.SetSizer(sizer)
	self.Layout()
	self.Fit()

    def getMultiplicity(self):
        return self.multiplicity.GetValue()

    def useScaling(self):
        return self.scale.GetValue()

def main():
    app = wx.PySimpleApp()
    frame = DuplicateDialog()
    result = frame.ShowModal()
    if result == wx.ID_OK:
        print 'OK'
        print frame.getMultiplicity()
        print frame.useScaling()
    else:
        print 'Cancel'
    frame.Destroy()
    app.MainLoop()

if __name__ == '__main__':
    main()
