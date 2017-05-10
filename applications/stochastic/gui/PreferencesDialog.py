"""Implements the preferences dialog."""

import wx

class PreferencesDialog(wx.Dialog):
    """The preferences dialog."""
    
    def __init__(self, preferences, parent=None):
	"""."""
	wx.Dialog.__init__(self, parent, -1, 'Cain Preferences.')

        # Store a reference so we can reset the data.
        self.preferences = preferences

	panels = wx.BoxSizer(wx.VERTICAL)
        # The text and choice controls.
        self.controls = {}
        # For each category.
        for subject in preferences.default:
            controlList = {}
            panel = wx.BoxSizer(wx.VERTICAL)
            # Header. The first element is the category name.
            category = subject[0]
            header = wx.StaticText(self, -1, category)
            panel.Add(header, 0, wx.ALL, 5)
            panel.Add(wx.StaticLine(self), 0, wx.EXPAND|wx.ALL, 5)
            # The fields.
            entries = wx.FlexGridSizer(len(subject) - 1, 2, hgap=10)
            for (field, value, width) in subject[1:]:
                entries.Add(wx.StaticText(self, -1, field), border=0)
                # If the value field is a list.
                if type(value) == type([]):
                    # Add a choice control.
                    # Note: The choice control widget has an inappropriately
                    # large size. Use short strings to mitigate this problem.
                    control = wx.Choice(self, choices=value)
                    control.SetSelection(0)
                else:
                    # Add a text control.
                    control = wx.TextCtrl(self, -1, value, size=(width, -1))
                entries.Add(control, border=0)
                controlList[field] = control
            panel.Add(entries, 0, wx.ALL, 5)
            panels.Add(panel, 0, border=5)
            self.controls[category] = controlList

        # Buttons.
	okButton = wx.Button(self, wx.ID_OK, 'OK')
        okButton.SetDefault()
	cancelButton = wx.Button(self, wx.ID_CANCEL, 'Cancel')
        restoreButton = wx.Button(self, -1, 'Restore Defaults')
        self.Bind(wx.EVT_BUTTON, self.onRestoreDefaults, restoreButton)
	buttons = wx.BoxSizer(wx.HORIZONTAL)
        buttons.Add(okButton, 0, wx.ALIGN_RIGHT, 5)
        buttons.Add(cancelButton, 0, wx.ALIGN_RIGHT, 5)
        buttons.Add(restoreButton, 0, wx.ALIGN_RIGHT, 5)

	sizer = wx.BoxSizer(wx.VERTICAL)
	sizer.Add(panels, 0, wx.EXPAND | wx.ALL, border=5)
	sizer.Add(buttons, 0, wx.ALIGN_RIGHT | wx.ALIGN_BOTTOM, 5)

	self.SetSizer(sizer)
	self.Layout()
	self.Fit()

    def onRestoreDefaults(self, event):
        self.reset()
        for subject in self.preferences.default:
            category = subject[0]
            for (field, value, width) in subject[1:]:
                if type(value) == type([]):
                    self.controls[category][field].SetSelection(0)
                else:
                    self.controls[category][field].SetValue(value)

    def reset(self):
        self.preferences.reset()

def main():
    from Preferences import Preferences
    app = wx.PySimpleApp()
    preferences = Preferences()
    frame = PreferencesDialog(preferences)
    result = frame.ShowModal()
    if result == wx.ID_OK:
        print 'OK'
        for subject in preferences.default:
            category = subject[0]
            print category
            for (field, value, width) in subject[1:]:
                if type(value) == type([]):
                    print '  ', field,\
                        frame.controls[category][field].GetSelection()
                else:
                    print '  ', field,\
                        frame.controls[category][field].GetValue()
    else:
        print 'Cancel'
    frame.Destroy()
    app.MainLoop()

if __name__ == '__main__':
    main()
