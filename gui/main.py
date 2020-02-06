from tkinter import *
import pandas as pd
import numpy as np
from itertools import compress


class GUI:

    def __init__(self):
        self.loginPage()

    def loginPage(self):
        self.login = Tk()
        self.login.title("Click UPLOAD to begin")
        self.login.config(bg="white")

        frame = Frame(self.login)

        upload_btn = Button(frame, text="UPLOAD", width=8, command=self.getfilename)
        upload_btn.grid(column=1, row=3, sticky=W)

        quit_btn = Button(frame, text="QUIT", width=8, command=self.login.destroy)
        quit_btn.grid(column=2, row=3, sticky=W)

        frame.grid(row=3, column=2)

        self.login.mainloop()

    def getfilename(self):
        from tkinter.filedialog import askopenfilename
        filename = askopenfilename()

        if not filename:
            return

        msg = self.checkfile()

        if msg:
            return

        keep = ['qty_6mos',
                'cogs_6mos',
                'sales_6_mos',
                'picks_6mos',
                'margin_%',
                'net_OH',
                'net_OH_usd',
                'pallet_quantity',
                'item_poi_days',
                'dioh']

        self.field_options = ['turn_and_earn', 'revenue_6mos'] + [x for x in self.df.columns if x in keep]

        wh_options = self.df.legacy_division_cd.unique().astype(str)

        if wh_options is None:
            return

        wh_options = np.append(['All'], wh_options)

        self.login.withdraw()
        verify = Toplevel()
        self.verify = verify
        verify.title("Verify inputs")
        verify.config(bg="white")
        frame = Frame(verify)

        label = Label(frame, text="Modify the following inputs and select RUN. ")
        label.grid(column=0, row=0)

        run_btn = Button(frame, text="RUN", width=8, command=self.formatdata)
        run_btn.grid(column=1, row=0)

        wh_label = Label(frame, text="Select warehouse(s) to analyze: ")
        wh_label.grid(column=0, row=1)

        self.wh_var = StringVar(self.login, value='All')
        wh = OptionMenu(frame, self.wh_var, *wh_options)
        wh.grid(column=1, row=1)

        field_label = Label(frame, text="Select fields(s) to include in analysis: ")
        field_label.grid(column=0, row=2)

        self.field_var = []  # .set('All')
        for idx in range(len(self.field_options)):
            if idx in [0, 1]:
                var = IntVar(self.login, value=1)
            else:
                var = IntVar(self.login)

            Checkbutton(frame, text=self.field_options[idx], variable=var).grid(row=2, column=idx + 1)
            self.field_var.append(var)

        frame.grid(row=3, column=len(self.field_options))

        verify.mainloop()

    def formatdata(self):
        self.verify.withdraw()
        outputs = Toplevel()
        self.outputs = outputs
        outputs.title("Outputs")
        outputs.config(bg="white")
        frame = Frame(outputs)

        label = Label(frame, text="Success!").grid(row=1, column=1)

        frame.grid(row=2, column=2)

        outputs.mainloop()

    def checkfile(self):
        try:
            with open('input_data/path.txt') as f:
                fname = f.read()
        except FileNotFoundError:
            return 'File not found. Select another file.'

        ext = fname.split('.')[1]
        if ext == 'xlsx':
            try:
                self.df = pd.read_excel(fname, sheet_name='Sheet1')
            except:  # TODO - run test on xlsx w/o multiple sheets and catch error
                self.df = pd.read_excel(fname)
        elif ext == '.csv':
            self.df = pd.read_csv(fname)
        else:
            return 'File extension not valid. Select another file.'

        self.df.columns = [c.replace(' ', '_').lower() for c in self.df.columns]

        try:
            self.df = self.df[self.df['sales_channel'] == 'Warehouse']
        except KeyError:
            pass


GUI()
