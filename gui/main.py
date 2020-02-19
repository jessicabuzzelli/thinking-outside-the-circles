from tkinter import *
import pandas as pd
import numpy as np
from gui.scripts.vectorize import WarehouseLevelVectors, RegionLevelVectors


class GUI:

    def __init__(self):
        self.home()

    def home(self):
        self.root = Tk()
        self.root.title("Click UPLOAD to begin")
        self.root.config(bg="white")

        frame = Frame(self.root)

        Button(frame, text="UPLOAD", width=8, command=self.get_filename).grid(column=1, row=3, sticky=W)
        Button(frame, text="QUIT", width=8, command=self.root.destroy).grid(column=2, row=3, sticky=W)

        frame.grid(row=3, column=2)

        self.root.mainloop()

    def get_filename(self):
        self.root.withdraw()

        from tkinter.filedialog import askopenfilename
        self.fname = askopenfilename()

        if self.fname is None:
            return

        loading = Toplevel()
        self.loading = loading
        loading.title("Loading...")
        loading.config(bg="white")
        frame = Frame(loading)

        Label(frame, text='Loading...').grid(row=2, column=2, pady=100, padx=100)
        frame.grid(row=4, column=4)

        self.loading.after(200, self.check_read)
        loading.mainloop()

    def check_read(self):
        failed_read = self.read_file()
        if failed_read is True:
            self.get_filename()
        else:
            failed_set_var = self.set_vars()
            if failed_set_var is True:
                self.get_filename()
            else:
                self.input_page()

    def read_file(self):
        ext = self.fname.split('.')[1]
        if ext == 'xlsx':
            try:
                self.df = pd.read_excel(self.fname, sheet_name='Sheet1')
            except:
                self.df = pd.read_excel(self.fname)
        elif ext == '.csv':
            self.df = pd.read_csv(self.fname)
        else:
            return 'File extension not valid. Select another file.'

        self.df.columns = [c.replace(' ', '_').lower() for c in self.df.columns]

        try:
            self.df = self.df[self.df['sales_channel'] == 'Warehouse']
        except KeyError:
            pass

        keep = ['legacy_division_cd',
                'legacy_system_cd',
                'segment',
                'legacy_product_cd',
                'sales_channel',
                'sales_6_mos',
                'cogs_6mos',
                'qty_6mos',
                'picks_6mos',
                'net_oh',
                'legacy_customer_cd',
                'core_item_flag',
                'margin_%',
                'net_oh_$',
                'dioh']

        self.df = self.df[keep].fillna(0).replace('-', 0)

    def set_vars(self):
        options = ['sales_6_mos',
                   'cogs_6mos',
                   'qty_6mos',
                   'picks_6mos',
                   'net_oh',
                   'pallet_quantity',
                   'margin_%']

        self.field_options = ['turn_and_earn', 'profit_6mos', 'customers_per_product'] + \
                             [x for x in self.df.columns if x in options]

        wh_options = self.df.legacy_division_cd.unique().astype(str)

        if wh_options is None:
            return 'no warehouses'

        self.segment_options = self.df.segment.unique()
        if self.segment_options is None:
            return 'no segments'

        self.region_options = self.df['legacy_system_cd'].unique()
        if self.region_options is None:
            return 'no regions'

        self.level_options = ['warehouse', 'region']  # TODO - check to see if add hub
        self.region_options = np.append(['All'], self.region_options)
        self.wh_options = np.append(['All'], wh_options)

        self.wh_var = StringVar(self.root, value='All')
        self.segment_var = StringVar(self.root, value=self.segment_options[0])
        self.cutoff_var = StringVar(self.root, value='20')
        self.field_var = []
        self.region_var = StringVar(self.root, value='All')
        self.level_var = StringVar(self.root)

    def input_page(self):
        self.loading.withdraw()

        define = Toplevel()
        self.define = define
        define.title("Define inputs")
        define.config(bg="white")
        frame = Frame(define)
        self.input_frame = frame

        Label(frame, text="Modify model inputs below and click RUN. ").grid(row=0, column=1, pady=10)
        Button(frame, text="RUN", width=8, command=self.check_inputs).grid(row=0, column=2, pady=10)

        Label(frame, text="Select segment: ").grid(row=1, column=0, pady=10)
        OptionMenu(frame, self.segment_var, *self.segment_options).grid(row=1, column=1, pady=10)

        Label(frame, text="Select scope and press REFRESH: ").grid(row=2, column=0, pady=10)
        OptionMenu(frame, self.level_var, *self.level_options).grid(row=2, column=1, pady=10)
        Button(frame, text='REFRESH', command=self.popup_options).grid(row=2, column=2, pady=10)

        Label(frame, text=" % core products: ").grid(row=3, column=0, pady=10)
        Entry(frame, textvariable=self.cutoff_var).grid(row=3, column=1, pady=10)

        Label(frame, text="Field(s) to consider: ").grid(row=4, column=0, pady=10)

        pad = len(max(self.field_options, key=len))
        for idx in range(len(self.field_options)):
            if idx in [0, 1, 2]:
                var = IntVar(self.root, value=1)
            else:
                var = IntVar(self.root)

            txt = self.field_options[idx]
            b = Checkbutton(frame, text=txt + '  ' * (pad - len(txt)), variable=var, anchor="w")
            b.grid(row=4 + (idx // 2), column=(idx % 2) + 1, pady=10)
            self.field_var.append(var)

        self.last_row = 3 + (len(self.field_options) // 2)

        frame.grid(row=4, column=len(self.field_options))

        define.mainloop()

    def popup_options(self):
        try:
            self.wh_label.destroy()
            self.wh_optionmenu.destroy()
        except AttributeError:
            pass

        try:
            self.region_label.destroy()
            self.region_optionmenu.destroy()
        except AttributeError:
            pass

        if self.level_var.get() == 'warehouse':
            self.wh_label = Label(self.input_frame, text="Select warehouse(s): ")
            self.wh_label.grid(row=2, column=3, pady=10)
            self.wh_optionmenu = OptionMenu(self.input_frame, self.wh_var, *self.wh_options)
            self.wh_optionmenu.grid(row=2, column=4, pady=10)

        elif self.level_var.get() == 'region':
            self.region_label = Label(self.input_frame, text="Select region(s): ")
            self.region_label.grid(row=2, column=3, pady=10)
            self.region_optionmenu = OptionMenu(self.input_frame, self.region_var, *self.region_options)
            self.region_optionmenu.grid(row=2, column=4, pady=10)

    def check_inputs(self):
        err = False
        try:
            float(self.cutoff_var.get())

        except ValueError:
            err = True

        if err:
            Label(self.input_frame, text='ERROR. Enter a numeric value for % core products.') \
                .grid(row=self.last_row + 1, column=0)
            return

        else:
            self.format_vars()
            self.loading_page2()

    def format_vars(self):
        field_var = [x.get() for x in self.field_var]

        cutoff_var = self.cutoff_var.get()
        segment_var = self.segment_var.get()
        wh_var = self.wh_var.get()
        region_var = self.region_var.get()
        level_var = self.level_var.get()

        try:
            wh_var = [int(wh_var)]
            # print(self.wh_var)

        except ValueError:
            assert wh_var == 'All'
            wh_var = self.df['legacy_division_cd'].unique()

        params = [segment_var, field_var, self.field_options, cutoff_var, self.df, self.fname]

        if level_var == 'warehouse':
            self.model = WarehouseLevelVectors(wh_var, *params)

        elif level_var == 'region':
            self.model = RegionLevelVectors(region_var, *params)

    def loading_page2(self):
        loading2 = Toplevel()
        self.loading2 = loading2
        loading2.title("Loading...")
        loading2.config(bg="white")
        frame = Frame(loading2)

        Label(frame, text='Loading...').grid(row=2, column=2, pady=100, padx=100)
        frame.grid(row=4, column=4)

        self.loading2.after(200, self.move_on)
        loading2.mainloop()

    def move_on(self):
        self.model.run()
        self.output_page()

    def output_page(self):
        self.loading2.withdraw()

        outputs = Toplevel()
        self.outputs = outputs
        outputs.title("Outputs")
        outputs.config(bg="white")
        frame = Frame(outputs)

        # print(self.field_options)
        # print(self.field_var)

        Label(frame, text="Success!").grid(row=0, column=0, pady=10)
        Button(frame, text="Export to Excel", command=self.model.export).grid(row=1, column=0, pady=10)

        text = Text(frame)
        text.grid(row=2, column=0, pady=0)
        text.insert(INSERT, self.model.string_output())

        frame.grid(row=2, column=2)

        outputs.mainloop()


if __name__ == "__main__":
    GUI()