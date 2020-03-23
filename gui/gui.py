from tkinter import *
import pandas as pd
import numpy as np
from itertools import compress
# from gui.scripts.vectorize import Vectorize
from vectorize import Vectorize

class GUI:

    def __init__(self):
        self.root = Tk()
        self.home()

    def home(self):
        # self.root.eval('tk::PlaceWindow %s center' % self.root.winfo_pathname(self.root.winfo_id()))
        self.root.title("Click UPLOAD to begin")
        self.root.config(bg="white")

        frame = Frame(self.root)

        Button(frame, text="UPLOAD", width=8, command=self.get_filename).grid(column=1, row=3, sticky=W)
        Button(frame, text="QUIT", width=8, command=self.root.destroy).grid(column=2, row=3, sticky=W)

        frame.grid(row=3, column=2)

        self.root.mainloop()

    def get_filename(self):
        try:
            self.root.withdraw()
            self.loading.withdraw()

        except:
            pass

        from tkinter.filedialog import askopenfilename
        self.fname = askopenfilename()

        if self.fname is tuple():
            self.home()
        else:
            self.loading1()

    def loading1(self):
        loading = Toplevel()
        self.loading = loading
        # self.root.eval('tk::PlaceWindow %s center' % self.loading.winfo_pathname(self.loading.winfo_id()))
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
        self.ext = self.fname.split('.')[1]
        if self.ext == 'xlsx':
            try:
                self.df = pd.read_excel(self.fname, sheet_name='Sheet1')
            except:
                self.df = pd.read_excel(self.fname)
        elif self.ext == '.csv':
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
                'dioh',
                'national_acct_flag']

        self.df = self.df[keep].fillna(0).replace('-', 0)

    def set_vars(self):
        options = ['sales_6_mos',
                   'cogs_6mos',
                   'qty_6mos',
                   'picks_6mos',
                   'net_oh_$_6mos',
                   # 'pallet_quantity',
                   'margin_%']

        try:
            self.field_options = ['turn_6mos', 'profit_6mos', 'customers_per_product'] + \
                                 [x for x in self.df.columns if x in options]
        except AttributeError:
            self.loading.withdraw()
            self.get_filename()

        wh_options = self.df.legacy_division_cd.unique().astype(str)

        if wh_options is None:
            return 'no warehouses'

        self.segment_options = self.df.segment.unique()
        if self.segment_options is None:
            return 'no segments'

        self.region_options = self.df['legacy_system_cd'].unique()
        if self.region_options is None:
            return 'no regions'

        self.level_options = ['warehouse', 'region', 'enterprise']  # TODO - check to see if add hub
        self.region_options = np.append(['All'], self.region_options)
        self.wh_options = np.append(['All'], wh_options)

        self.wh_var = StringVar(self.root, value='All')
        self.segment_var = StringVar(self.root, value=self.segment_options[0])
        self.cutoff_var = StringVar(self.root, value='20')
        self.field_var = []
        self.weight_var = [StringVar(self.root, value='33.3'),
                           StringVar(self.root, value='33.3'),
                           StringVar(self.root, value='33.3')]
        self.region_var = StringVar(self.root, value='All')
        self.level_var = StringVar(self.root, value='warehouse')
        self.objective = StringVar(self.root, value='Identify core products')
        self.natl_acct = IntVar(self.root, value=1)

    def input_page(self):
        self.loading.withdraw()

        try:
            # if tool has already done one analysis and needs to do another
            self.outputs.withdraw()

        except:
            pass

        define = Toplevel()
        self.define = define
        # self.root.eval('tk::PlaceWindow %s center' % self.define.winfo_pathname(self.define.winfo_id()))
        define.title("Define inputs")
        define.config(bg="white")
        frame = Frame(define)
        self.input_frame = frame

        Label(frame, text="Modify model inputs below and click RUN. ").grid(row=0, column=1, pady=10)
        Button(frame, text="RUN", width=8, command=self.check_inputs).grid(row=0, column=2, pady=10)

        Label(frame, text="Select model goal: ").grid(row=1, column=0, pady=10)
        OptionMenu(frame, self.objective, 'Identify core products', 'Identify products to remove')\
            .grid(row=1, column=1, pady=10)

        Label(frame, text="Select segment: ").grid(row=2, column=0, pady=10)
        OptionMenu(frame, self.segment_var, *self.segment_options).grid(row=2, column=1, pady=10)

        Label(frame, text="Exclude products ordered by national account(s)? ").grid(row=3, column=0, pady=10)
        Checkbutton(frame, text='', variable=self.natl_acct, justify=LEFT, anchor="w").grid(row=3, column=1)

        Label(frame, text="Select scope level and press REFRESH: ").grid(row=4, column=0, pady=10)
        OptionMenu(frame, self.level_var, *self.level_options).grid(row=4, column=1, pady=10)
        Button(frame, text='REFRESH', command=self.popup_level_options).grid(row=4, column=2, pady=10)


        Label(frame, text="Set % to identify: ").grid(row=5, column=0, pady=10)
        Entry(frame, textvariable=self.cutoff_var).grid(row=5, column=1, pady=10)

        Label(frame, text="Select field(s) to consider and enter weights: ").grid(row=6, column=0, pady=10)

        self.btns = []
        self.entries = []
        self.rows = []
        preselected = list(compress(list(range(len(self.field_options))), [float(x.get()) for x in self.weight_var]))

        for idx in range(len(self.field_options)):
            if idx in preselected:
                var = IntVar(self.root, value=1)
                weightvar = StringVar(self.root, value=33.33)
            else:
                var = IntVar(self.root)
                weightvar = IntVar(self.root, value=0)
                self.weight_var.append(weightvar)

            self.field_var.append(var)

            txt = self.field_options[idx]

            btn = Checkbutton(frame, text=txt, variable=self.field_var[idx], justify=LEFT, anchor="w")
            btn.grid(row=6 + idx, column=1, pady=10)

            entry = Entry(frame, text=self.weight_var[idx].get(), textvariable=self.weight_var[idx])
            entry.grid(row=6 + idx, column=2, pady=10, padx=10)

            self.btns += [btn]
            self.entries += [entry]
            self.rows += [4 + idx]

        self.last_row = 4 + len(self.field_options)

        Button(frame, text="Set equal weights among checked fields", wraplength=150, command=self.reset_weights)\
            .grid(row=6, column=3, pady=10)

        frame.grid(row=4, column=len(self.field_options))

        define.mainloop()

    def popup_level_options(self):
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
            self.wh_label.grid(row=4, column=3, pady=10, padx=10)
            self.wh_optionmenu = OptionMenu(self.input_frame, self.wh_var, *self.wh_options)
            self.wh_optionmenu.grid(row=4, column=4, pady=10)

        elif self.level_var.get() == 'region':
            self.region_label = Label(self.input_frame, text="Select region(s): ")
            self.region_label.grid(row=4, column=3, pady=10)
            self.region_optionmenu = OptionMenu(self.input_frame, self.region_var, *self.region_options)
            self.region_optionmenu.grid(row=4, column=4, pady=10)

    def reset_weights(self):
        total = sum([x.get() for x in self.field_var])
        for x in range(len(self.field_var)):
            if self.field_var[x].get() == 1:
                self.entries[x].destroy()
                self.weight_var[x] = StringVar(self.root, value=round(100 / total, 2))
                self.entries[x] = Entry(self.input_frame, text=self.weight_var[x].get(), textvariable=self.weight_var[x])
                self.entries[x].grid(row=self.rows[x], column=2, pady=10, padx=10)
            else:
                self.entries[x].destroy()
                self.weight_var[x] = StringVar(self.root, value=0)
                self.entries[x] = Entry(self.input_frame, text=self.weight_var[x].get(),
                                        textvariable=self.weight_var[x])
                self.entries[x].grid(row=self.rows[x], column=2, pady=10, padx=10)

    def check_inputs(self):
        try:
            self.ErrorLabel.destroy()

        except AttributeError:
            pass

        err = ''

        try:
            self.cutoff = float(self.cutoff_var.get())

        except ValueError:
            err = 'ERROR. Enter a numeric value for % core products.'

        try:
            self.weights = [float(w.get()) for w in self.weight_var]
            total = sum(self.weights)
            assert 99 <= total <= 101

        except TclError:
            err = 'ERROR. Enter a numeric value for each of the field weights.'

        except AssertionError:
            err = 'ERROR. Sum of field weights must equal 100.'

        if err:
            self.ErrorLabel = Label(self.input_frame, text=err).grid(row=self.last_row + 1, column=0)
            return

        else:
            self.format_vars()
            self.loading_page2()

    def format_vars(self):
        field_var = [x.get() for x in self.field_var]

        segment_var = self.segment_var.get()
        wh_var = self.wh_var.get()
        region_var = self.region_var.get()
        level_var = self.level_var.get()
        obj_var = self.objective.get()
        natl_acct_var = self.natl_acct.get()

        try:
            wh_var = [int(wh_var)]

        except ValueError:
            assert wh_var == 'All'
            wh_var = self.df['legacy_division_cd'].unique()

        try:
            region_var = [int(region_var)]

        except ValueError:
            assert region_var == 'All'
            region_var = self.df['legacy_system_cd'].unique()

        params = [obj_var,
                  segment_var,
                  field_var,
                  natl_acct_var,
                  self.field_options,
                  self.cutoff,
                  self.weights,
                  self.df,
                  self.fname]

        if level_var == 'warehouse':
            self.model = Vectorize(level_var, wh_var, *params)

        elif level_var == 'region':
            self.model = Vectorize(level_var, region_var, *params)

        else:
            # self.model = Vectorize(level_var, wh_var, *params)
            # enterprise pathway
            pass

    def loading_page2(self):
        self.define.withdraw()

        loading2 = Toplevel()
        self.loading2 = loading2
        # self.root.eval('tk::PlaceWindow %s center' % self.loading2.winfo_pathname(self.loading2.winfo_id()))
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
        self.redo = True
        self.loading2.withdraw()

        outputs = Toplevel()
        self.outputs = outputs
        # self.root.eval('tk::PlaceWindow %s center' % self.outputs.winfo_pathname(self.outputs.winfo_id()))
        outputs.title("Outputs")
        outputs.config(bg="white")
        self.output_frame = Frame(outputs)

        # Label(self.output_frame, text="Success!").grid(row=0, column=0, pady=10)
        Button(self.output_frame, text="Export to Excel", command=self.export).grid(row=1, column=0, pady=10)
        Button(self.output_frame, text="Rerun with new parameters", command=self.input_page).grid(row=2, column=0, pady=10)

        text = Text(self.output_frame)
        text.grid(row=3, column=0, pady=0)
        text.insert(INSERT, self.model.string_output())

        self.output_frame.grid(row=2, column=2)

        outputs.mainloop()

    def export(self):
        from tkinter.filedialog import asksaveasfilename
        addon = '_new_core' if self.objective.get() == 'Identify core products' else '_rationalized'
        newfname = self.fname.split('/')[-1][:-len(self.ext) - 1] + addon

        fout = asksaveasfilename(initialdir=''.join(self.fname.split('/')[:-1]),
                                 initialfile=newfname,
                                 filetypes=[('Excel spreadsheet', '.xlsx')])

        if fout is '':
            return
        else:
            self.model.df.to_excel(fout)

            Label(self.output_frame, text="Exported successfully!").grid(row=0, column=0, pady=10)


if __name__ == "__main__":
    GUI()
