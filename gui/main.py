from tkinter import *
import pandas as pd
import numpy as np
from itertools import compress
from sklearn.preprocessing import MinMaxScaler
from gui.scripts.vectorize import Vectorize


class GUI:

    def __init__(self):
        self.home_page()

    def home_page(self):
        self.root = Tk()
        self.root.title("Click UPLOAD to begin")
        self.root.config(bg="white")

        frame = Frame(self.root)

        Button(frame, text="UPLOAD", width=8, command=self.get_filename).grid(column=1, row=3, sticky=W)
        Button(frame, text="QUIT", width=8, command=self.root.destroy).grid(column=2, row=3, sticky=W)

        frame.grid(row=3, column=2)

        self.root.mainloop()

    def get_filename(self):
        """ get file and define it contains recognized columns """
        from tkinter.filedialog import askopenfilename

        self.fname = askopenfilename()
        if not self.fname:
            return

        msg = self.check_file()
        if msg is not None:
            return

        keep = ['legacy_division_cd',
                'segment',
                'legacy_product_cd',
                'sales_channel',
                'sales_6_mos',
                'cogs_6mos',
                'qty_6mos',
                'picks_6mos',
                'net_oh',
                'pallet_quantity',
                # 'item_poi_days',
                'legacy_customer_cd',
                'core_item_flag',
                'margin_%',
                'net_oh_$',
                'dioh']

        if all(x in keep[:5] for x in self.df.columns):
            return

        """ set input variables for input_page screen """
        self.field_options = ['turn_and_earn', 'revenue_6mos', 'customers_per_product'] + [x for x in self.df.columns if x in keep]

        wh_options = self.df.legacy_division_cd.unique().astype(str)

        if wh_options is None:
            return

        self.wh_options = np.append(['All'], wh_options)

        self.segment_options = self.df.segment.unique()
        if self.segment_options is None:
            return

        self.wh_var = StringVar(self.root, value='All')
        self.segment_var = StringVar(self.root, value=self.segment_options[0])
        self.cutoff_var = StringVar(self.root, value='20')
        self.field_var = []

        self.input_page()

    def input_page(self):
        self.root.withdraw()
        define = Toplevel()
        self.define = define
        define.title("Define inputs")
        define.config(bg="white")
        frame = Frame(define)
        self.input_frame = frame

        Label(frame, text="Modify model inputs below and click RUN. ").grid(row=0, column=0, pady=10)
        Button(frame, text="RUN", width=8, command=self.check_inputs).grid(row=0, column=1, pady=10)

        Label(frame, text="Select segment: ").grid(row=1, column=0, pady=10)
        OptionMenu(frame, self.segment_var, *self.segment_options).grid(row=1, column=1, pady=10)

        Label(frame, text="Select warehouse(s): ").grid(row=2, column=0, pady=10)
        OptionMenu(frame, self.wh_var, *self.wh_options).grid(row=2, column=1, pady=10)

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
            b.grid(row=4 + (idx // 2), column=((idx + 1) % 2) + 1, pady=10)
            self.field_var.append(var)

        self.last_row = 3 + (len(self.field_options) // 2)

        frame.grid(row=4, column=len(self.field_options))

        define.mainloop()

    def check_inputs(self):
        try:
            float(self.cutoff_var.get())
            self.format_vars()
            self.output_page()

        except ValueError:
            Label(self.input_frame,text='ERROR. Enter a numeric value for % core products.')\
                .grid(row=self.last_row + 1, column=0)
            return

    def format_vars(self):
        self.field_var = [x.get() for x in self.field_var]

        for var in [self.cutoff_var, self.segment_var, self.wh_var]:
            var = var.get()

        # if self.wh_var == 'All':
        #     self_wh_var = self.wh_options[1:]

    def output_page(self):
        self.define.withdraw()
        outputs = Toplevel()
        self.outputs = outputs
        outputs.title("Outputs")
        outputs.config(bg="white")
        frame = Frame(outputs)
        # self.outputs.geometry('500x500')

        """ RUN BACKEND """
        model = Vectorize(wh=self.wh_var.get(),
                          segment=self.segment_var,
                          fields=self.field_var,
                          field_options=self.field_options,
                          cutoff=self.cutoff_var.get(),
                          df=self.df,
                          fname=self.fname)

        label = Label(frame, text="Success!").grid(row=0, column=0, pady=10)
        Button(frame, text="Download new Excel sheet", command=model.export).grid(row=1, column=0, pady=10)

        text = Text(frame)
        text.grid(row=2, column=0, pady=0)
        text.insert(INSERT, model.string_output())

        frame.grid(row=2, column=2)

        outputs.mainloop()

    def check_file(self):
        ext = self.fname.split('.')[1]
        if ext == 'xlsx':
            try:
                self.df = pd.read_excel(self.fname, sheet_name='Sheet1')
            except:  # TODO - run test on xlsx w/o multiple sheets and catch error
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

        continuous_labels = ['sales_6_mos',
                             'qty_6mos',
                             'cogs_6mos',
                             'margin_%',
                             'picks_6mos',
                             'net_oh',
                             'net_oh_$',
                             'dioh']

        self.df[continuous_labels] = self.df[continuous_labels][(self.df[continuous_labels] > 0)].fillna(0)


        # TODO -- figure out why it isn't touching the last 5 columns
        # scaler = MinMaxScaler()
        #
        # out = scaler.fit_transform(self.df[continuous_labels])
        # df2 = pd.DataFrame(out, columns=continuous_labels)
        #
        # for column in df2.columns:
        #     self.df[column] = df2[column]
        #
        # self.df.fillna(0)
        self.df[continuous_labels] = self.df[continuous_labels].apply(self.get_max, axis=0)

    def get_max(self, col):
        return col / max(col)


if __name__ == "__main__":
    GUI()
