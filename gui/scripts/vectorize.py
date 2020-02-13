import pandas as pd
from itertools import compress
import numpy as np


def getdf(fname):
    df = pd.read_excel(fname, sheet_name='Sheet1')

    df.columns = [c.replace(' ', '_').lower() for c in df.columns]

    df = df[df['sales_channel'] == 'Warehouse']

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
            'item_poi_days',
            'legacy_customer_cd',
            'core_item_flag',
            'margin_%',
            'net_oh_$',
            'dioh']

    df = df[keep].fillna(0).replace('-', 0)

    continuous_labels = ['sales_6_mos',
                         'qty_6mos',
                         'cogs_6mos',
                         'margin_%',
                         'picks_6mos',
                         'net_oh',
                         'net_oh_$',
                         'dioh']

    df[continuous_labels] = df[continuous_labels][(df[continuous_labels] > 0)].fillna(0)
    # print(df.shape[0])
    return df


class WarehouseLevelVectors:
    def __init__(self, wh, segment, fields, field_options, cutoff, df, fname):
        self.fname = fname

        self.segment = segment
        self.df = df[df['segment'] == self.segment]

        self.wh = wh
        try:
            self.wh = int(self.wh)
        except TypeError:
            pass

        if isinstance(self.wh, int):
            self.df = self.df[self.df['legacy_division_cd'] == wh]
            self.wh = [int(self.wh)]
        elif isinstance(self.wh, list):
            for wh in self.wh:
                self.df = self.df[self.df['legacy_division_cd'] == wh]
        elif isinstance(self.wh, str):
            if self.wh == 'All':
                self.wh = self.df['legacy_division_cd'].unique()
            else:
                self.df = self.df[self.df['legacy_division_cd'] == wh]
                self.wh = [int(self.wh)]

        self.field_options = field_options
        self.fields = fields
        self.cutoff = cutoff
        self.selected_fields = compress(self.field_options, [bool(x) for x in self.fields])

        self.wh_to_prod = self.get_wh_to_prod()

        self.wp_to_sales, \
        self.wp_to_costs, \
        self.wp_to_picks, \
        self.wp_to_quantity, \
        self.wp_to_ncustomers, \
        self.wp_to_pallet, \
        self.wp_to_margin, \
        self.wp_to_coreflag, \
        self.wp_to_oh,\
        self.wp_to_te,\
        self.wp_to_profit = self.get_mappings()

        self.vecs, self.wp_to_vector = self.get_vectors()

        self.wp_to_flag = self.get_flags()

    def get_wh_to_prod(self):
        wh_to_prod = {}

        for wh in self.wh:
            wh_to_prod[wh] = self.df[self.df['legacy_division_cd'] == wh]['legacy_product_cd'].unique()

        return wh_to_prod

    def get_mappings(self):
        wp_to_sales = {}
        wp_to_costs = {}
        wp_to_picks = {}
        wp_to_quantity = {}
        wp_to_ncustomers = {}
        wp_to_pallet = {}
        wp_to_oh = {}

        for w in self.wh:
            for p in self.wh_to_prod[w]:
                wp_to_sales[w, p] = []
                wp_to_costs[w, p] = []
                wp_to_picks[w, p] = []
                wp_to_quantity[w, p] = []
                wp_to_ncustomers[w, p] = []
                wp_to_oh[w, p] = []

        cols = ['legacy_division_cd',
                'legacy_product_cd',
                'sales_6_mos',
                'cogs_6mos',
                'picks_6mos',
                'qty_6mos',
                'legacy_customer_cd',
                'net_oh']

        for w, p, s, c, pk, q, cust, oh in self.df[cols].values:
            if p in self.wh_to_prod[w]:
                wp_to_sales[w, p].append(s)
                wp_to_costs[w, p].append(c)
                wp_to_picks[w, p].append(pk)
                wp_to_quantity[w, p].append(q)
                wp_to_ncustomers[w, p].append(cust)
                wp_to_oh[w, p].append(oh)

        for w in self.wh:
            for p in self.wh_to_prod[w]:
                wp_to_sales[w, p] = sum(wp_to_sales[w, p])
                wp_to_costs[w, p] = sum(wp_to_costs[w, p])
                wp_to_picks[w, p] = sum(wp_to_picks[w, p])
                wp_to_quantity[w, p] = sum(wp_to_quantity[w, p])
                wp_to_ncustomers[w, p] = len(wp_to_ncustomers[w, p])
                wp_to_oh[w, p] = sum(wp_to_oh[w, p])

        wp_to_coreflag = {}
        for w, p, cf in self.df[['legacy_division_cd', 'legacy_product_cd', 'core_item_flag']].values:
            if p in self.wh_to_prod[w]:
                if cf == "Y":
                    wp_to_coreflag[w, p] = 1
                if cf == "N":
                    wp_to_coreflag[w, p] = 0

        wp_to_margin = {}
        for w in self.wh:
            for p in self.wh_to_prod[w]:
                s = wp_to_sales[w, p]
                c = wp_to_costs[w, p]
                #         ##DATA HAS TO BE CLEANED SO THAT COSTS THAT ARE EQUAL TO 0 DO NOT EXIST
                #         if s == 0 or c==0:
                #             wp_to_margin[w,p] = 0
                #         else:
                try:
                    wp_to_margin[w, p] = 100 * ((s - c) / s)
                except ZeroDivisionError:
                    wp_to_margin[w, p] = 0

        wp_to_te = {}
        wp_to_profit = {}
        for w in self.wh:
            for p in self.wh_to_prod[w]:
                if wp_to_oh[w, p] == 0:
                    wp_to_te[w, p] = 0
                else:
                    wp_to_te[w, p] = wp_to_margin[w, p] * wp_to_costs[w, p] / wp_to_oh[w, p]
                wp_to_profit[w, p] = wp_to_sales[w, p] - wp_to_costs[w, p]

        return wp_to_sales, \
               wp_to_costs, \
               wp_to_picks, \
               wp_to_quantity, \
               wp_to_ncustomers, \
               wp_to_pallet, \
               wp_to_margin, \
               wp_to_coreflag, \
               wp_to_oh, \
               wp_to_te, \
               wp_to_profit

    def get_vectors(self):
        vecs = []
        wp_to_vector = {}

        for w in self.wh:
            for p in self.wh_to_prod[w]:
                vec = [self.wp_to_te[w, p], self.wp_to_profit[w, p], self.wp_to_ncustomers[w, p]]
                vec = np.linalg.norm(vec)
                # vec = self.norm(vec)
                wp_to_vector[w, p] = vec
                vecs.append(vec)

        return vecs, wp_to_vector

    def get_flags(self):
        wp_to_flag = {}

        for w in self.wh:
            prod_to_score = {}
            for p in self.wh_to_prod[w]:
                prod_to_score[p] = self.wp_to_vector[w, p]

            prods_by_score = sorted(prod_to_score, key=prod_to_score.__getitem__)

            cutoffIdx = int(len(prods_by_score) * (1 - (float(self.cutoff) / 100)))
            self.n_core = cutoffIdx

            non_core_prods = prods_by_score[:cutoffIdx]
            core_prods = prods_by_score[cutoffIdx:]

            for p in non_core_prods:
                wp_to_flag[w, p] = 0
            for p in core_prods:
                wp_to_flag[w, p] = 1

        return wp_to_flag

    def norm(self, vec):
        vec = np.array(vec)
        length = len(vec)
        try:
            return sum(vec ** length) ** (1 / length)
        except ZeroDivisionError:
            return 0

    def export(self):
        self.df['New Core Flag'] = self.df.apply(self.iscore, axis=1)
        self.df.to_excel(str(self.fname.split('.')[0]) + '_newflags.xlsx')

    def iscore(self, row):
        if self.wp_to_flag[row['legacy_division_cd'], row['legacy_product_cd']] == 0:
            return 'N'
        else:
            return 'Y'

    def string_output(self):
        core = []
        non_core = []

        for wh in self.wh:
            for p in self.wh_to_prod[wh]:
                if self.wp_to_flag[wh, p] == 0:
                    non_core.append(p)
                else:
                    core.append(p)

        core_avg_profit = []
        non_core_avg_profit = []
        core_avg_TE = []
        non_core_avg_TE = []
        core_avg_ncust = []
        non_core_avg_ncust = []

        for p in core:
            for w in self.wh:
                try:
                    core_avg_profit.append(self.wp_to_profit[w, p])
                    core_avg_TE.append(self.wp_to_te[w, p])
                    core_avg_ncust.append(self.wp_to_ncustomers[w, p])
                except KeyError:
                    pass

        core_avg_profit = np.average(core_avg_profit)
        core_avg_TE = np.average(core_avg_TE)
        core_avg_ncust = np.average(core_avg_ncust)

        for p in non_core:
            for w in self.wh:
                try:
                    non_core_avg_profit.append(self.wp_to_profit[w, p])
                    non_core_avg_TE.append(self.wp_to_te[w, p])
                    non_core_avg_ncust.append(self.wp_to_ncustomers[w, p])
                except KeyError:
                    pass

        non_core_avg_profit = np.round(np.average(non_core_avg_profit), 2)
        non_core_avg_TE = np.round(np.average(non_core_avg_TE), 2)
        non_core_avg_ncust = np.round(np.average(non_core_avg_ncust), 2)

        # Average profit
        avg_profit = []
        # Average TE
        avg_TE = []
        # Average number of customers
        avg_ncust = []
        for wh in self.wh:
            for p in self.wh_to_prod[wh]:
                avg_profit.append(self.wp_to_profit[wh, p])
                avg_TE.append(self.wp_to_te[wh, p])
                avg_ncust.append(self.wp_to_ncustomers[wh, p])
        avg_profit = np.round(np.average(avg_profit),2)
        avg_TE = np.round(np.average(avg_TE),2)
        avg_ncust = np.round(np.average(avg_ncust),2)

        inputs = [self.wh, self.n_core, core_avg_profit, core_avg_TE, core_avg_ncust]
        string = """For warehouse(s) {}:
        Number of core items: {}
        Core items average profit: {}
        Core items average turn and earn: {}
        Core items average number of customers: {}""".format(*inputs)

        return string
        #
        # print("Number of non core items", len(non_core))
        # print("Non Core Items Average Profit", non_core_avg_profit)
        # print('Non Core Items Average TE', non_core_avg_TE)
        # print("Non Core Items Average number of customers", non_core_avg_ncust)
        # print()
        # print('All Items in warehouse average profit', avg_profit)
        # print('All Items in warehouse average TE', avg_TE)
        # print('All Items in warehouse average number of customers', avg_ncust)
        # print()


class RegionLevelVectors:
    def __init__(self, region, segment, fields, field_options, cutoff, df, fname):
        pass

    def export(self):
        print('Exported!')

    def string_output(self):
        print('String output!')
        return "This is the string!"


if __name__ == '__main__':
    m = WarehouseLevelVectors(wh=83,
                              segment='Facility Solutions',
                              fields=[1, 1, 1],
                              field_options=['turn_and_earn', 'profit_6mos', 'cogs_6mos'],
                              cutoff=20,
                              df=getdf("../Clean_Data_short.xlsx"),
                              fname="../Clean_Data_short.xlsx")

    m.export()
    # print(m.string_output())

    m2 = WarehouseLevelVectors(wh='83',
                              segment='Facility Solutions',
                              fields=[1, 1, 1],
                              field_options=['turn_and_earn', 'profit_6mos'],
                              cutoff=20,
                              df=getdf("../Clean_Data_short.xlsx"),
                              fname="../Clean_Data_short.xlsx")

    m2.export()
    print(m2.string_output())
