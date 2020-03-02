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

    return df


class WarehouseLevelVectors:
    def __init__(self, wh, segment, fields, field_options, cutoff, df, fname, scaled=False):
        self.scaled = scaled
        self.fname = fname

        self.segment = segment
        self.df = df[df['segment'] == self.segment]

        self.wh = wh
        for wh in self.df['legacy_division_cd'].unique():
            if wh not in self.wh:
                self.df = self.df[self.df['legacy_division_cd'] != wh]

        self.maxes = self.df.max(axis=0).to_dict()
        self.format_df()

        self.field_options = field_options
        self.fields = fields
        self.cutoff = cutoff
        self.selected_fields = compress(self.field_options, [bool(x) for x in self.fields])

        self.wh_to_prod = self.get_wh_to_prod()

    def format_df(self):
        self.continuous_labels = ['sales_6_mos',
                             'qty_6mos',
                             'cogs_6mos',
                             # 'margin_%',
                             'picks_6mos',
                             'net_oh',
                             'net_oh_$',
                             'dioh']

        self.df[self.continuous_labels] = self.df[self.continuous_labels][(self.df[self.continuous_labels] > 0)].fillna(0)

        if self.scaled:
            self.df[self.continuous_labels] = self.df[self.continuous_labels].apply(self.get_max, axis=0)
            # self.df.to_excel('../input_data/masked.xlsx')

    def get_max(self, col):
        return col / self.maxes[col.name]

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

        self.maxes['turn_and_earn'] = wp_to_te[max(wp_to_te, key=wp_to_te.get)]
        self.maxes['profit_6mos'] = wp_to_profit[max(wp_to_profit, key=wp_to_profit.get)]
        self.maxes['customers_per_product'] = wp_to_ncustomers[max(wp_to_ncustomers, key=wp_to_ncustomers.get)]

        self.wp_to_sales = wp_to_sales
        self.wp_to_costs = wp_to_costs
        self.wp_to_picks = wp_to_picks
        self.wp_to_quantity = wp_to_quantity
        self.wp_to_ncustomers = wp_to_ncustomers
        self.wp_to_pallet = wp_to_pallet
        self.wp_to_margin = wp_to_margin
        self.wp_to_coreflag = wp_to_coreflag
        self.wp_to_oh = wp_to_oh
        self.wp_to_te = wp_to_te
        self.wp_to_profit = wp_to_profit

    def get_vectors(self):
        vecs = []
        wp_to_vector = {}

        var_dict = {'profit_6mos': self.wp_to_profit,
                    "margin_%": self.wp_to_margin,
                    'turn_and_earn': self.wp_to_te,
                    'customers_per_product': self.wp_to_ncustomers,
                    'sales_6_mos': self.wp_to_sales,
                    'cogs_6mos': self.wp_to_costs,
                    'qty_6mos': self.wp_to_quantity,
                    'picks_6mos': self.wp_to_picks,
                    'net_oh': self.wp_to_oh}

        for w in self.wh:
            for p in self.wh_to_prod[w]:
                vec = []
                for key in self.selected_fields:
                    if key in self.continuous_labels:
                        vec.append(var_dict[key][w, p] / self.maxes[key])
                    else:
                        vec.append(var_dict[key][w, p])
                # vec = [self.wp_to_te[w, p], self.wp_to_profit[w, p], self.wp_to_ncustomers[w, p]]
                # vec = np.linalg.norm(vec)
                vec = self.norm(vec)
                wp_to_vector[w, p] = vec
                vecs.append(vec)

        self.vecs, self.wp_to_vector = vecs, wp_to_vector

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

        self.wp_to_flag = wp_to_flag

    def norm(self, vec):
        vec = np.array(vec)
        length = len(vec)
        try:
            return sum(np.sign(vec) * (np.abs(vec) ** length)) ** (1 / length)
        except ZeroDivisionError:
            return 0

    def export(self, fout):
        self.df['New Core Flag'] = self.df.apply(self.iscore, axis=1)
        self.df.to_excel(fout)

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

        avg_profit = []
        avg_TE = []
        avg_ncust = []
        for wh in self.wh:
            for p in self.wh_to_prod[wh]:
                avg_profit.append(self.wp_to_profit[wh, p])
                avg_TE.append(self.wp_to_te[wh, p])
                avg_ncust.append(self.wp_to_ncustomers[wh, p])
        avg_profit = np.round(np.average(avg_profit),2)
        avg_TE = np.round(np.average(avg_TE),2)
        avg_ncust = np.round(np.average(avg_ncust),2)

        inputs = [self.wh,
                  self.n_core,
                  core_avg_profit,
                  core_avg_TE,
                  core_avg_ncust,
                  len(non_core),
                  non_core_avg_profit,
                  non_core_avg_TE,
                  non_core_avg_ncust,
                  avg_ncust,
                  avg_TE,
                  avg_ncust]

        string = """For warehouse(s) {}:
        
            Number of core items: {}
            Core items average profit: {}
            Core items average turn and earn: {}
            Core items average number of customers: {}
            
            Number of non core items: {}
            Non Core Items Average Profit: {}
            Non Core Items Average TE: {}
            Non Core Items Average number of customers: {}
            
            All Items in warehouse average profit: {}
            All Items in warehouse average TE: {}
            All Items in warehouse average number of customers: {}""".format(*inputs)

        return string

    def run(self):
        self.get_mappings()
        self.get_vectors()
        self.get_flags()


class RegionLevelVectors:
    def __init__(self, region, segment, fields, field_options, cutoff, df, fname, scaled=False):
        self.scaled = scaled
        self.fname = fname

        self.segment = segment
        self.df = df[df['segment'] == self.segment]

        self.region = region
        for region in self.df['legacy_system_cd'].unique():
            if region not in self.region:
                self.df = self.df[self.df['legacy_system_cd'] != region]

        self.maxes = self.df.max(axis=0).to_dict()
        self.format_df()

        self.field_options = field_options
        self.fields = fields
        self.cutoff = cutoff
        self.selected_fields = compress(self.field_options, [bool(x) for x in self.fields])

        self.region_to_prod = self.get_region_to_prod()

    def format_df(self):
        self.continuous_labels = ['sales_6_mos',
                                  'qty_6mos',
                                  'cogs_6mos',
                                  # 'margin_%',
                                  'picks_6mos',
                                  'net_oh',
                                  'net_oh_$',
                                  'dioh']

        self.df[self.continuous_labels] = self.df[self.continuous_labels][(self.df[self.continuous_labels] > 0)].fillna(
            0)

        if self.scaled:
            self.df[self.continuous_labels] = self.df[self.continuous_labels].apply(self.get_max, axis=0)
            # self.df.to_excel('../input_data/masked.xlsx')

    def get_max(self, col):
        return col / self.maxes[col.name]

    def get_region_to_prod(self):
        region_to_prod = {}

        for region in self.region:
            region_to_prod[region] = self.df[self.df['legacy_system_cd'] == region]['legacy_product_cd'].unique()

        return region_to_prod

    def get_mappings(self):
        r_to_sales = {}
        r_to_costs = {}
        r_to_picks = {}
        r_to_quantity = {}
        r_to_ncustomers = {}
        r_to_pallet = {}
        r_to_oh = {}

        for r in self.region:
            for p in self.region_to_prod[r]:
                r_to_sales[r, p] = []
                r_to_costs[r, p] = []
                r_to_picks[r, p] = []
                r_to_quantity[r, p] = []
                r_to_ncustomers[r, p] = []
                r_to_oh[r, p] = []

        cols = ['legacy_system_cd',
                'legacy_product_cd',
                'sales_6_mos',
                'cogs_6mos',
                'picks_6mos',
                'qty_6mos',
                'legacy_customer_cd',
                'net_oh']

        for r, p, s, c, pk, q, cust, oh in self.df[cols].values:
            if p in self.region_to_prod[r]:
                r_to_sales[r, p].append(s)
                r_to_costs[r, p].append(c)
                r_to_picks[r, p].append(pk)
                r_to_quantity[r, p].append(q)
                r_to_ncustomers[r, p].append(cust)
                r_to_oh[r, p].append(oh)

        for w in self.region:
            for p in self.region_to_prod[w]:
                r_to_sales[r, p] = sum(r_to_sales[r, p])
                r_to_costs[w, p] = sum(r_to_costs[r, p])
                r_to_picks[w, p] = sum(r_to_picks[r, p])
                r_to_quantity[w, p] = sum(r_to_quantity[r, p])
                r_to_ncustomers[w, p] = len(r_to_ncustomers[r, p])
                r_to_oh[w, p] = sum(r_to_oh[r, p])

        r_to_coreflag = {}
        for r, p, cf in self.df[['legacy_system_cd', 'legacy_product_cd', 'core_item_flag']].values:
            if p in self.region_to_prod[w]:
                if cf == "Y":
                    r_to_coreflag[r, p] = 1
                if cf == "N":
                    r_to_coreflag[r, p] = 0

        r_to_margin = {}
        for r in self.region:
            for p in self.region_to_prod[w]:
                s = r_to_sales[r, p]
                c = r_to_costs[r, p]
                #         ##DATA HAS TO BE CLEANED SO THAT COSTS THAT ARE EQUAL TO 0 DO NOT EXIST
                #         if s == 0 or c==0:
                #             wp_to_margin[w,p] = 0
                #         else:
                try:
                    r_to_margin[r, p] = 100 * ((s - c) / s)
                except ZeroDivisionError:
                    r_to_margin[r, p] = 0

        r_to_te = {}
        r_to_profit = {}
        for r in self.region:
            for p in self.region_to_prod[w]:
                if r_to_oh[r, p] == 0:
                    r_to_te[r, p] = 0
                else:
                    r_to_te[r, p] = r_to_margin[r, p] * r_to_costs[r, p] / r_to_oh[r, p]
                r_to_profit[r, p] = r_to_sales[r, p] - r_to_costs[r, p]

        self.maxes['turn_and_earn'] = r_to_te[max(r_to_te, key=r_to_te.get)]
        self.maxes['profit_6mos'] = r_to_profit[max(r_to_profit, key=r_to_profit.get)]
        self.maxes['customers_per_product'] = r_to_ncustomers[max(r_to_ncustomers, key=r_to_ncustomers.get)]

        self.r_to_sales = r_to_sales
        self.r_to_costs = r_to_costs
        self.r_to_picks = r_to_picks
        self.r_to_quantity = r_to_quantity
        self.r_to_ncustomers = r_to_ncustomers
        self.r_to_pallet = r_to_pallet
        self.r_to_margin = r_to_margin
        self.r_to_coreflag = r_to_coreflag
        self.r_to_oh = r_to_oh
        self.r_to_te = r_to_te
        self.r_to_profit = r_to_profit

    def get_vectors(self):
        vecs = []
        r_to_vector = {}

        var_dict = {'profit_6mos': self.r_to_profit,
                    "margin_%": self.r_to_margin,
                    'turn_and_earn': self.r_to_te,
                    'customers_per_product': self.r_to_ncustomers,
                    'sales_6_mos': self.r_to_sales,
                    'cogs_6mos': self.r_to_costs,
                    'qty_6mos': self.r_to_quantity,
                    'picks_6mos': self.r_to_picks,
                    'net_oh': self.r_to_oh}

        for w in self.region:
            for p in self.region_to_prod[w]:
                vec = []
                for key in self.selected_fields:
                    if key in self.continuous_labels:
                        vec.append(var_dict[key][w, p] / self.maxes[key])
                    else:
                        vec.append(var_dict[key][w, p])
                # vec = [self.wp_to_te[w, p], self.wp_to_profit[w, p], self.wp_to_ncustomers[w, p]]
                # vec = np.linalg.norm(vec)
                vec = self.norm(vec)
                r_to_vector[w, p] = vec
                vecs.append(vec)

        self.vecs, self.r_to_vector = vecs, r_to_vector

    def get_flags(self):
        r_to_flag = {}

        for w in self.region:
            prod_to_score = {}
            for p in self.region_to_prod[w]:
                prod_to_score[p] = self.r_to_vector[w, p]

            prods_by_score = sorted(prod_to_score, key=prod_to_score.__getitem__)

            cutoffIdx = int(len(prods_by_score) * (1 - (float(self.cutoff) / 100)))
            self.n_core = cutoffIdx

            non_core_prods = prods_by_score[:cutoffIdx]
            core_prods = prods_by_score[cutoffIdx:]

            for p in non_core_prods:
                r_to_flag[w, p] = 0
            for p in core_prods:
                r_to_flag[w, p] = 1

        self.r_to_flag = r_to_flag

    def norm(self, vec):
        vec = np.array(vec)
        length = len(vec)
        try:
            return sum(np.sign(vec) * (np.abs(vec) ** length)) ** (1 / length)
        except ZeroDivisionError:
            return 0

    def export(self, fout):
        self.df['New Core Flag'] = self.df.apply(self.iscore, axis=1)
        self.df.to_excel(fout)

    def iscore(self, row):
        if self.r_to_flag[row['legacy_system_cd'], row['legacy_product_cd']] == 0:
            return 'N'
        else:
            return 'Y'

    def string_output(self):
        core = []
        non_core = []

        for wh in self.region:
            for p in self.region_to_prod[wh]:
                if self.r_to_flag[wh, p] == 0:
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
            for w in self.region:
                try:
                    core_avg_profit.append(self.r_to_profit[w, p])
                    core_avg_TE.append(self.r_to_te[w, p])
                    core_avg_ncust.append(self.r_to_ncustomers[w, p])
                except KeyError:
                    pass

        core_avg_profit = np.average(core_avg_profit)
        core_avg_TE = np.average(core_avg_TE)
        core_avg_ncust = np.average(core_avg_ncust)

        for p in non_core:
            for w in self.region:
                try:
                    non_core_avg_profit.append(self.r_to_profit[w, p])
                    non_core_avg_TE.append(self.r_to_te[w, p])
                    non_core_avg_ncust.append(self.r_to_ncustomers[w, p])
                except KeyError:
                    pass

        non_core_avg_profit = np.round(np.average(non_core_avg_profit), 2)
        non_core_avg_TE = np.round(np.average(non_core_avg_TE), 2)
        non_core_avg_ncust = np.round(np.average(non_core_avg_ncust), 2)

        avg_profit = []
        avg_TE = []
        avg_ncust = []
        for wh in self.region:
            for p in self.region_to_prod[wh]:
                avg_profit.append(self.r_to_profit[wh, p])
                avg_TE.append(self.r_to_te[wh, p])
                avg_ncust.append(self.r_to_ncustomers[wh, p])
        avg_profit = np.round(np.average(avg_profit), 2)
        avg_TE = np.round(np.average(avg_TE), 2)
        avg_ncust = np.round(np.average(avg_ncust), 2)

        inputs = [self.region,
                  self.n_core,
                  core_avg_profit,
                  core_avg_TE,
                  core_avg_ncust,
                  len(non_core),
                  non_core_avg_profit,
                  non_core_avg_TE,
                  non_core_avg_ncust,
                  avg_ncust,
                  avg_TE,
                  avg_ncust]

        string = """For warehouse(s) {}:

            Number of core items: {}
            Core items average profit: {}
            Core items average turn and earn: {}
            Core items average number of customers: {}

            Number of non core items: {}
            Non Core Items Average Profit: {}
            Non Core Items Average TE: {}
            Non Core Items Average number of customers: {}

            All Items in warehouse average profit: {}
            All Items in warehouse average TE: {}
            All Items in warehouse average number of customers: {}""".format(*inputs)

        return string

    def run(self):
        self.get_mappings()
        self.get_vectors()
        self.get_flags()


if __name__ == '__main__':
    m = WarehouseLevelVectors(wh=[19],
                              segment='Facility Solutions',
                              fields=[1, 1, 1],
                              field_options=['turn_and_earn', 'profit_6mos', 'cogs_6mos'],
                              cutoff=20,
                              df=getdf("../../data/Clean_Data_short.xlsx"),
                              fname="../../data/Clean_Data.xlsx")

    m.run()
    print(m.string_output())
    # m.export()
    # print(m.wp_to_vector)
    # print(m.wp_to_coreflag)
