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


class Vectorize:
    def __init__(self, level, selections, segment, fields, field_options, cutoff, weights, df, fname):
        self.fname = fname

        self.segment = segment
        self.df = df[df['segment'] == self.segment]

        if level == 'warehouse':
            self.level = 'legacy_division_cd'
        elif level == 'region':
            self.level = 'legacy_system_cd'

        self.selections = selections  # warehouses selected OR regions selected, depending on level

        for selection in self.df[self.level].unique():
            if selection not in self.selections:
                self.df = self.df[self.df[self.level] != selections]

        self.maxes = self.df.max(axis=0).to_dict()
        self.mins = self.df.min(axis=0).to_dict()

        self.format_df()

        self.field_options = field_options
        self.fields = fields
        self.cutoff = cutoff
        self.weights = np.array(list(filter(lambda num: num != 0, weights)))

        self.selected_fields = compress(self.field_options, [bool(x) for x in self.fields])

        self.selections_to_prod = self.get_selections_to_prod()
        print(self.selections_to_prod)

    def format_df(self):
        self.continuous_labels = ['sales_6_mos',
                                  'qty_6mos',
                                  'cogs_6mos',
                                  # 'margin_%',
                                  'picks_6mos',
                                  'net_oh',
                                  'net_oh_$',
                                  'dioh']

        # self.df[self.continuous_labels] = self.df[self.continuous_labels][(self.df[self.continuous_labels] > 0)].fillna(
        #     0)

        # self.df[self.continuous_labels] = self.df[self.continuous_labels].apply(self.scale, axis=0)

    # def scale(self, col):
    #     return (col + min(col)) / max(col)

    def get_selections_to_prod(self):
        selections_to_prod = {}

        for selections in self.selections:
            selections_to_prod[selections] = self.df[self.df[self.level] == selections]['legacy_product_cd'].unique()

        return selections_to_prod

    def get_mappings(self):
        wp_to_sales = {}
        wp_to_costs = {}
        wp_to_picks = {}
        wp_to_quantity = {}
        wp_to_ncustomers = {}
        wp_to_pallet = {}
        wp_to_oh = {}

        for w in self.selections:
            for p in self.selections_to_prod[w]:
                wp_to_sales[w, p] = []
                wp_to_costs[w, p] = []
                wp_to_picks[w, p] = []
                wp_to_quantity[w, p] = []
                wp_to_ncustomers[w, p] = []
                wp_to_oh[w, p] = []

        cols = [self.level,
                'legacy_product_cd',
                'sales_6_mos',
                'cogs_6mos',
                'picks_6mos',
                'qty_6mos',
                'legacy_customer_cd',
                'net_oh_$']

        for w, p, s, c, pk, q, cust, oh in self.df[cols].values:
            if p in self.selections_to_prod[w]:
                wp_to_sales[w, p].append(s)
                wp_to_costs[w, p].append(c)
                wp_to_picks[w, p].append(pk)
                wp_to_quantity[w, p].append(q)
                wp_to_ncustomers[w, p].append(cust)
                wp_to_oh[w, p].append(oh)

        for w in self.selections:
            for p in self.selections_to_prod[w]:
                wp_to_sales[w, p] = sum(wp_to_sales[w, p])
                wp_to_costs[w, p] = sum(wp_to_costs[w, p])
                wp_to_picks[w, p] = sum(wp_to_picks[w, p])
                wp_to_quantity[w, p] = sum(wp_to_quantity[w, p])
                wp_to_ncustomers[w, p] = len(wp_to_ncustomers[w, p])
                wp_to_oh[w, p] = sum(wp_to_oh[w, p])

        wp_to_coreflag = {}
        for w, p, cf in self.df[[self.level, 'legacy_product_cd', 'core_item_flag']].values:
            if p in self.selections_to_prod[w]:
                if cf == "Y":
                    wp_to_coreflag[w, p] = 1
                if cf == "N":
                    wp_to_coreflag[w, p] = 0

        wp_to_margin = {}
        for w in self.selections:
            for p in self.selections_to_prod[w]:
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

        wp_to_turn = {}
        wp_to_profit = {}
        for w in self.selections:
            for p in self.selections_to_prod[w]:
                if wp_to_oh[w, p] == 0:
                    wp_to_turn[w, p] = 0
                else:
                    # wp_to_turn[w, p] = wp_to_margin[w, p] * wp_to_costs[w, p] / wp_to_oh[w, p]
                    wp_to_turn[w, p] = wp_to_costs[w, p] / wp_to_oh[w, p]

                wp_to_profit[w, p] = wp_to_sales[w, p] - wp_to_costs[w, p]

        self.maxes['turn_6mos'] = wp_to_turn[max(wp_to_turn, key=wp_to_turn.get)]
        self.maxes['profit_6mos'] = wp_to_profit[max(wp_to_profit, key=wp_to_profit.get)]
        self.maxes['customers_per_product'] = wp_to_ncustomers[max(wp_to_ncustomers, key=wp_to_ncustomers.get)]

        self.mins['turn_6mos'] = wp_to_turn[min(wp_to_turn, key=wp_to_turn.get)]
        self.mins['profit_6mos'] = wp_to_profit[min(wp_to_profit, key=wp_to_profit.get)]
        self.mins['customers_per_product'] = wp_to_ncustomers[min(wp_to_ncustomers, key=wp_to_ncustomers.get)]

        self.wp_to_sales = wp_to_sales
        self.wp_to_costs = wp_to_costs
        self.wp_to_picks = wp_to_picks
        self.wp_to_quantity = wp_to_quantity
        self.wp_to_ncustomers = wp_to_ncustomers
        self.wp_to_pallet = wp_to_pallet
        self.wp_to_margin = wp_to_margin
        self.wp_to_coreflag = wp_to_coreflag
        self.wp_to_oh = wp_to_oh
        self.wp_to_turn = wp_to_turn
        self.wp_to_profit = wp_to_profit

    def get_vectors(self):
        vecs = []
        wp_to_vectorscore = {}

        var_dict = {'profit_6mos': self.wp_to_profit,
                    "margin_%": self.wp_to_margin,
                    'turn_6mos': self.wp_to_turn,
                    'customers_per_product': self.wp_to_ncustomers,
                    'sales_6_mos': self.wp_to_sales,
                    'cogs_6mos': self.wp_to_costs,
                    'qty_6mos': self.wp_to_quantity,
                    'picks_6mos': self.wp_to_picks,
                    'net_oh_$_6mos': self.wp_to_oh}

        for w in self.selections:
            for p in self.selections_to_prod[w]:
                vec = []
                for key in self.selected_fields:
                    if key in self.continuous_labels:
                        # scales here by adding most negative value, dividing by most positive number
                        # negative numbers were NOT previuosly filtered out and dictionary values are the actual values
                        # vector and vectorscore ONLY will reflect scaled values -- can use dictionary values for output
                        vec.append((var_dict[key][w, p] + self.mins[key]) / self.maxes[key])
                    else:
                        vec.append(var_dict[key][w, p])

                wp_to_vectorscore[w, p] = 0 if vec == [] else self.norm(vec)
                vecs.append(vec)

        self.vecs, self.wp_to_vectorscore = vecs, wp_to_vectorscore

    def get_flags(self):
        wp_to_flag = {}

        for w in self.selections:
            prod_to_score = {}

            for p in self.selections_to_prod[w]:
                prod_to_score[p] = self.wp_to_vectorscore[w, p]

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
            return sum((vec * self.weights) ** length) ** (1 / length)

        except ZeroDivisionError:
            return 0

        except ValueError:
            # handles negative roots, but they shouldn't come up since all values >= 0
            return -(-sum(np.sign(vec) * (np.abs(vec * self.weights) ** length)) ** (1 / length))

    def export(self, fout):
        self.df['New Core Flag'] = self.df.apply(self.iscore, axis=1)
        self.df.to_excel(fout)

    def iscore(self, row):
        if self.wp_to_flag[row[self.level], row['legacy_product_cd']] == 0:
            return 'N'
        else:
            return 'Y'

    def string_output(self):
        core = []
        non_core = []

        for selections in self.selections:
            for p in self.selections_to_prod[selections]:
                if self.wp_to_flag[selections, p] == 0:
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
            for w in self.selections:
                try:
                    core_avg_profit.append(self.wp_to_profit[w, p])
                    core_avg_TE.append(self.wp_to_turn[w, p])
                    core_avg_ncust.append(self.wp_to_ncustomers[w, p])

                except KeyError:
                    pass

        core_avg_profit = np.average(core_avg_profit)
        core_avg_TE = np.average(core_avg_TE)
        core_avg_ncust = np.average(core_avg_ncust)

        for p in non_core:
            for w in self.selections:
                try:
                    non_core_avg_profit.append(self.wp_to_profit[w, p])
                    non_core_avg_TE.append(self.wp_to_turn[w, p])
                    non_core_avg_ncust.append(self.wp_to_ncustomers[w, p])

                except KeyError:
                    pass

        non_core_avg_profit = np.round(np.average(non_core_avg_profit), 2)
        non_core_avg_TE = np.round(np.average(non_core_avg_TE), 2)
        non_core_avg_ncust = np.round(np.average(non_core_avg_ncust), 2)

        avg_profit = []
        avg_TE = []
        avg_ncust = []
        for selections in self.selections:
            for p in self.selections_to_prod[selections]:
                avg_profit.append(self.wp_to_profit[selections, p])
                avg_TE.append(self.wp_to_turn[selections, p])
                avg_ncust.append(self.wp_to_ncustomers[selections, p])
        avg_profit = np.round(np.average(avg_profit), 2)
        avg_TE = np.round(np.average(avg_TE), 2)
        avg_ncust = np.round(np.average(avg_ncust), 2)

        inputs = [self.selections,
                  len(core),
                  core_avg_profit,
                  core_avg_TE,
                  core_avg_ncust,
                  len(non_core),
                  non_core_avg_profit,
                  non_core_avg_TE,
                  non_core_avg_ncust,
                  avg_profit,
                  avg_TE,
                  avg_ncust]

        string = """For warehouse(s) {}:

            Number of core items: {}
            Core items average profit: {}
            Core items average Turn: {}
            Core items average number of customers: {}

            Number of non core items: {}
            Non Core Items Average Profit: {}
            Non Core Items Average Turn: {}
            Non Core Items Average number of customers: {}

            All Items in warehouse average profit: {}
            All Items in warehouse average Turn: {}
            All Items in warehouse average number of customers: {}""".format(*inputs)

        return string

    def run(self):
        self.get_mappings()
        self.get_vectors()
        self.get_flags()


if __name__ == '__main__':
    m = Vectorize(level='region',
                  selections=['All'],
                  segment='Facility Solutions',
                  fields=[1, 1, 1],
                  field_options=['turn_6mos', 'profit_6mos', 'cogs_6mos'],
                  cutoff=20,
                  weights=[33.33, 33.33, 33.33],
                  df=getdf("../data/Clean_Data_short.xlsx"),
                  fname="../../data/Clean_Data.xlsx")

    m.run()
    print(m.string_output())
