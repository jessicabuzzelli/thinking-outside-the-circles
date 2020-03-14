import pandas as pd
import numpy as np
from itertools import compress


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

    return df


class Vectorize:

    def __init__(self, level, choices, obj, segment, fields, field_options, cutoff, weights, df, fname):
        self.obj = obj  # either "Identify core products" or "Identify products to cut"
        self.segment = segment
        self.fname = fname
        self.cutoff = cutoff  # % core or % cut

        self.level = 'legacy_division_cd' if level == 'warehouse' else 'legacy_system_cd'  # if region

        self.selected_fields = list(compress(field_options, [*fields]))
        self.weights = np.array(list(filter(lambda num: num != 0, weights)))

        self.df, self.choices = self.format_df(df, choices)
        self.maxes = self.df.max(axis=0).to_dict()
        self.mins = self.df.min(axis=0).to_dict()

        # each unique entity (wh or region) based on level to their list of products
        self.level_to_prod = {choice: self.df[self.df[self.level] == choice]['legacy_product_cd'].unique() for
                              choice in self.choices}

        self.combs = []
        for w in self.choices:
            for p in self.level_to_prod[w]:
                self.combs += [(w, p)]

    def format_df(self, df, choices):
        df = df[df['segment'] == self.segment]

        # get rid of rows that don't map to one of the selected regions/warehouses
        if choices != ['All']:
            for option in df[self.level].unique():
                if option not in choices:
                    df = df[df[self.level] != option]

        choices = choices if choices != ['All'] else df[self.level].unique()

        df.loc[:, 'core_item_flag'] = df['core_item_flag'].map(lambda x: 0 if x == 'F' else 1)

        return df, choices

    def get_mappings(self):
        needed = []
        for field in self.selected_fields:
            if field in ['sales_6_mos', 'cogs_6mos', 'qty_6mos', 'picks_6mos', 'net_oh_$']:
                needed += [field]
            elif field in ['profit_6mos', 'margin']:
                needed += ['cogs_6mos', 'sales_6_mos']
            elif field == 'turn_6mos':
                needed += ['net_oh_$', 'cogs_6mos']
            elif field == 'customers_per_product':
                needed += ['legacy_customer_cd']
            else:
                pass

        needed = set(needed)

        needed_dicts = {}
        needed_cols = [x for x in needed
                       if x in ['sales_6_mos', 'cogs_6mos', 'qty_6mos', 'picks_6mos', 'net_oh_$']]

        if needed_cols:
            for arg in needed_cols:
                needed_dicts[arg] = {(w, p):
                                     self.df[(self.df[self.level] == w) & (self.df['legacy_product_cd'] == p)][arg].mean()
                                     for (w, p) in self.combs}

        if 'customers_per_product' in self.selected_fields:
            needed_dicts['customers_per_product'] = {(w, p):
                                                     len(self.df[(self.df[self.level] == w) & (self.df['legacy_product_cd'] == p)]['legacy_customer_cd'].unique())
                                                     for (w, p) in self.combs}
        if 'turn_6mos' in self.selected_fields:
            needed_dicts['turn_6mos'] = {(w, p):
                                         needed_dicts['cogs_6mos'][w, p] / needed_dicts['net_oh_$'][w, p]
                                         if needed_dicts['net_oh_$'][w, p] != 0
                                         else 0
                                         for (w, p) in self.combs}
        if 'profit_6mos' in self.selected_fields:
            needed_dicts['profit_6mos'] = {(w, p):
                                           needed_dicts['sales_6_mos'][w, p] - needed_dicts['cogs_6mos'][w, p]
                                           for (w, p) in self.combs}
        if 'margin' in self.selected_fields:
            needed_dicts['margin'] = {(w, p): 100 * needed_dicts['profit_6mos'][w, p] / needed_dicts['cogs_6mos'][w, p]
                                      if needed_dicts['cogs_6mos'] != 0
                                      else 0
                                      for (w, p) in self.combs}

        self.vector_dicts = {key: needed_dicts[key] for key in self.selected_fields}

        for key in self.selected_fields:
            if key in ['margin', 'turn_6mos', 'profit_6mos', 'customers_per_product']:
                self.maxes[key] = max(needed_dicts[key].values())
                self.mins[key] = min(needed_dicts[key].values())

    def get_vectors(self):
        for s in self.selected_fields:
            self.df.loc[:, s + '_scaled'] = 0

        self.df.loc[:, 'vector'] = 0

        self.wp_to_vectorscore = {}
        vecs = []

        for w in self.level_to_prod.keys():
            for p in self.level_to_prod[w]:

                idxs = self.df[(self.df['legacy_division_cd'] == w) & (self.df['legacy_product_cd'] == p)].index

                vec = []

                for key in self.selected_fields:
                    scaled = (self.vector_dicts[key][w, p] + abs(self.mins[key])) \
                             / (abs(self.maxes[key]) + abs(self.mins[key]))
                    vec.append(scaled)

                    for idx in idxs:
                        self.df.loc[idx, key + '_scaled'] = scaled

                self.wp_to_vectorscore[w, p] = 0 if vec == [] else self.norm(vec)

                for idx in idxs:
                    self.df.loc[idx, 'vector'] = self.wp_to_vectorscore[w, p]

                vecs.append(vec)

    def norm(self, vec):
        vec = np.array(vec)
        length = len(vec)

        try:
            return sum((vec * self.weights) ** length) ** (1 / length)

        except ZeroDivisionError:
            return 0

        except ValueError:
            # handles negative roots, but they shouldn't come up since all values scales to >= 0
            return -(-sum(np.sign(vec) * (np.abs(vec * self.weights) ** length)) ** (1 / length))

    def reshape(self, oneisall=False):
        if self.obj == 'Identify core products':
            self.cutoff = 100 - self.cutoff
            self.targetname = 'core'
            self.nontargetname = 'noncore'
        else:
            self.targetname = 'remove'
            self.targetname = 'keep'

        target_at_level = {}

        for w in self.choices:
            # counts multiple (w, p)'s as multiple SKUs -- needs to loop on p's to fix
            idxs = self.df[self.df[self.level] == w]['vector'].sort_values().index

            cutoffidx = int(len(idxs) * self.cutoff / 100)

            nontarget = self.df.loc[idxs[:cutoffidx], 'legacy_product_cd'].values
            target = self.df.loc[idxs[cutoffidx:], 'legacy_product_cd'].values

            target_at_level[w] = target.tolist()
            # print([str(x)[-3:] for x in target.tolist()])
            # print([str(x)[-3:] for x in nontarget.tolist()])

            self.df.loc[idxs[cutoffidx:], 'new_core'] = 0

            if oneisall == False:
                self.df.loc[idxs[:cutoffidx], 'new_core'] = 1

        if oneisall == True:
            targetprods = []
            for w in self.choices:
                targetprods += target_at_level[w]

            self.df.loc[:, 'new_core'] = 0
            for tp in targetprods:
                self.df.loc[self.df[self.df['legacy_product_cd'] == tp].index, 'new_core'] = 1

            # for p in self.df['legacy_product_cd'].unique():
            #     if p not in targetprods:
            #         print(p)

        self.df.to_excel('pls_be_results.xlsx')

    def run(self):
        self.get_mappings()
        self.get_vectors()
        # self.df.to_excel('pls_be_results.xlsx')
        self.reshape()


if __name__ == '__main__':
    m = Vectorize(weights=[33.33, 33.33, 33.33],
                  obj='Identify core products',
                  level='warehouse',
                  choices=['All'],
                  segment='Facility Solutions',
                  fields=[1, 1, 1],
                  field_options=['turn_6mos', 'profit_6mos', 'cogs_6mos'],
                  cutoff=20,
                  df=getdf("../data/Clean_Data_short.xlsx"),
                  fname="../data/Clean_Data_short.xlsx")
    m.run()
