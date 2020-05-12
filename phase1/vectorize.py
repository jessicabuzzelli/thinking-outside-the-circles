import pandas as pd
import numpy as np
from itertools import compress
from math import floor, isnan


class Vectorize:

    def __init__(self, level, choices, obj, segment, fields, natl_acct, field_options, cutoff, weights, df, fname):
        self.obj = obj  # either "Identify core products" or "Identify products to cut"
        self.segment = segment
        self.fname = fname
        self.field_options = field_options
        self.cutoff = cutoff  # % for core or cut (depending on self.obj)
        self.natl_acct = natl_acct  # true if ignore SKUs that serve any national account(s) at any warehouse(s)

        self.level = 'legacy_division_cd' if level == 'warehouse' else 'legacy_system_cd'

        self.selected_fields = list(compress(field_options, [*fields]))
        self.weights = np.array(list(filter(lambda num: num != 0, weights)))

        self.df, self.choices = self.format_df(df, choices)

        # initialize max/min dictionaries for later use to scale attributes into a [0,1] range
        self.maxes = self.df.max(axis=0).to_dict()
        self.mins = self.df.min(axis=0).to_dict()

        # each unique entity (wh or region, depends on self.level) to their list of products
        self.level_to_prod = {choice: self.df[self.df[self.level] == choice]['legacy_product_cd'].unique() for
                              choice in self.choices}

        # get all warehouse/region to product combinations, used later for vectors
        # inefficient method but not a big time sink for now -- will scale poorly
        self.combs = []
        for w in self.choices:
            for p in self.level_to_prod[w]:
                self.combs += [(w, p)]

    def format_df(self, df, choices):
        df = df[df['segment'] == self.segment]

        # get rid of rows that don't map to the selected regions/warehouses
        if choices[0] != 'All':
            for option in df[self.level].unique():
                if option not in choices:
                    df = df[df[self.level] != option]

        choices = choices if choices[0] != 'All' else df[self.level].unique()

        if self.natl_acct:
            # filter out rows where customer's national account = True
            df = df[df['national_acct_flag'] == 'N']

        return df, choices

    def get_mappings(self):
        # goal: a list of dictionaries, each dictionary used to get vector components (if applicable) and output stats

        needed = ['sales_6_mos',
                  'cogs_6mos',
                  'qty_6mos',
                  'picks_6mos',
                  'net_oh_$']

        self.vector_dicts = {}

        for arg in needed:
            self.vector_dicts[arg] = {(w, p):
                                          self.df[(self.df[self.level] == w) & (self.df['legacy_product_cd'] == p)][
                                              arg].mean()
                                      for (w, p) in self.combs}

        self.vector_dicts['customers_per_product'] = {(w, p):
            len(
                self.df[(self.df[self.level] == w)
                        & (self.df['legacy_product_cd'] == p)]
                ['legacy_customer_cd'].unique())
            for (w, p) in self.combs}

        self.vector_dicts['turn_6mos'] = {(w, p):
                                              self.vector_dicts['cogs_6mos'][w, p] / self.vector_dicts['net_oh_$'][w, p]
                                              if self.vector_dicts['net_oh_$'][w, p] != 0
                                              else 0
                                          for (w, p) in self.combs}

        self.vector_dicts['profit_6mos'] = {(w, p):
                                                self.vector_dicts['sales_6_mos'][w, p] - self.vector_dicts['cogs_6mos'][
                                                    w, p]
                                            for (w, p) in self.combs}

        self.vector_dicts['margin_%'] = {(w, p): 100 * self.vector_dicts['profit_6mos'][w, p] /
                                                 self.vector_dicts['cogs_6mos'][w, p]
        if self.vector_dicts['cogs_6mos'] != 0
        else 0
                                         for (w, p) in self.combs}

        for key in self.selected_fields:
            if key in ['margin', 'turn_6mos', 'profit_6mos', 'customers_per_product']:
                self.maxes[key] = max(self.vector_dicts[key].values())
                self.mins[key] = min(self.vector_dicts[key].values())

    def get_vectors(self):
        # adds column for scaled attributes to dataset for optional user reconciliation
        for s in self.field_options:
            self.df.loc[:, s + '_scaled'] = 0

        self.df.loc[:, 'vector'] = 0

        # get a score based off of a modified vector magnitude formula in order to rank SKUs at each warehouse/division
        self.wp_to_vectorscore = {}

        for w in self.level_to_prod.keys():
            for p in self.level_to_prod[w]:
                # find rows corresponding to the (warehouse/division, product) pair
                idxs = self.df[(self.df['legacy_division_cd'] == w) & (self.df['legacy_product_cd'] == p)].index

                vec = []

                # create vectors from dictionaries created in self.get_mappings
                for key in self.field_options:
                    try:
                        # scale down the vector component to [0,1]
                        scaled = (self.vector_dicts[key][w, p]
                                  + abs(self.mins[key])) \
                                 / (abs(self.maxes[key]) + abs(self.mins[key]))

                    except ZeroDivisionError:
                        scaled = 0
                    if key in self.selected_fields:
                        vec.append(scaled)

                    # update dataframe with scaled attribute for optional user reconciliation
                    for idx in idxs:
                        self.df.loc[idx, key + '_scaled'] = scaled

                self.wp_to_vectorscore[w, p] = 0 if vec == [] else self.norm(vec)

                # add vector to dataframe for optional user reconciliation
                for idx in idxs:
                    self.df.loc[idx, 'vector'] = self.wp_to_vectorscore[w, p]

    def norm(self, vec):
        # we've defined a vector for each (warehouse/division, product) pair, now we reduce down to a single number
        vec = np.array(vec)
        length = len(vec)

        try:
            return sum((vec * self.weights) ** length) ** (1 / length)

        except ZeroDivisionError:
            return 0

        except ValueError:
            # handles negative roots, but shouldn't come up since all values scales to >= 0
            return -(-sum(np.sign(vec) * (np.abs(vec * self.weights) ** length)) ** (1 / length))

    def reshape(self):
        if self.obj == 'Identify core products':
            self.cutoff = 100 - self.cutoff
            self.targetname = 'new core'
            self.nontargetname = 'new noncore'

        else:
            self.targetname = 'remove'
            self.nontargetname = 'keep'

        self.target_at_level = {}
        self.nontarget_at_level = {}

        for w in self.choices:
            # ignores duplicate (w, p) pairs when finding # of core and # of noncore
            sorted_df = self.df[self.df[self.level] == w][['vector', 'legacy_product_cd']].sort_values(by='vector')
            idxs = sorted_df.drop_duplicates(subset='legacy_product_cd', inplace=False, ignore_index=False).index
            duplicates = sorted_df[sorted_df.duplicated(subset='legacy_product_cd')]['legacy_product_cd']

            cutoffidx = int(floor(len(idxs) * self.cutoff / 100))

            self.target_at_level[w] = self.df.loc[idxs[cutoffidx:], 'legacy_product_cd'].values.tolist()
            self.nontarget_at_level[w] = self.df.loc[idxs[:cutoffidx], 'legacy_product_cd'].values.tolist()

            self.df.loc[idxs[:cutoffidx], self.targetname] = 0
            self.df.loc[idxs[cutoffidx:], self.targetname] = 1

            for index, product in duplicates.items():
                if product in self.target_at_level[w]:
                    self.df.loc[index, self.targetname] = 1
                else:
                    self.df.loc[index, self.targetname] = 0

    def output_stats(self):
        if self.obj == 'Identify core products':
            oldcore = self.df[self.df['core_item_flag'] == 'Y']
            oldmean = oldcore.mean(axis=0)
            newcore = self.df[self.df['new core'] == 1]
            newmean = newcore.mean(axis=0)
            out = (100 * (newmean - oldmean) / oldmean).to_dict()
            out['num'] = 100 * (newcore.shape[0] - oldcore.shape[0]) / oldcore.shape[0]

        else:
            oldnoncore = self.df[self.df['core_item_flag'] == 'N']
            oldmean = oldnoncore.mean(axis=0)
            cut = self.df[self.df['remove'] == 1]
            newmean = cut.mean(axis=0)
            out = (100 * (oldmean - newmean) / oldmean).to_dict()
            out['num'] = 100 * (oldnoncore.shape[0] - cut.shape[0]) / oldnoncore.shape[0]

            cut = self.df[self.df['remove'] == 1]

            n_cut = len(cut['legacy_product_cd'].unique())

            out['cut but core'] = 100 * self.df[(self.df['remove'] == 1) & (self.df['core_item_flag'] == 'Y')].shape[0] \
                                  / self.df[self.df['core_item_flag'] == 'Y'].shape[0]

            out['buy cut'] = 100 * len(cut['legacy_customer_cd'].unique()) / n_cut

            n_cust = 100 * len(self.df['legacy_customer_cd'].unique())
            out['only buy cut'] = (n_cust - len(self.df[self.df['remove'] == 0]['legacy_customer_cd'].unique())) \
                                   / out['buy cut']

            sc1 = (1 + self.mins['customers_per_product']) / self.maxes['customers_per_product']
            out['1 cust'] = 100 * len(self.df[(self.df['remove'] == 1) &
                                        (self.df['customers_per_product_scaled'] == sc1)]['legacy_product_cd'].unique()) / n_cut

            out['above 25'] = 100 * len(self.df[(self.df['remove'] == 1) & (self.df['margin_%'] > .25)]
                                  ['legacy_product_cd'].unique()) / n_cut

            out['ohi'] = cut['net_oh_$'].sum()
            out['profit'] = cut['sales_6_mos'].sum() - cut['cogs_6mos'].sum()
            out['fcf'] = out['ohi'] - out['profit']

        return out

    def string_output(self):
        results = self.output_stats()

        inorder = [results[k] for k in ['num',
                                        'customers_per_product_scaled',
                                        'profit_6mos_scaled',
                                        'turn_6mos_scaled']]
        inorder += [results['turn_6mos_scaled'] * results['margin_%']]
        inorder += [results[k] for k in ['picks_6mos_scaled',
                                         'net_oh_$',
                                         'dioh',
                                         'margin_%',
                                         'sales_6_mos_scaled']]

        updown = ['up' if n >= 0 else 'down' for n in inorder]

        fmt = []
        for x, y in zip(updown, inorder):
            fmt += [x]
            fmt += [float(round(y, 2))]

        string = """
        S U M M A R Y   S T A T I S T I C S:
        ------------------------------------
        
        As calculated by (new core - old core) * 100 / old core: 
        (i.e. new core classification's improvement over the existing core)
        
        - # of core SKUs: {} by {}%
        - Mean # of customers per SKU: {} by {}%
        - Mean profit: {} by {}%
        - Mean inventory turns: {} by {}%
        - Mean turn-and-earn: {} by {}%
        - Mean picks: {} by {}%
        - Mean on-hand inventory: {} by {}%
        - Mean days inventory on-hand: {} by {}%
        - Mean margin %: {} by {}%
        - Mean sales ($s): {} by {}%
        """.format(*fmt)

        if self.obj == 'Identify core products':
            return string

        else:
            fmt = [round(results[k], 2) for k in ['cut but core', 'buy cut', 'only buy cut', '1 cust', 'above 25']]
            fmt2 = [round(results[k], 2) for k in ['ohi', 'profit', 'fcf']]

            string += """
        R I S K   A N A L Y S I S:
        --------------------------
        
        - % of rationalized SKUs that were previously considered core: {}%
        - % of customers that purchase >= 1 rationalized SKU(s): {}%
            - % of those customers who only bought rationalized SKUs: {}%
        - % of rationalized SKUs bought by only 1 customer: {}%
        - % of rationalized SKUs that had a margin >25%: {}%
        
        V A L U A T I O N:
        ------------------
        Segment: Packaging
          
        Total reduction in on-hand inventory: ${}
        Total reduction in profit: ${}
        --> Total increase in free cash flow: ${}
        """.format(*fmt, *fmt2)

        return string

    def run(self):
        self.get_mappings()
        self.get_vectors()
        self.reshape()


class EnterpriseClassifications:
    def __init__(self, *kwargs):
        self.warehouse_model = Vectorize('warehouse', ['All'], *kwargs)
        self.region_model = Vectorize('region', ['All'], *kwargs)

    def run(self):
        self.warehouse_model.run()
        self.region_model.run()

        self.targetname = self.warehouse_model.targetname

        self.df = pd.concat([self.warehouse_model.df,
                             self.region_model.df[self.targetname]], axis=1)
        self.df.columns = [*self.warehouse_model.df.columns[:-1],
                           'warehouse ' + self.targetname,
                           'region ' + self.targetname]

        self.overall_core()

    def overall_core(self):
        self.df.loc[:, 'enterprise ' + self.targetname] = self.df[['warehouse ' + self.targetname,
                                                                   'region ' + self.targetname]].sum(axis=1)
        self.df.loc[:, 'enterprise ' + self.targetname] = self.df['enterprise ' + self.targetname] \
            .map(lambda x: 1 if x >= 1 else 0)

        self.warehouse_model.df = self.df

    def string_output(self):
        return self.warehouse_model.string_output()
