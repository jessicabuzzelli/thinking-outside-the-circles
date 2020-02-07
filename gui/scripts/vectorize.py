import pandas as pd


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


class Vectorize:
    def __init__(self, wh, segment, fields, field_options, cutoff, df):
        self.wh = wh
        self.segment = segment
        self.field_options = field_options
        self.fields = fields
        self.cutoff = cutoff
        self.df = df[df['segment'] == self.segment]

        self.wh_to_prod = self.get_wh_to_prod()
        self.dict_list = self.get_dict_list()

    def get_wh_to_prod(self):
        if self.wh == 'All':
            warehouses = self.df.legacy_division_cd.unique()
        else:
            warehouses = [int(self.wh)]

        return {wh: self.df[self.df['legacy_division_cd'] == wh]['legacy_product_cd'].unique() for wh in warehouses}

    def get_dict_list(self):
        dict_list = []

        for field in self.fields[2:]:
            dict_list.append({})

        for w in self.wh:
            for p in self.wh_to_prod[w]:
                self.df[self.df['legacy_division_cd'] == wh & self.df['legacy_product_cd'] == p]



if __name__ == '__main__':
    Vectorize(19, 'Facility Solutions', [1, 1, 1], ['turn-and-earn', 'profit', 'cogs_6mos], 20, getdf('Clean_Data_short.xlsx'))
