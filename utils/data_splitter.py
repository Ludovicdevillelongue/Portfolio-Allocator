import pandas as pd


class DataSplitter:
    def __init__(self, in_sample_period, out_sample_period):
        self.in_sample_period = pd.Timedelta(days=in_sample_period)
        self.out_sample_period = pd.Timedelta(days=out_sample_period)

    def split(self, data, end_date):
        in_sample_start_date = end_date - self.in_sample_period - self.out_sample_period
        in_sample_end_date = end_date - self.out_sample_period
        out_sample_start_date = in_sample_end_date + pd.Timedelta(days=1)
        out_sample_end_date = end_date
        in_sample_data = data.loc[in_sample_start_date:in_sample_end_date]
        out_sample_data = data.loc[out_sample_start_date:out_sample_end_date]
        return in_sample_data, out_sample_data