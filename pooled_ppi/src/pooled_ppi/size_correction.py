
import math
import numpy as np, pandas as pd, seaborn as sns

class SizeCorrection:
    def __init__(self, tokens, scores, polyfit=None):
        data_ = pd.DataFrame({'tokens': tokens, 'scores': scores}).sort_values('tokens').reset_index(drop=True)
        data_['tokens_sqrt'] = data_['tokens'].map(math.sqrt)
        data_['scores_min'] = data_['scores'].rolling(window=51, center=True).min()
        self.data = data_.query('scores_min == scores_min').reset_index(drop=True)

        print(len(data_.query('scores == scores')))
        print(len(data_.query('scores_min == scores_min')))

        if polyfit is None:
            self.predict_baseline = np.poly1d(np.polyfit(x=self.data.tokens_sqrt, y=self.data.scores_min, deg=1))
        else:
            self.predict_baseline = polyfit
        print(self.predict_baseline)

    def transform(self, tokens, scores):
        data_ = pd.DataFrame({'tokens': tokens, 'scores': scores})
        data_['tokens_sqrt'] = data_['tokens'].map(math.sqrt)
        data_['baseline'] = self.predict_baseline(data_.tokens_sqrt)
        data_['scores_corrected'] = data_['scores'] - data_['baseline']       
        return data_['scores_corrected']

    def transform_plot(self, tokens, scores):
        data_ = pd.DataFrame({'tokens': tokens, 'scores': scores})
        data_['tokens_sqrt'] = data_['tokens'].map(math.sqrt)
        data_['baseline'] = self.predict_baseline(data_.tokens_sqrt)
        data_['scores_corrected'] = data_['scores'] - data_['baseline']       
        sns.scatterplot(data_, x='tokens', y='scores')
        sns.lineplot(data=data_, x='tokens', y='baseline', color='tab:red')

def size_correction(tokens, scores):
    polyfit_ = np.poly1d([0.0044, 0.04]) # Approximate averages from mgen+yeast
    size_correction_ = SizeCorrection(tokens, scores, polyfit=polyfit_)
    return size_correction_.transform(tokens, scores)
