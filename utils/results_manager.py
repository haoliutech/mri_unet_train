import pandas as pd
from collections import Counter, defaultdict
import os
from pathlib import Path


class ResultsManager(object):  
    """
    This will be used to manage results through the disk. Df saved to file
    created by the first time we add a result in which case it will read the column names to add
    """

    def __init__(self, data_path=None, fname=None, results_df_path=None, sort_columns=False):

        self.sort_columns = sort_columns

        if results_df_path:
            # If the file path is provided in full, then save and exit
            self.results_df_path = Path(results_df_path)
            return None

        # Otherwise use the default (./data/exp_results.csv)

        if data_path is None:
            data_path = Path.cwd() / 'data'
            data_path.mkdir(parents=True, exist_ok=True)
        else:
            data_path = Path(data_path)

        if fname is None:
            fname = 'results_df.csv'
            
        if results_df_path is None:
            self.results_df_path = data_path / fname
        else:
            self.results_df_path = results_df_path

        return None

    def add_to_results_df(self, **kwargs):

        df_columns = list(kwargs.keys())
        if self.sort_columns:
            df_columns.sort()
        results_df = self.load_results_df(df_columns)

        values = []
        for key_ in df_columns:
            value_ = kwargs.pop(key_, None)
            if value_ is None:
                value_ = ''    
            values.append(value_)

        row = pd.Series(values, index = df_columns)
        results_df = results_df.append(row, ignore_index=True)

        self.save_results_df(results_df)

    def save_results_df(self, results_df):
        # Save manual changes to the df
        results_df.to_csv(self.results_df_path, index=True)

    def load_results_df(self, df_columns=None):

        if os.path.isfile(self.results_df_path):
            df_results_ = pd.read_csv(self.results_df_path, index_col=0)
        else:
            if df_columns:
                df_results_ = pd.DataFrame(columns=df_columns)
            else:
                df_results_ = pd.DataFrame()

        return df_results_

    def get_results_df(self):

        if os.path.isfile(self.results_df_path):
            results_df = pd.read_csv(self.results_df_path, index_col=0)
            return results_df       

        else:
            print('No file exists')
            return pd.DataFrame()

    def display_results_df(self):

        results_df = self.get_results_df()
        display(results_df)

    def clear_results_df(self):
        # Start over (delete existing file if it exists)
        if os.path.isfile(self.results_df_path):
            os.remove(self.results_df_path)


# def create_sub_figure(tex_fname, ):

#     """
#     \begin{figure}[htp]
#     \centering
#     \begin{subfigure}[b]{0.24\textwidth}
#         \includegraphics[width=\linewidth]{./Figures/label_uncertainty/compare_labellers_3123.png}
#         %\caption{Contact angle with various pseudo dosages}
#         \label{fig:2µm_lines_CA_graph}
#     \end{subfigure}
#     \hfil
#     \caption{Contact angles on 2 $\mu$m Lines}\label{fig:2µm_lines_CA_graphs}
#     \end{figure}
#     """
