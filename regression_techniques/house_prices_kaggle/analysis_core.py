from os import path

import pandas as pd

# Print or save information about a dataframe

def df_info(dataframe, out_file=None, out_csv=False):
    if out_csv:
        out_df = pd.DataFrame(
columns=['Column',
 'Distinct Value Count',
 'Percent Nulls',
 'F Type'],
 index=None,
)
        for col in dataframe.columns:
            out_df = out_df.append({'Column':col, 
                'F Type':dataframe[col].ftype,
                'Distinct Value Count':dataframe[col].nunique(),
                'Percent Nulls':100*(
dataframe[col].isnull().sum()/len(
dataframe[col],
))}, ignore_index=True
)
        out_df.to_csv(out_file)
        return out_df
    else:
        for col in dataframe.columns:
            print(
"Column: {0}\tData Type: {1}\tDistinct Value count: {2}\tPercent Nulls: {3}\tF Type: {4}\n".format(
                col, 
                dataframe[col].dtype, 
                dataframe[col].nunique(), 
                100*(dataframe[col].isnull().sum()/len(dataframe[col])),
                dataframe[col].ftype))
