import pandas as pd 


df = pd.read_csv("submission.csv")
#swap 2nd and 3rd columns
df = df[['index', 'emotion', 'category']]
df.to_csv("submission.csv", index=False)