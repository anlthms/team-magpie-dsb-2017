import pandas as pd

predictions = ['predictions.stage2.master.val10.4895.csv','predictions.stage2.mse.val10.4867.csv']
dfs = []
for p in predictions:
    df = pd.read_csv(p)
    dfs.append(df)

dfc = pd.concat(dfs)
mean = dfc.groupby(dfc.id)['cancer'].mean().reset_index()
mean.to_csv('predictions_average.csv',header=True,columns=['id','cancer'],index=False)
