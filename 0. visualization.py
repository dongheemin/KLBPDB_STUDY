import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def bar_chart(data, feature, title):
    normal = data[data['HP_DX_YN'] == 0][feature].value_counts()
    hyper = data[data['HP_DX_YN'] == 1][feature].value_counts()

    df = pd.DataFrame([normal, hyper])
    df.index = ['NORMAL', 'HYPER']
    df.plot(kind='bar', stacked=True, title=title)

datas = pd.read_csv('./KLBPDB_HTN.csv')

print('data shape : ', datas.shape)
print('--------------------[dataset info]--------------------')
print(datas.info())
print('--------------------[missing info]--------------------')
print(datas.isnull().sum())
print('--------------------[describe info]--------------------')
print(datas.describe().T)

drop_datas = datas
for cols in drop_datas.columns:
    drop_datas = drop_datas[drop_datas[cols] != -1]

print('data shape : ', drop_datas.shape)
print('--------------------[drop dataset info]--------------------')
print(drop_datas.info())
print('--------------------[drop missing info]--------------------')
print(drop_datas.isnull().sum())
print('--------------------[drop describe info]--------------------')
print(drop_datas.describe().T)
#
# # visualization hypertension vs normal by SBP & DBP
datas['SBP_units'] = (datas['SBP'].fillna(100)*0.1).astype(int)*10
datas['DBP_units'] = (datas['DBP'].fillna(100)*0.1).astype(int)*10

drop_datas['SBP_units'] = (drop_datas['SBP'].fillna(100)*0.1).astype(int)*10
drop_datas['DBP_units'] = (drop_datas['DBP'].fillna(100)*0.1).astype(int)*10

bar_chart(datas, "SBP_units", "HYPERTENSION vs NORMAL by SBP_UNITS")
bar_chart(datas, "DBP_units", "HYPERTENSION vs NORMAL by DBP_UNITS")

bar_chart(drop_datas, "SBP_units", "Dropped NA HYPERTENSION vs NORMAL by SBP_UNITS")
bar_chart(drop_datas, "DBP_units", "Dropped NA HYPERTENSION vs NORMAL by DBP_UNITS")

# variable plot
'''
pair = sns.pairplot(datas, hue='HP_DX_YN', corner=True)
heat = sns.heatmap(datas.corr(), annot=True)
'''

plt.show()