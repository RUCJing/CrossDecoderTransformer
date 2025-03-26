import numpy as np
import pandas as pd
import re, os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_feather('data.feather')
data.rename(columns={
    '科室': 'department',
    '职称': 'title'
}, inplace=True)
data['注册时间'] = pd.to_datetime(data['注册时间'])
data['创建时间'] = pd.to_datetime(data['创建时间'])
data['注册时间天数'] = (data['创建时间'] - data['注册时间']).dt.days.abs()

columns_to_concat = [
    '服务人次(主页)', '好评率', '同行认可', '患者心意', '图文资讯价格',
    '关注人数', '态度非常好', '讲解很清楚', '回复很及时', '建议很有帮助', '注册时间天数'
]
for col in columns_to_concat:
    mean_value = data[col].mean()
    data[col].fillna(mean_value, inplace=True)

def generate_vector(x, quantile):
    qs = quantile
    vector = np.array([1 / (abs(x - q) + 1e-9) for q in qs])
    vector = vector / np.sum(vector)
    return vector
def concat_vector(row, cols, quantiles):
    quant = np.array([])
    for idx, col in enumerate(cols):
        vec = generate_vector(row[col], quantiles[idx])
        quant = np.hstack([quant, vec])
    return quant
def cdb_embedding(cols):
    quantiles = []
    for col in cols:
        kmedoids = KMedoids(n_clusters=10, metric='euclidean', init='k-medoids++', random_state=42)
        kmedoids.fit(samples[col].values.reshape(-1, 1))
        quantiles.append(kmedoids.cluster_centers_.reshape(1, -1).tolist()[0])
        print(quantiles)
    tqdm.pandas(desc='pandas bar')
    concat_vector_partial = partial(concat_vector, cols = cols, quantiles = quantiles)
    quant = data.progress_apply(concat_vector_partial, axis = 1)
    return quant
samples = data.sample(n=30000)

quant = cdb_embedding(columns_to_concat)
data['quant'] = quant

encoder = LabelEncoder()
data['id'] = encoder.fit_transform(data['id'])
data['department'] = encoder.fit_transform(data['department'])
data['title'] = encoder.fit_transform(data['title'])
data['id'] = data['id'] + 1
data['department'] = data['department'] + 1
data['title'] = data['title'] + 1

data['label'] = data['label'].map({'满意': 0, '不满意': 1})
columns_to_keep = ['id', 'text', 'Text', 'speaker', 'turn', 'department', 'title', 'quant', 'label']
df = data[columns_to_keep]
random_seed = 42
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=random_seed)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=random_seed)
train_df = train_df.reset_index(drop = True)
val_df = val_df.reset_index(drop = True)
test_df = test_df.reset_index(drop = True)
train_df.to_feather('train_df_cdt.feather')
val_df.to_feather('val_df_cdt.feather')
test_df.to_feather('test_df_cdt.feather')

