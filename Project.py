import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

wine_data = pd.read_csv("../input/winemag-data_first150k.csv")
wine_data[:3]

subset_data = wine_data[['country', 'variety', 'price', 'points', 'province']]
France_data = subset_data[subset_data['country']=='France']

sns.lmplot("price", "points", data=France_data, hue="variety", fit_reg=False, col='province', col_wrap=3)
plt.show()

Champagne_data = France_data.loc[(France_data['province']=='Champagne') & (France_data['price'] > 1000)]
Champagne_data

top_wines = subset_data.loc[(subset_data['points']>=99)]
top_wines[:3]

stats = top_wines.groupby(['variety']).describe()

stats['price']['mean'].plot.bar()
plt.show()

### Playing around with TF-IDF and text analysis

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

France_desc_data = wine_data[wine_data['country']=='France']
France_desc_data = France_desc_data[['description', 'points']]

top_France_desc_data = France_desc_data[France_desc_data['points']> 95]

top_France_desc_data[:3]

tf_idf_vectorizer = TfidfVectorizer(analyzer="word", use_idf=True, smooth_idf=True, ngram_range=(2, 3))
tf_idf_matrix = tf_idf_vectorizer.fit_transform(top_France_desc_data['description'])

clustering_model = KMeans(
    n_clusters=10,
    max_iter=100,
    precompute_distances="auto",
    n_jobs=-1
)

labels = clustering_model.fit_predict(tf_idf_matrix)

X = tf_idf_matrix.todense()


reduced_data = PCA(n_components=2).fit_transform(X)
# print reduced_data

fig, ax = plt.subplots()
for index, instance in enumerate(reduced_data):
    # print instance, index, labels[index]
    pca_comp_1, pca_comp_2 = reduced_data[index]
    ax.scatter(pca_comp_1, pca_comp_2)
plt.show()

