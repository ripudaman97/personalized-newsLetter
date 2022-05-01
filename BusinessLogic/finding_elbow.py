from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

# loading data
credit_data = pd.read_csv("../raw_data.csv")

# normalizing and scaling data

normalized_credit_data = preprocessing.MinMaxScaler()
col_names = credit_data.columns
d = normalized_credit_data.fit_transform(credit_data)

scaled_credit_data = pd.DataFrame(d,columns=col_names)

arguments_kmeans = {
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **arguments_kmeans)
    kmeans.fit(scaled_credit_data)
    sse.append(kmeans.inertia_)


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

# Elbow can be found at 4