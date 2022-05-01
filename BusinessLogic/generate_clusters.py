import BusinessLogic.data_processing as dp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def generate_clusters():

    kmeans = KMeans(
        init="k-means++",
        n_clusters=4,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    scaled_data = dp.preprocess_credit_data()
    label = kmeans.fit(scaled_data)

    print(label)

generate_clusters()