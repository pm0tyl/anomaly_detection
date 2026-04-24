import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN

X, _ = make_blobs(
    n_samples=400,
    centers=[(30, 30), (70, 70)],
    cluster_std=5,
    random_state=42
)

anomalies = np.random.uniform(low=0, high=100, size=(20, 2))
X = np.vstack([X, anomalies])

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=15)
plt.title("Dane (CPU vs RAM)")
plt.xlabel("CPU usage (%)")
plt.ylabel("RAM usage (%)")
plt.show()


kmeans = KMeans(n_clusters=2, random_state=42)
labels_km = kmeans.fit_predict(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels_km, s=15)
plt.title("KMeans – grupowanie zachowań systemu")
plt.show()


db = DBSCAN(eps=6, min_samples=5)
labels_db = db.fit_predict(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels_db, s=15)
plt.title("DBSCAN – wykrywanie anomalii")
plt.show()


anomaly_points = X[labels_db == -1]

print(f"Liczba wykrytych anomalii: {len(anomaly_points)}")