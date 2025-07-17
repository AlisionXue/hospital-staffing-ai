import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. 
df = pd.read_csv('data/hospital_staffing_summary.csv')

# 2. 
features = ['fte_doctors', 'treatment_count', 'patient_days', 'total_billed_amount', 'efficiency']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# 3. 
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 4. 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

# 5. 
plt.figure(figsize=(10, 6))
for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}', s=100)

plt.title('KMeans Clustering (K=3) with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)

# Save the plot image to the docs folder
plt.savefig('docs/week3_kmeans_pca_plot.png')
plt.show()

# 6. Save the data with cluster labels to a new CSV file
df.to_csv('data/hospital_week3_clustered.csv', index=False)
