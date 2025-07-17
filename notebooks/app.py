import streamlit as st
import pandas as pd
from PIL import Image

# 页面标题
st.title("🏥 Hospital Staffing AI Dashboard")

# 1. 显示 Week 3 聚类图
st.header("📌 Week 3: KMeans Clustering")
cluster_img = Image.open("docs/week3_kmeans_pca_plot.png")
st.image(cluster_img, caption="KMeans Clustering with PCA", use_container_width=True)

# 2. 显示 Week 4 模型预测图
st.header("📊 Week 4: Model Prediction vs Actual")
model_img = Image.open("docs/week4_model_test_plot.png")
st.image(model_img, caption="Test Set Prediction vs Actual", use_container_width=True)

# 3. 上传你自己的数据
st.header("📝 Upload Your Data")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.write("Your uploaded data:")
    st.dataframe(user_df.head())
