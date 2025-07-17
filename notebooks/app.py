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

# 4. Week 5: Prophet Forecast
st.header("🔮 Week 5: Forecast with Prophet")

df_path = "data/hospital_week4_timeseries_lagged.csv"
try:
    df = pd.read_csv(df_path)
    df["ds"] = pd.to_datetime(df["quarter_dt"])
    df = df.rename(columns={"treatment_count": "y"})

    branches = df["hospital_branch"].unique().tolist()
    selected_branch = st.selectbox("Select hospital branch for Prophet forecast:", branches)

    df_h = df[df["hospital_branch"] == selected_branch]
    train = df_h[df_h["dataset_split"] == "train"]
    test = df_h[df_h["dataset_split"] == "test"]

    if len(train) >= 2 and not test.empty:
        from prophet import Prophet
        model = Prophet()
        model.fit(train[["ds", "y"]])

        forecast = model.predict(test[["ds"]])
        pred = forecast["yhat"].values[0]
        true = test["y"].values[0]

        st.success(f"✅ Prediction for {selected_branch}")
        st.write(f"True: {true}, Predicted: {pred:.2f}")
        st.line_chart(forecast[["ds", "yhat"]].set_index("ds"))
    else:
        st.warning("Not enough data for this branch to run Prophet.")
except Exception as e:
    st.error(f"⚠️ Failed to run Prophet forecast: {e}")

