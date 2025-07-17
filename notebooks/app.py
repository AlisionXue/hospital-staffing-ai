import streamlit as st
import pandas as pd
from PIL import Image
from prophet import Prophet

# 页面标题
st.title("🏥 Hospital Staffing AI Dashboard")

# 1. 显示 Week 3 聚类图
st.header("📌 Week 3: KMeans Clustering")
try:
    cluster_img = Image.open("docs/week3_kmeans_pca_plot.png")
    st.image(cluster_img, caption="KMeans Clustering with PCA", use_container_width=True)
except Exception as e:
    st.warning("⚠️ 聚类图无法加载。请检查文件路径或名称。")

# 2. 显示 Week 4 模型预测图
st.header("📊 Week 4: Model Prediction vs Actual")
try:
    model_img = Image.open("docs/week4_model_test_plot.png")
    st.image(model_img, caption="Test Set Prediction vs Actual", use_container_width=True)
except Exception as e:
    st.warning("⚠️ 模型预测图无法加载。请检查文件路径或名称。")

# 3. 上传你自己的数据
st.header("📝 Upload Your Data")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        st.write("Your uploaded data:")
        st.dataframe(user_df.head())
    except Exception as e:
        st.error(f"读取上传数据失败: {e}")

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

    if len(train) >= 2:
        # 用户自定义预测季度数
        periods = st.slider("Select number of quarters to forecast:", min_value=1, max_value=8, value=4)

        model = Prophet()
        model.fit(train[["ds", "y"]])

        future = model.make_future_dataframe(periods=periods, freq="Q")
        forecast = model.predict(future)

        st.success(f"✅ Forecast for {selected_branch}")
        st.write(f"📈 Showing next {periods} quarters forecast.")
        st.line_chart(forecast[["ds", "yhat"]].set_index("ds"))
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods))
    else:
        st.warning("⚠️ Not enough training data (min 2 rows required).")
except Exception as e:
    st.error(f"⚠️ Failed to run Prophet forecast: {e}")
