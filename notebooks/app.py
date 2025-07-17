import streamlit as st
import pandas as pd
from PIL import Image
from prophet import Prophet
import plotly.express as px
import io

# 页面标题
st.title("🏥 Hospital Staffing AI Dashboard")

# 1. 显示 Week 3 聚类图
st.header("📌 Week 3: KMeans Clustering")
try:
    cluster_img = Image.open("docs/week3_kmeans_pca_plot.png")
    st.image(cluster_img, caption="KMeans Clustering with PCA", use_container_width=True)
except Exception as e:
    st.warning(f"⚠️ Failed to load Week 3 image: {e}")

# 2. 显示 Week 4 模型预测图
st.header("📊 Week 4: Model Prediction vs Actual")
try:
    model_img = Image.open("docs/week4_model_test_plot.png")
    st.image(model_img, caption="Test Set Prediction vs Actual", use_container_width=True)
except Exception as e:
    st.warning(f"⚠️ Failed to load Week 4 image: {e}")

# 3. 上传你自己的数据
st.header("📝 Upload Your Data")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.write("Your uploaded data:")
    st.dataframe(user_df.head())

# 4. Week 5 & 6: Prophet Forecast
st.header("🔮 Week 5 & 6: Prophet Forecast + Summary")
df_path = "data/hospital_week4_timeseries_lagged.csv"
try:
    df = pd.read_csv(df_path)
    df["ds"] = pd.to_datetime(df["quarter_dt"])
    df = df.rename(columns={"treatment_count": "y"})

    results = []

    for branch in df["hospital_branch"].unique():
        df_h = df[df["hospital_branch"] == branch]
        train = df_h[df_h["dataset_split"] == "train"]
        test = df_h[df_h["dataset_split"] == "test"]

        if len(train) >= 2 and not test.empty:
            model = Prophet()
            model.fit(train[["ds", "y"]])
            forecast = model.predict(test[["ds"]])

            pred = forecast["yhat"].values[0]
            true = test["y"].values[0]
            error = abs(true - pred)

            results.append({
                "Hospital Branch": branch,
                "True Value": true,
                "Predicted Value": round(pred, 2),
                "Error (Abs)": round(error, 2)
            })

    if results:
        summary_df = pd.DataFrame(results)

        # 展示表格
        st.subheader("🌟 Week 6: Summary Table")
        st.dataframe(summary_df)

        # 展示误差条形图
        fig_err = px.bar(summary_df, x="Hospital Branch", y="Error (Abs)", title="Prediction Error by Hospital")
        st.plotly_chart(fig_err)

        # 显示总体 MAE 和 RMSE
        mae = round(summary_df["Error (Abs)"].mean(), 2)
        rmse = round((summary_df["Error (Abs)"]**2).mean()**0.5, 2)
        st.write(f"✅ Mean Absolute Error (MAE): {mae}")
        st.write(f"✅ Root Mean Squared Error (RMSE): {rmse}")

        # 添加 CSV 下载按钮
        csv_buffer = io.StringIO()
        summary_df.to_csv(csv_buffer, index=False)
        st.download_button("📥 Download Summary CSV", csv_buffer.getvalue(), "week6_summary.csv", "text/csv")

    else:
        st.warning("⚠️ Not enough data across all branches to generate summary table.")

except Exception as e:
    st.error(f"⚠️ Failed to run Prophet forecast: {e}")
