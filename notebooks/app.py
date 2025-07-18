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
uploaded_file = st.file_uploader(
    "Upload a CSV file (same format as hospital_week4_timeseries_lagged.csv)",
    type="csv"
)
if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.write("Your uploaded data:")
    st.dataframe(user_df.head())

# 4. Week 5 & 6: Prophet Forecast + Summary
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

            future = model.make_future_dataframe(periods=4, freq='Q')
            forecast = model.predict(future)

            pred = forecast["yhat"].iloc[-1]
            true = test["y"].values[0]
            error = abs(true - pred)

            results.append({
                "Hospital Branch": branch,
                "True Value": true,
                "Predicted Value": round(pred, 2),
                "Error (Abs)": round(error, 2),
                "Forecast": forecast[["ds", "yhat", "yhat_upper", "yhat_lower"]]
            })

    if results:
        summary_df = pd.DataFrame([{
            "Hospital Branch": r["Hospital Branch"],
            "True Value": r["True Value"],
            "Predicted Value": r["Predicted Value"],
            "Error (Abs)": r["Error (Abs)"]
        } for r in results])

        # 展示表格
        st.subheader("🌟 Week 6: Summary Table")
        st.dataframe(summary_df)

        # 展示误差条形图（按误差大小排序）
        fig_err = px.bar(
            summary_df.sort_values(by="Error (Abs)", ascending=False),
            x="Hospital Branch", y="Error (Abs)", title="Prediction Error by Hospital"
        )
        fig_err.update_layout(yaxis_tickformat=".2f")
        st.plotly_chart(fig_err)

        # 展示真值 vs 预测值对比图
        st.subheader("📊 True vs Predicted Treatment Count")
        fig_comp = px.bar(
            summary_df.melt(id_vars="Hospital Branch", value_vars=["True Value", "Predicted Value"],
                            var_name="Type", value_name="Treatment Count"),
            x="Hospital Branch", y="Treatment Count", color="Type", barmode="group"
        )
        fig_comp.update_layout(yaxis_tickformat=".2f")
        st.plotly_chart(fig_comp)

        # 显示总体 MAE 和 RMSE
        mae = round(summary_df["Error (Abs)"].mean(), 2)
        rmse = round((summary_df["Error (Abs)"]**2).mean()**0.5, 2)
        col1, col2 = st.columns(2)
        col1.metric("📌 Mean Absolute Error (MAE)", mae)
        col2.metric("📌 Root Mean Squared Error (RMSE)", rmse)

        st.markdown(f"""
        **📈 模型评价指标说明：**  
        - **MAE（平均绝对误差）**：预测值与真实值差异的平均值。越小越好。  
        - **RMSE（均方根误差）**：对大误差更敏感。越小越好。
        """)

        # 显示每个医院的预测趋势
        with st.expander("📈 Forecast Trends for Each Branch"):
            for r in results:
                fig = px.line(r["Forecast"], x="ds", y=["yhat", "yhat_upper", "yhat_lower"],
                              title=f"Forecast Trend - {r['Hospital Branch']}",
                              line_shape="spline")
                fig.update_layout(xaxis_tickformat="%b %Y", yaxis_tickformat=".2f", hovermode="x unified")
                st.plotly_chart(fig)

    else:
        st.warning("⚠️ Not enough data across all branches to generate summary table.")

except Exception as e:
    st.error(f"⚠️ Failed to run Prophet forecast: {e}")
