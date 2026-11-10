# -*- coding: utf-8 -*-
"""
酒店预订取消预测 - Streamlit 交互式网页应用
运行方式：  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# 页面配置
# ----------------------------------------------------------------------------
st.set_page_config(page_title="酒店预订取消预测", page_icon="🏨", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
       "#008300", "#6250d6", "#e34948"]

# ----------------------------------------------------------------------------
# 加载数据与模型（缓存）
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "data", "hotels_clean.csv"))
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    return df

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE, "models", "best_model.pkl"))
    meta = joblib.load(os.path.join(BASE, "models", "meta.pkl"))
    return model, meta

df = load_data()
model, meta = load_model()
cat_cols = meta["categorical_cols"]
num_cols = meta["numeric_cols"]
top_countries = meta["top_countries"]

# ----------------------------------------------------------------------------
# 侧边栏
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🏨 酒店预订分析")
    st.markdown("**数据集**: Hotel Booking Demand")
    st.markdown("**样本量**: 119,390 条预订记录")
    st.markdown("**最优模型**: XGBoost (AUC ≈ 0.943)")
    st.divider()
    st.markdown("**课程**: 商务大数据案例分析 — 期中汇报")
    st.markdown("**来源**: Antonio, Almeida & Nunes (2019)")

# ----------------------------------------------------------------------------
# Tab 1: 数据概览
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 数据概览", "🔍 交互式探索", "🎯 取消预测", "📈 模型对比"])

with tab1:
    st.header("数据概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总预订量", f"{len(df):,}")
    c2.metric("整体取消率", f"{df['is_canceled'].mean():.1%}")
    c3.metric("平均房价 ADR", f"€{df['adr'].mean():.0f}")
    c4.metric("平均提前预订天数", f"{df['lead_time'].mean():.0f} 天")

    st.subheader("关键发现")
    col1, col2 = st.columns(2)
    with col1:
        g = df.groupby("hotel")["is_canceled"].mean().reset_index()
        g["is_canceled"] = (g["is_canceled"] * 100).round(1)
        fig = px.bar(g, x="hotel", y="is_canceled", color="hotel",
                     color_discrete_sequence=CAT[:2], text="is_canceled",
                     labels={"hotel": "酒店类型", "is_canceled": "取消率 (%)"})
        fig.update_layout(showlegend=False, yaxis_title="取消率 (%)",
                          xaxis_title="", yaxis_range=[0, 45])
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        g = df.groupby("deposit_type")["is_canceled"].mean().reset_index()
        g["is_canceled"] = (g["is_canceled"] * 100).round(1)
        g = g.sort_values("is_canceled")
        fig = px.bar(g, x="is_canceled", y="deposit_type", orientation="h",
                     color_discrete_sequence=[CAT[0]], text="is_canceled",
                     labels={"is_canceled": "取消率 (%)", "deposit_type": "押金类型"})
        fig.update_layout(showlegend=False, xaxis_title="取消率 (%)",
                          yaxis_title="", xaxis_range=[0, 105])
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("取消率的月度季节性")
    order = ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]
    g = df.groupby("arrival_date_month")["is_canceled"].mean().reindex(order).reset_index()
    g["is_canceled"] = (g["is_canceled"] * 100).round(1)
    fig = px.line(g, x="arrival_date_month", y="is_canceled", markers=True,
                  color_discrete_sequence=[CAT[0]],
                  labels={"arrival_date_month": "入住月份", "is_canceled": "取消率 (%)"})
    fig.update_traces(line=dict(width=2.5))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 2: 交互式探索
# ----------------------------------------------------------------------------
with tab2:
    st.header("交互式数据探索")
    f1, f2, f3, f4 = st.columns(4)
    hotel_f = f1.selectbox("酒店类型", ["全部"] + sorted(df["hotel"].unique()))
    seg_f = f2.selectbox("客源渠道", ["全部"] + sorted(df["market_segment"].unique()))
    dep_f = f3.selectbox("押金类型", ["全部"] + sorted(df["deposit_type"].unique()))
    ctype_f = f4.selectbox("客户类型", ["全部"] + sorted(df["customer_type"].unique()))

    sub = df.copy()
    if hotel_f != "全部":
        sub = sub[sub["hotel"] == hotel_f]
    if seg_f != "全部":
        sub = sub[sub["market_segment"] == seg_f]
    if dep_f != "全部":
        sub = sub[sub["deposit_type"] == dep_f]
    if ctype_f != "全部":
        sub = sub[sub["customer_type"] == ctype_f]

    c1, c2, c3 = st.columns(3)
    c1.metric("筛选后样本量", f"{len(sub):,}")
    c2.metric("筛选后取消率", f"{sub['is_canceled'].mean():.1%}")
    c3.metric("平均 ADR", f"€{sub['adr'].mean():.0f}")

    col1, col2 = st.columns(2)
    with col1:
        # lead_time 分箱取消率
        sub2 = sub.copy()
        sub2["lead_bin"] = pd.cut(sub2["lead_time"],
                                  bins=[-1, 7, 30, 90, 180, 365, np.inf],
                                  labels=["≤7天", "8-30天", "31-90天", "91-180天", "181-365天", ">365天"])
        g = sub2.groupby("lead_bin", observed=True)["is_canceled"].mean().reset_index()
        g["is_canceled"] = (g["is_canceled"] * 100).round(1)
        fig = px.bar(g, x="lead_bin", y="is_canceled", color_discrete_sequence=[CAT[1]],
                     text="is_canceled",
                     labels={"lead_bin": "提前预订天数", "is_canceled": "取消率 (%)"})
        fig.update_layout(showlegend=False, yaxis_title="取消率 (%)")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        g = sub.groupby("country")["is_canceled"].agg(["mean", "count"]).reset_index()
        g.columns = ["country", "rate", "count"]
        g = g[g["count"] > 200].sort_values("rate", ascending=False).head(10)
        g["rate"] = (g["rate"] * 100).round(1)
        fig = px.bar(g, x="rate", y="country", orientation="h",
                     color_discrete_sequence=[CAT[2]], text="rate",
                     labels={"rate": "取消率 (%)", "country": "来源国家"})
        fig.update_layout(showlegend=False, xaxis_title="取消率 (%)",
                          yaxis_title="", yaxis=dict(categoryorder="total ascending"))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("ADR 分布（对数尺度）")
    fig = px.histogram(sub[sub["adr"] > 0], x=np.log1p(sub[sub["adr"] > 0]["adr"]),
                       nbins=60, color_discrete_sequence=[CAT[0]],
                       labels={"x": "ADR（log 尺度）"})
    fig.update_layout(showlegend=False, yaxis_title="频数")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 3: 取消预测
# ----------------------------------------------------------------------------
with tab3:
    st.header("🎯 单条预订取消概率预测")
    st.caption("输入一条预订的特征，模型将实时预测该预订被取消的概率（基于 XGBoost）。")

    c1, c2, c3 = st.columns(3)
    hotel = c1.selectbox("酒店类型", ["City Hotel", "Resort Hotel"],
                         format_func=lambda x: "城市酒店" if x == "City Hotel" else "度假酒店")
    country = c2.selectbox("来源国家", top_countries + ["OTH"], index=top_countries.index("PRT"))
    market = c3.selectbox("客源渠道", sorted(df["market_segment"].unique()))

    c1, c2, c3 = st.columns(3)
    deposit = c1.selectbox("押金类型", sorted(df["deposit_type"].unique()),
                           index=list(sorted(df["deposit_type"].unique())).index("No Deposit"))
    customer = c2.selectbox("客户类型", sorted(df["customer_type"].unique()))
    meal = c3.selectbox("膳食套餐", sorted(df["meal"].unique()))

    c1, c2, c3 = st.columns(3)
    month = c1.selectbox("入住月份",
                         ["January", "February", "March", "April", "May", "June",
                          "July", "August", "September", "October", "November", "December"],
                         index=6)
    channel = c2.selectbox("分销渠道", sorted(df["distribution_channel"].unique()))
    repeated = c3.selectbox("是否回头客", [0, 1], format_func=lambda x: "是" if x else "否")

    c1, c2, c3 = st.columns(3)
    lead_time = c1.slider("提前预订天数", 0, 365, 60)
    adr = c2.number_input("日均房价 ADR (€)", 20.0, 500.0, 120.0, step=5.0)
    week_nights = c3.slider("周中入住晚数", 0, 15, 2)

    c1, c2, c3 = st.columns(3)
    weekend_nights = c1.slider("周末入住晚数", 0, 10, 1)
    adults = c2.slider("成人数", 1, 4, 2)
    children = c3.slider("儿童数", 0, 3, 0)

    c1, c2, c3 = st.columns(3)
    babies = c1.slider("婴儿数", 0, 2, 0)
    booking_changes = c2.slider("预订修改次数", 0, 10, 0)
    special_requests = c3.slider("特殊需求数", 0, 5, 0)

    c1, c2, c3 = st.columns(3)
    prev_cancel = c1.slider("历史取消次数", 0, 5, 0)
    wait_days = c2.slider("等待名单天数", 0, 200, 0)
    parking = c3.slider("停车位需求", 0, 3, 0)

    c1, c2 = st.columns(2)
    has_agent = c1.selectbox("是否通过中介预订", [0, 1], format_func=lambda x: "是" if x else "否")
    has_company = c2.selectbox("是否公司客户", [0, 1], format_func=lambda x: "是" if x else "否")

    # 构造完整特征向量
    total_nights = weekend_nights + week_nights
    total_guests = adults + children + babies
    row = {
        "hotel": hotel, "arrival_date_month": month, "meal": meal,
        "country": country, "market_segment": market,
        "distribution_channel": channel, "deposit_type": deposit,
        "customer_type": customer,
        "lead_time": lead_time, "arrival_date_week_number": 30,
        "arrival_date_day_of_month": 15, "stays_in_weekend_nights": weekend_nights,
        "stays_in_week_nights": week_nights, "adults": adults,
        "children": children, "babies": babies, "is_repeated_guest": repeated,
        "previous_cancellations": prev_cancel,
        "previous_bookings_not_canceled": 0, "booking_changes": booking_changes,
        "days_in_waiting_list": wait_days, "adr": adr,
        "required_car_parking_spaces": parking,
        "total_of_special_requests": special_requests,
        "total_nights": total_nights, "total_guests": total_guests,
        "has_agent": has_agent, "has_company": has_company,
    }
    X_input = pd.DataFrame([row])[cat_cols + num_cols]

    proba = model.predict_proba(X_input)[0][1]
    pred = "取消" if proba >= 0.5 else "不取消"

    st.divider()
    pc1, pc2 = st.columns(2)
    with pc1:
        color = CAT[1] if proba >= 0.5 else CAT[2]
        st.markdown(f"### 预测结果：**{pred}**")
        st.markdown(f"#### 取消概率：<span style='color:{color};font-size:32px'>{proba:.1%}</span>",
                    unsafe_allow_html=True)
    with pc2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100,
            number={"suffix": "%"},
            title={"text": "取消概率"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": CAT[1]},
                   "steps": [{"range": [0, 50], "color": "#e1e0d9"},
                             {"range": [50, 100], "color": "#cde2fb"}]}))
        fig.update_layout(height=250, margin=dict(t=40, b=10, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 4: 模型对比
# ----------------------------------------------------------------------------
with tab4:
    st.header("模型对比与评估")
    metrics = pd.DataFrame(meta["metrics"]).T
    st.dataframe(metrics.style.format("{:.4f}").highlight_max(axis=0, color="#cde2fb"),
                 use_container_width=True)

    st.subheader("ROC 曲线")
    st.image(os.path.join(BASE, "figures", "fig_roc.png"), use_container_width=True)

    st.subheader("XGBoost 特征重要性")
    st.image(os.path.join(BASE, "figures", "fig_importance.png"), use_container_width=True)

    st.subheader("混淆矩阵（XGBoost）")
    st.image(os.path.join(BASE, "figures", "fig_cm.png"), use_container_width=True)
