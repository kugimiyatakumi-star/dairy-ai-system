import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

st.set_page_config(page_title="乳製品AIシステム", layout="wide")

st.title("🐄 乳製品AIシステム v1")

# アップロード
uploaded_file = st.file_uploader("📂 Excelファイルをアップロード", type=["xlsx"])

model = None

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.subheader("📊 データ確認")
    st.dataframe(df)

    target_column = st.selectbox("🎯 予測したい列を選択", df.columns)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if st.button("🚀 AIを学習させる"):
        model = RandomForestRegressor()
        model.fit(X, y)

        joblib.dump(model, "model.pkl")
        st.success("✅ 学習完了 & モデル保存完了")

# 既存モデル読み込み
if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
    st.success("📦 保存済みモデル読み込み済み")

if model is not None:
    st.subheader("🔮 予測モード")

    input_data = []

    if uploaded_file is not None:
        for col in X.columns:
            value = st.number_input(f"{col}", value=0.0)
            input_data.append(value)

        if st.button("📈 予測実行"):
            prediction = model.predict([input_data])
            st.success(f"予測結果：{prediction[0]}")
