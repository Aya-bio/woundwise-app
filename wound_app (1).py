import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

st.set_page_config(page_title="HealMate AI", layout="centered")
st.title("HealMate - النظام الذكي لعلاج الجروح")

st.markdown("ادخل بيانات الحالة ليقترح الذكاء الاصطناعي العلاج المناسب ويشخص الحالة")

temp = st.number_input("درجة الحرارة", 34.0, 42.0, step=0.1)
ph = st.number_input("pH", 4.0, 9.0, step=0.1)
moisture = st.slider("رطوبة الجرح", 0, 100)
infection = st.selectbox("عدوى؟", ["Yes", "No"])
diabetic = st.selectbox("مريض سكري؟", ["Yes", "No"])
wound_type = st.selectbox("نوع الجرح", ["Acute", "Chronic", "Surgical", "Minor"])
healing = st.selectbox("الجرح يلتئم بسرعة؟", ["Yes", "No"])
gender = st.selectbox("النوع", ["Male", "Female"])
age = st.slider("العمر", 1, 100)

df = pd.DataFrame([{
    "Temperature": temp,
    "pH": ph,
    "Moisture": moisture,
    "Infection": 1 if infection == "Yes" else 0,
    "Diabetic": 1 if diabetic == "Yes" else 0,
    "Wound Type": ["Acute", "Chronic", "Surgical", "Minor"].index(wound_type),
    "Healing Fast": 1 if healing == "Yes" else 0,
    "Gender": 1 if gender == "Male" else 0,
    "Age": age
}])

@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'Temperature': np.random.normal(37.0, 0.8, size=n),
        'Moisture': np.random.randint(30, 100, size=n),
        'pH': np.round(np.random.uniform(6.0, 8.0, size=n), 2),
        'Diabetic': np.random.randint(0, 2, size=n),
        'Age': np.random.randint(20, 80, size=n),
        'Gender': np.random.randint(0, 2, size=n),
        'Infection': np.random.randint(0, 2, size=n),
        'Wound Type': np.random.randint(0, 4, size=n)
    })
    df['Healing Fast'] = ((df['Diabetic'] == 0) & (df['Infection'] == 0)).astype(int)
    def assign_treatment(row):
        if row['Diabetic'] == 1 and row['Infection'] == 1 and row['pH'] < 6.8:
            return "Patch with Stem Cells + Antibiotic"
        elif row['Healing Fast'] == 1 and row['Temperature'] < 37.5:
            return "Patch with Collagen only"
        elif row['Wound Type'] == 2 and row['Moisture'] > 70:
            return "Patch with Antimicrobial + GF"
        elif row['pH'] > 7.5:
            return "Patch with Stem Cells only"
        else:
            return "Patch with Collagen + Antibiotic"
    df['Recommended Treatment'] = df.apply(assign_treatment, axis=1)
    return df

@st.cache_resource
def load_model():
    data = generate_data()
    X = data.drop("Recommended Treatment", axis=1)
    y = data["Recommended Treatment"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
    model.fit(X_scaled, y_encoded)
    return model, scaler, le, X.columns, y

model, scaler, le, feature_cols, full_labels = load_model()

if st.button("اقترح العلاج"):
    X_input_scaled = scaler.transform(df[feature_cols])
    pred = model.predict(X_input_scaled)
    treatment = le.inverse_transform(pred)[0]
    st.success(f"✅ العلاج المقترح: {treatment}")

    # تشخيص الحالة
    risk_score = 0
    if df["Infection"][0] == 1:
        risk_score += 1
    if df["Diabetic"][0] == 1:
        risk_score += 1
    if df["pH"][0] < 6.8 or df["pH"][0] > 7.8:
        risk_score += 1

    if risk_score == 0:
        status = "Stable"
        color = "green"
    elif risk_score == 1:
        status = "Moderate"
        color = "orange"
    else:
        status = "Critical"
        color = "red"

    st.markdown(f"<h4 style='color:{color}'>🩺 حالة الجرح: {status}</h4>", unsafe_allow_html=True)

    # رسم بياني بعد النتيجة
    chart_data = pd.Series(full_labels).value_counts()
    fig, ax = plt.subplots()
    chart_data.plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)
    
