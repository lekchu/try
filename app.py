import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Train a dummy model in-app (just for testing/demo)
def train_dummy_model():
    X = np.array([
        [25, 1, 2, 0, 1, 1, 2, 3, 2, 2, 1, 2, 2, 3, 1, 0],
        [30, 0, 1, 1, 0, 0, 1, 2, 2, 3, 3, 3, 1, 0, 0, 2],
        [22, 1, 0, 0, 2, 2, 1, 2, 1, 2, 0, 1, 2, 2, 2, 1]
    ])
    y = ['Mild', 'Severe', 'Moderate']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_dummy_model()

st.title("🧠 Postpartum Depression Risk Predictor")

st.header("👩 Personal Info")
age = st.slider("Age", 18, 45, 25)
employment = st.selectbox("Employment Status", ["Employed", "Unemployed"])
children = st.selectbox("Number of Children", [0, 1, 2, 3, "More"])
pregnancy = st.selectbox("Currently Pregnant?", ["Yes", "No"])
delivery = st.selectbox("Recently Given Birth?", ["Yes", "No"])
family_support = st.selectbox("Family Support Level", ["Low", "Medium", "High"])

# Encode inputs
emp = 1 if employment == "Employed" else 0
child = 4 if children == "More" else int(children)
preg = 1 if pregnancy == "Yes" else 0
deliv = 1 if delivery == "Yes" else 0
support = {"Low": 0, "Medium": 1, "High": 2}[family_support]

st.header("📝 EPDS Questionnaire")
def qbox(q, options, key): return st.selectbox(q, list(options.keys()), key=key)

questions = {
    "Q1": {"As much as I always could": 0, "Not quite so much now": 1, "Definitely not so much now": 2, "Not at all": 3},
    "Q2": {"As much as I ever did": 0, "Rather less than I used to": 1, "Definitely less than I used to": 2, "Hardly at all": 3},
    "Q3": {"No, never": 0, "Not very often": 1, "Yes, some of the time": 2, "Yes, most of the time": 3},
    "Q4": {"No, not at all": 0, "Hardly ever": 1, "Yes, sometimes": 2, "Yes, very often": 3},
    "Q5": {"No, not at all": 0, "No, not much": 1, "Yes, sometimes": 2, "Yes, quite a lot": 3},
    "Q6": {"No, I coped as ever": 0, "Coped quite well": 1, "Not coping sometimes": 2, "Not coping at all": 3},
    "Q7": {"No, not at all": 0, "Not very often": 1, "Yes, sometimes": 2, "Yes, most of the time": 3},
    "Q8": {"No, not at all": 0, "Not very often": 1, "Yes, quite often": 2, "Yes, most of the time": 3},
    "Q9": {"No, never": 0, "Only occasionally": 1, "Yes, quite often": 2, "Yes, most of the time": 3},
    "Q10": {"Never": 0, "Hardly ever": 1, "Sometimes": 2, "Yes, quite often": 3}
}

scores = []
for i, (q, opt) in enumerate(questions.items(), 1):
    response = qbox(f"{q}. Question {i}", opt, key=q)
    scores.append(opt[response])

# Final input vector
features = [age, emp, child, preg, deliv, support] + scores

if st.button("🔍 Predict Risk"):
    result = model.predict([features])[0]
    st.success(f"🎯 Predicted Risk: **{result}**")

    st.subheader("📊 Risk Level (for demo)")
    fig, ax = plt.subplots()
    ax.bar(["Risk"], [1 if result == "Mild" else 2 if result == "Moderate" else 3], color="salmon")
    ax.set_ylim([0, 3])
    ax.set_ylabel("Risk Level")
    st.pyplot(fig)

