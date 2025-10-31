import streamlit as st
import joblib


#Load Model & Vectorizer

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


#Page Configuration

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
)


page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #000000; /* solid black */
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background-color: #1a1a1a; /* dark grey sidebar */
}
h1, h2, h3, h4, h5, h6, p, label, div, span {
    color: #ffffff !important; /* white text */
}
.stTextInput, .stTextArea {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid #333333 !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


#Sidebar Info
st.sidebar.title("ℹ️ About")
st.sidebar.write("👨‍💻 Developed by **Atharva K. A.**")
st.sidebar.write("🎓 B.E. Artificial Intelligence & Machine Learning")
st.sidebar.write("🏫 Sir M. Visvesvaraya Institute of Technology, Bengaluru")
st.sidebar.metric("📈 Model Accuracy", "98.8%")
st.sidebar.caption("Using TF-IDF + Logistic Regression")


# 📰 App Main Title
st.title("📰 Fake News Detection System")
st.write("Detect whether a given news article is **Real** or **Fake** using an AI model trained on Kaggle datasets.")


#User Input

user_input = st.text_area("📝 Enter News Text Below:", height=200)


#Prediction Logic

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        if prediction == 1:
            st.error("🚨 This news appears to be **FAKE**!")
        else:
            st.success("✅ This news seems **REAL**.")


#Footer
st.markdown("---")
st.caption("🧠 Project: AI/ML Fake News Detection | Built with Python, scikit-learn, and Streamlit.")
