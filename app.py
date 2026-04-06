import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go

# إعداد الصفحة
st.set_page_config(
    page_title="تحليل المشاعر للتقييمات",
    page_icon="🧠",
    layout="wide"
)

# ✅ تحديد المسارات بشكل صحيح
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_FILE = SCRIPT_DIR / "sentiment_model.pkl"
VECTORIZER_FILE = SCRIPT_DIR / "tfidf_vectorizer.pkl"
DATA_FILE = SCRIPT_DIR / "Reviews.csv"  # ✅ إضافة DATA_FILE

# ✅ التحقق من وجود الملفات قبل أي حاجة
for f in [DATA_FILE, MODEL_FILE, VECTORIZER_FILE]:
    if not f.exists():
        st.error(f"❌ الملف مش موجود: {f.name}")
        st.stop()

# تحميل البيانات والنموذج
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)  # ✅ DATA_FILE معرّفة دلوقتي
        return df
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {e}")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {e}")
        return None, None

# تحليل المشاعر
def predict_sentiment(text, model, vectorizer):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    return prediction, probability

def analyze_text_sentiment(text, model, vectorizer):
    blob_sentiment = TextBlob(text).sentiment.polarity
    ml_prediction, ml_proba = predict_sentiment(text, model, vectorizer)

    return {
        'text': text[:100] + '...' if len(text) > 100 else text,
        'textblob_sentiment': blob_sentiment,
        'ml_prediction': 'إيجابي' if ml_prediction == 1 else 'سلبي',
        'ml_confidence': max(ml_proba),
        'overall_sentiment': 'إيجابي' if (blob_sentiment > 0.1 and ml_prediction == 1) else 'سلبي'
    }

# الصفحة الرئيسية
def main():
    st.title("🧠 تحليل المشاعر للتقييمات")
    st.markdown("---")

    df = load_data()
    model, vectorizer = load_model()

    if df is None or model is None:
        st.stop()

    # إحصائيات سريعة
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("عدد التقييمات", f"{len(df):,}")
    with col2:
        avg_score = df['Score'].mean()
        st.metric("متوسط التقييم", f"{avg_score:.2f}")
    with col3:
        positive_reviews = len(df[df['Score'] >= 4])
        st.metric("التقييمات الإيجابية", f"{positive_reviews:,}")
    with col4:
        negative_reviews = len(df[df['Score'] < 4])
        st.metric("التقييمات السلبية", f"{negative_reviews:,}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 التحليلات", "📈 الرسوم البيانية", "🔍 التنبؤ", "📋 عينات"])

    with tab1:
        st.header("تحليل البيانات الأساسي")
        st.subheader("توزيع التقييمات")
        score_counts = df['Score'].value_counts().sort_index()
        st.bar_chart(score_counts)

        st.subheader("إحصائيات مفصلة")
        st.dataframe(df.describe())

        st.subheader("عينة من التقييمات")
        st.dataframe(df[['Score', 'Summary', 'Text']].head(10))

    with tab2:
        st.header("الرسوم البيانية التفاعلية")

        fig1 = px.histogram(df, x='Score', title='توزيع التقييمات')
        st.plotly_chart(fig1)

        if 'Time' in df.columns:
            df['Date'] = pd.to_datetime(df['Time'], unit='s')
            df['Year'] = df['Date'].dt.year
            yearly_counts = df.groupby('Year').size().reset_index(name='Count')
            fig2 = px.line(yearly_counts, x='Year', y='Count', title='عدد التقييمات سنوياً')
            st.plotly_chart(fig2)

        df['Text_Length'] = df['Text'].str.len()
        fig3 = px.histogram(df, x='Text_Length', title='توزيع طول النصوص')
        st.plotly_chart(fig3)

    with tab3:
        st.header("تنبؤ بالمشاعر")

        user_text = st.text_area("اكتب نص التقييم:", height=100)

        if st.button("تحليل المشاعر", type="primary"):
            if user_text.strip():
                result = analyze_text_sentiment(user_text, model, vectorizer)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("نتائج TextBlob")
                    if result['textblob_sentiment'] > 0:
                        st.success(f"إيجابي: {result['textblob_sentiment']:.3f}")
                    elif result['textblob_sentiment'] < 0:
                        st.error(f"سلبي: {result['textblob_sentiment']:.3f}")
                    else:
                        st.info(f"محايد: {result['textblob_sentiment']:.3f}")

                with col2:
                    st.subheader("نتائج الذكاء الاصطناعي")
                    confidence = result['ml_confidence'] * 100
                    if result['ml_prediction'] == 'إيجابي':
                        st.success(f"{result['ml_prediction']} ({confidence:.1f}%)")
                    else:
                        st.error(f"{result['ml_prediction']} ({confidence:.1f}%)")

                st.subheader("التقييم العام")
                if result['overall_sentiment'] == 'إيجابي':
                    st.success("🟢 التقييم العام: إيجابي")
                else:
                    st.error("🔴 التقييم العام: سلبي")
            else:
                st.warning("من فضلك اكتب نص للتحليل")

    with tab4:
        st.header("عينات من التقييمات")

        st.subheader("عينات إيجابية")
        positive_samples = df[df['Score'] >= 4][['Score', 'Summary', 'Text']].sample(5)
        for _, row in positive_samples.iterrows():
            with st.expander(f"⭐ {row['Score']} - {row['Summary'][:50]}..."):
                st.write(row['Text'])

        st.subheader("عينات سلبية")
        negative_samples = df[df['Score'] < 4][['Score', 'Summary', 'Text']].sample(5)
        for _, row in negative_samples.iterrows():
            with st.expander(f"⭐ {row['Score']} - {row['Summary'][:50]}..."):
                st.write(row['Text'])

if __name__ == "__main__":
    main()
