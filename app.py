import streamlit as st
import joblib
import re

# 1. دالة تنظيف النص (نفس اللي استخدمتها في التدريب)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text) # شلت الحروف العربي لو مش محتاجها
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 2. تحميل الموديلات الجاهزة
@st.cache_resource
def load_models():
    # تأكد أن هذه الملفات موجودة في نفس المجلد على GitHub
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# محاولة تحميل الموديل
try:
    model, vectorizer = load_models()
except:
    st.error("خطأ: لم نجد ملفات الموديل (.pkl). تأكد من رفعها على GitHub بجانب هذا الملف.")

# 3. واجهة التطبيق
st.title("تحليل مشاعر مراجعات المنتجات 📝")
st.write("اكتب رأيك بالإنجليزية، والموديل هيتوقع لو كان إيجابي ولا سلبي.")

user_input = st.text_area("ادخل مراجعتك هنا:", placeholder="e.g. This product is amazing!")

if st.button("تحليل"):
    if user_input.strip():
        # تنظيف النص
        cleaned = clean_text(user_input)
        # تحويل النص لـ Vector
        vec = vectorizer.transform([cleaned])
        # التوقع
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        
        # عرض النتيجة
        if prediction == 1:
            st.success(f"النتيجة: **إيجابي ✅** (ثقة: {max(proba):.1%})")
        else:
            st.error(f"النتيجة: **سلبي ❌** (ثقة: {max(proba):.1%})")
    else:
        st.warning("من فضلك اكتب نصاً أولاً.")

st.markdown("---")
st.caption("برمجة وتطوير: AmrNour1995")
