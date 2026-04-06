import streamlit as st
import joblib
from pathlib import Path

# 1. إعداد عنوان الصفحة
st.set_page_config(page_title="محلل مشاعر المراجعات", page_icon="📊")
st.title("🤖 نظام تحليل المشاعر الذكي")
st.markdown("أدخل نص التقييم بالإنجليزية وسيقوم النموذج بتوقعه!")

# 2. تحديد مسار الملفات
SCRIPT_DIR = Path(__file__).resolve().parent
POSSIBLE_DIRS = [
    SCRIPT_DIR,
    SCRIPT_DIR / "archive",
    Path.cwd(),
    Path.cwd() / "archive",
]

# دالة لاختيار الملف من أي مسار متاح
def find_asset(filename):
    tried = []

    # المسارات المباشرة المفترضة
    for folder in POSSIBLE_DIRS:
        path = folder / filename
        tried.append(path)
        if path.exists():
            return path, tried

    # البحث في المجلدات الوالدية
    for root in [SCRIPT_DIR, Path.cwd()]:
        for parent in [root] + list(root.parents):
            path = parent / filename
            tried.append(path)
            if path.exists():
                return path, tried
            archive_path = parent / "archive" / filename
            tried.append(archive_path)
            if archive_path.exists():
                return archive_path, tried

    # البحث في أي مكان داخل المشروع
    for folder in [SCRIPT_DIR, Path.cwd()]:
        for path in folder.rglob(filename):
            tried.append(path)
            if path.exists():
                return path, tried

    return None, tried

MODEL_FILE, model_tried = find_asset("sentiment_model.pkl")
VECTORIZER_FILE, vectorizer_tried = find_asset("tfidf_vectorizer.pkl")

missing_files = []
if MODEL_FILE is None:
    missing_files.append("sentiment_model.pkl")
if VECTORIZER_FILE is None:
    missing_files.append("tfidf_vectorizer.pkl")

if missing_files:
    st.error("❌ لم أجد الملفات التالية:")
    for name in missing_files:
        st.write(f"- {name}")
    st.write("\n**المسارات التي تم البحث فيها:**")
    all_tried = []
    if MODEL_FILE is None:
        all_tried += model_tried
    if VECTORIZER_FILE is None:
        all_tried += vectorizer_tried
    for path in all_tried:
        st.write(f"- {path}")
    st.write("\n**المسار الحالي للسكريبت:**")
    st.write(f"- SCRIPT_DIR = {SCRIPT_DIR}")
    st.write(f"- Path.cwd() = {Path.cwd()}")
    st.write("\nتأكد من رفع هذه الملفات إلى نفس مجلد `app.py` أو إلى مجلد `archive` داخل نفس المشروع.")
    st.stop()

# 3. تحميل النموذج والمحول
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer
    except Exception as e:
        st.error(f"خطأ في تحميل الملفات: {e}")
        st.stop()

model, vectorizer = load_assets()

# 4. واجهة المستخدم
user_input = st.text_area("أدخل مراجعتك هنا:", placeholder="مثال: This product is very good")

if st.button("تحليل"):
    if user_input:
        # معالجة النص والتوقع
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        # عرض النتيجة بشكل جمالي
        st.subheader("النتيجة:")
        if prediction == 1:  # إيجابي
            st.success(f"التقييم: إيجابي ✅ (ثقة: {max(probability):.1%})")
        else:  # سلبي
            st.error(f"التقييم: سلبي ❌ (ثقة: {max(probability):.1%})")
        
        # عرض التفاصيل
        st.write(f"**النص المدخل:** {user_input}")
        st.write(f"**الثقة في التنبؤ:** {max(probability):.1%}")
    else:
        st.warning("من فضلك اكتب نصاً أولاً!")
