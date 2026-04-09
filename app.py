import re
import streamlit as st
import joblib
from pathlib import Path
from textblob import TextBlob

# 1. إعداد عنوان الصفحة
st.set_page_config(page_title="محلل مشاعر المراجعات", page_icon="📊")
st.title("🤖 نظام تحليل المشاعر الذكي")
st.markdown("أدخل نص التقييم بالإنجليزية أو العربية، وسيقوم النموذج بتحليل المشاعر بطريقته الأفضل.")

# 2. تحديد مسار الملفات
SCRIPT_DIR = Path(__file__).resolve().parent
POSSIBLE_DIRS = [
    SCRIPT_DIR,
    SCRIPT_DIR / "archive",
    Path.cwd(),
    Path.cwd() / "archive",
]

def find_asset(filename):
    tried = []
    for folder in POSSIBLE_DIRS:
        path = folder / filename
        tried.append(path)
        if path.exists():
            return path, tried
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
    for folder in [SCRIPT_DIR, Path.cwd()]:
        for path in folder.rglob(filename):
            tried.append(path)
            if path.exists():
                return path, tried
    return None, tried

MODEL_FILE, model_tried         = find_asset("sentiment_model.pkl")
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
    st.write(f"\n- SCRIPT_DIR = {SCRIPT_DIR}")
    st.write(f"- Path.cwd() = {Path.cwd()}")
    st.write("\nتأكد من تشغيل train.py أولاً لإنشاء ملفات الموديل.")
    st.stop()

# 3. تحميل النموذج والمحول
@st.cache_resource
def load_assets():
    try:
        model      = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer
    except Exception as e:
        st.error(f"خطأ في تحميل الملفات: {e}")
        st.stop()

model, vectorizer = load_assets()

# 4. قوائم الكلمات السلبية
NEGATIVE_ARABIC_KEYWORDS = [
    "سيئ", "خدمة سيئة", "مزعج", "أسوأ", "مستاء", "مشكلة", "قرف", "سيئة",
    "مستحيل", "ضعيف", "غير جيد", "حزين", "كراهية", "سلبي", "ليس جيد",
    "لا ينصح", "ما عجبني", "أبداً", "مش حلو", "مش كويس", "وحش", "بايخ",
    "مش تمام", "زفت", "مش عاجبني", "زبالة", "موت", "فشل", "كارثة"
]

NEGATIVE_ENGLISH_KEYWORDS = [
    "bad", "terrible", "awful", "horrible", "worst", "poor", "disgusting",
    "hate", "useless", "disappointing", "not good", "not great", "waste",
    "broken", "pathetic", "garbage", "trash", "mediocre", "annoying",
    "death", "died", "dead", "fail", "failed", "disaster", "horrible"
]

POSITIVE_THRESHOLD = 0.65

# 5. تنظيف النص (نفس الـ train.py)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 6. دوال التنبؤ والتحليل
def predict_with_model(text):
    text_vector = vectorizer.transform([clean_text(text)])
    prediction  = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    return prediction, probability

def analyze_sentiment(text):
    text_lower = text.lower().strip()
    blob_score = TextBlob(text).sentiment.polarity

    # الأولوية 1: فحص الكلمات السلبية
    arabic_negative  = any(kw in text_lower for kw in NEGATIVE_ARABIC_KEYWORDS)
    english_negative = any(kw in text_lower for kw in NEGATIVE_ENGLISH_KEYWORDS)

    if arabic_negative or english_negative:
        return {
            'method'    : 'Keyword Rule',
            'prediction': 'سلبي',
            'confidence': 0.90,
            'blob_score': blob_score,
            'from_model': False,
        }

    # الأولوية 2: TextBlob سلبي بوضوح
    if blob_score <= -0.1:
        return {
            'method'    : 'TextBlob',
            'prediction': 'سلبي',
            'confidence': min(0.5 + abs(blob_score), 0.99),
            'blob_score': blob_score,
            'from_model': False,
        }

    # الأولوية 3: TextBlob إيجابي → نتأكد بالموديل
    if blob_score >= 0.1:
        ml_prediction, ml_proba = predict_with_model(text)
        pos_prob = ml_proba[1] if len(ml_proba) > 1 else ml_proba[0]

        if ml_prediction == 1 and pos_prob >= POSITIVE_THRESHOLD:
            return {
                'method'    : 'TextBlob + Model',
                'prediction': 'إيجابي',
                'confidence': pos_prob,
                'blob_score': blob_score,
                'from_model': True,
            }
        else:
            return {
                'method'    : 'TextBlob',
                'prediction': 'إيجابي',
                'confidence': min(0.5 + blob_score, 0.99),
                'blob_score': blob_score,
                'from_model': False,
            }

    # الأولوية 4: محايد → نعتمد على الموديل فقط
    ml_prediction, ml_proba = predict_with_model(text)
    pos_prob = ml_proba[1] if len(ml_proba) > 1 else ml_proba[0]

    if pos_prob < POSITIVE_THRESHOLD:
        return {
            'method'    : 'Model (uncertain → negative)',
            'prediction': 'سلبي',
            'confidence': 1 - pos_prob,
            'blob_score': blob_score,
            'from_model': True,
        }

    return {
        'method'    : 'Model',
        'prediction': 'إيجابي',
        'confidence': pos_prob,
        'blob_score': blob_score,
        'from_model': True,
    }

# 7. واجهة المستخدم
user_input = st.text_area("أدخل مراجعتك هنا:", placeholder="مثال: This product is very good")

if st.button("تحليل"):
    if user_input:
        result = analyze_sentiment(user_input)

        st.subheader("النتيجة:")
        if result['prediction'] == 'إيجابي':
            st.success(f"التقييم: إيجابي ✅ (ثقة: {result['confidence']:.1%})")
        else:
            st.error(f"التقييم: سلبي ❌ (ثقة: {result['confidence']:.1%})")

        st.write(f"**النص المدخل:** {user_input}")
        st.write(f"**طريقة التقييم:** {result['method']}")
        st.write(f"**درجة TextBlob:** {result['blob_score']:.3f}")
        st.write(f"**الثقة:** {result['confidence']:.1%}")
    else:
        st.warning("من فضلك اكتب نصاً أولاً!")
