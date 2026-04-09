import pandas as pd
import joblib
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# ⚙️ الإعدادات
# ============================================================
CSV_PATH     = "data.csv"
TEXT_COLUMN  = "Text"
LABEL_COLUMN = "Score"
# ============================================================


# 1. تحميل البيانات
df = pd.read_csv('Reviews.csv')
print("✅ تم تحميل البيانات")
print(f"   الحجم الكلي: {df.shape}\n")

# 2. الأعمدة المطلوبة فقط
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
df[TEXT_COLUMN]  = df[TEXT_COLUMN].astype(str).str.strip()
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

# 3. حذف الـ Score = 3 (محايد)
df = df[df[LABEL_COLUMN] != 3]
print(f"   بعد حذف المحايد (Score=3): {df.shape}\n")

# 4. تحويل Score لـ 0 و 1
df["label_encoded"] = df[LABEL_COLUMN].apply(
    lambda x: 1 if x >= 4 else 0
)

# 5. فحص التوزيع
print("📊 توزيع البيانات:")
counts = df["label_encoded"].value_counts()
print(f"   إيجابي (1): {counts.get(1, 0):,}  ({counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"   سلبي   (0): {counts.get(0, 0):,}  ({counts.get(0, 0)/len(df)*100:.1f}%)\n")

X = df[TEXT_COLUMN]
y = df["label_encoded"]

# 6. تنظيف النصوص
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

X = X.apply(clean_text)

# 7. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"✅ التقسيم: {len(X_train):,} تدريب | {len(X_test):,} اختبار\n")

# 8. TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# 9. تدريب الموديل
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    C=1.0,
    solver='lbfgs'
)

model.fit(X_train_vec, y_train)
print("✅ تم تدريب الموديل\n")

# 10. تقييم الموديل
y_pred = model.predict(X_test_vec)

print("📈 تقرير الأداء:")
print(classification_report(y_test, y_pred, target_names=["سلبي (1-2)", "إيجابي (4-5)"]))

# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["سلبي", "إيجابي"],
            yticklabels=["سلبي", "إيجابي"])
plt.title("Confusion Matrix")
plt.ylabel("الحقيقي")
plt.xlabel("المتوقع")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 12. حفظ الموديل
joblib.dump(model,      "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\n✅ تم حفظ sentiment_model.pkl")
print("✅ تم حفظ tfidf_vectorizer.pkl")

# 13. اختبار سريع
print("\n🧪 اختبار سريع:")
test_samples = [
    "This product is amazing and works perfectly",
    "This is the worst product I have ever bought",
    "Terrible quality, complete waste of money",
    "Absolutely love it, highly recommend",
    "garbage",
    "disgusting smell and bad taste",
]
for sample in test_samples:
    vec   = vectorizer.transform([clean_text(sample)])
    pred  = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label = "إيجابي ✅" if pred == 1 else "سلبي ❌"
    print(f"   '{sample}' → {label} (ثقة: {max(proba):.1%})")