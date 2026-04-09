import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF
import nltk
nltk.download('punkt', quiet=True)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "archive"
DATA_DIR.mkdir(exist_ok=True)

REVIEWS_FILE = DATA_DIR / "Reviews.csv"
CHART_FILE = DATA_DIR / "comprehensive_analysis.png"
PDF_FILE = DATA_DIR / "analysis_report.pdf"

# ===== تحميل البيانات =====
if not REVIEWS_FILE.exists():
    raise FileNotFoundError(f"لم يتم العثور على ملف البيانات: {REVIEWS_FILE}\nضع Reviews.csv داخل مجلد {DATA_DIR}")
df = pd.read_csv(REVIEWS_FILE)

print("=" * 60)
print("📊 تحليل شامل لتقييمات المنتجات")
print("=" * 60)

# ===== 1. معلومات أساسية =====
print("\n[1] معلومات أساسية:")
print(f"   إجمالي التقييمات: {df.shape[0]:,}")
print(f"   عدد الأعمدة: {df.shape[1]}")
print(f"   الأعمدة: {', '.join(df.columns.tolist())}")

# ===== 2. تنظيف البيانات =====
print("\n[2] تنظيف البيانات:")
print(f"   القيم الفارغة قبل التنظيف:")
missing_before = df.isnull().sum()
for col, count in missing_before[missing_before > 0].items():
    print(f"      {col}: {count}")

df['ProfileName'].fillna('Unknown', inplace=True)
df['Summary'].fillna('No Summary', inplace=True)
print(f"   إجمالي القيم الفارغة بعد التنظيف: {df.isnull().sum().sum()}")

# ===== 3. الإحصائيات الوصفية =====
print("\n[3] الإحصائيات الوصفية:")
print(f"\n   توزيع التقييمات:")
print(f"      المتوسط: {df['Score'].mean():.2f}")
print(f"      الوسيط: {df['Score'].median():.1f}")
print(f"      الانحراف المعياري: {df['Score'].std():.2f}")

print(f"\n   عدد التقييمات حسب النجوم:")
for score in sorted(df['Score'].unique()):
    count = (df['Score'] == score).sum()
    pct = count / len(df) * 100
    print(f"      {int(score)} نجوم: {count:,} ({pct:.1f}%)")

# ===== 4. تحليل المستخدمين والمنتجات =====
print("\n[4] تحليل المستخدمين والمنتجات:")
print(f"   عدد المستخدمين الفريدين: {df['UserId'].nunique():,}")
print(f"   عدد المنتجات الفريدة: {df['ProductId'].nunique():,}")

print(f"\n   أكثر 5 مستخدمين نشاطاً:")
for idx, (user, count) in enumerate(df['UserId'].value_counts().head(5).items(), 1):
    print(f"      {idx}. {user}: {count} تقييم")

print(f"\n   أكثر 5 منتجات تقييماً:")
for idx, (prod, count) in enumerate(df['ProductId'].value_counts().head(5).items(), 1):
    print(f"      {idx}. {prod}: {count} تقييم")

# ===== 5. تحليل المساعدية =====
print("\n[5] تحليل المساعدية:")
df['HelpfulnessRatio'] = df.apply(
    lambda row: row['HelpfulnessNumerator'] / row['HelpfulnessDenominator']
    if row['HelpfulnessDenominator'] > 0 else 0, axis=1
)

positive_reviews = (df['Score'] >= 4).sum()
negative_reviews = (df['Score'] <= 2).sum()
neutral_reviews = ((df['Score'] == 3)).sum()

print(f"   تقييمات إيجابية (4-5 نجوم): {positive_reviews:,} ({positive_reviews/len(df)*100:.1f}%)")
print(f"   تقييمات محايدة (3 نجوم): {neutral_reviews:,} ({neutral_reviews/len(df)*100:.1f}%)")
print(f"   تقييمات سلبية (1-2 نجوم): {negative_reviews:,} ({negative_reviews/len(df)*100:.1f}%)")

print(f"\n   متوسط نسبة المساعدية: {df['HelpfulnessRatio'].mean():.2%}")
print(f"   وسيط نسبة المساعدية: {df['HelpfulnessRatio'].median():.2%}")

# ===== 6. تحليل النصوص (Sentiment Analysis) =====
print("\n[6] تحليل النصوص (Sentiment Analysis):")

# عينة صغيرة للتحليل (لأن البيانات كبيرة)
sample_size = 10000
df_sample = df.sample(n=sample_size, random_state=42)

print(f"   تحليل عينة من {sample_size:,} تقييم...")

def get_sentiment(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0

df_sample['Sentiment'] = df_sample['Text'].apply(get_sentiment)

print(f"   متوسط المشاعر: {df_sample['Sentiment'].mean():.3f}")
print(f"   توزيع المشاعر:")
print(f"      إيجابية (>0.1): {(df_sample['Sentiment'] > 0.1).sum():,} ({(df_sample['Sentiment'] > 0.1).sum()/len(df_sample)*100:.1f}%)")
print(f"      محايدة (-0.1 إلى 0.1): {((df_sample['Sentiment'] >= -0.1) & (df_sample['Sentiment'] <= 0.1)).sum():,} ({((df_sample['Sentiment'] >= -0.1) & (df_sample['Sentiment'] <= 0.1)).sum()/len(df_sample)*100:.1f}%)")
print(f"      سلبية (<-0.1): {(df_sample['Sentiment'] < -0.1).sum():,} ({(df_sample['Sentiment'] < -0.1).sum()/len(df_sample)*100:.1f}%)")

# ===== 7. نموذج ML للتنبؤ بالتقييمات =====
print("\n[7] نموذج ML للتنبؤ بالتقييمات:")

# تحويل التقييمات إلى تصنيف ثنائي (إيجابي/سلبي)
df_ml = df_sample.copy()
df_ml['Sentiment_Class'] = df_ml['Score'].apply(lambda x: 1 if x >= 4 else 0)

# تحضير البيانات
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df_ml['Text'].fillna(''))
y = df_ml['Sentiment_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# التنبؤ والتقييم
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"   دقة النموذج: {accuracy:.3f}")
print(f"   تقرير التصنيف:")
print(classification_report(y_test, y_pred, target_names=['سلبي', 'إيجابي']))

# ===== 8. تحليل زمني =====
print("\n[8] تحليل زمني:")

# تحويل الوقت إلى تاريخ
df['Date'] = pd.to_datetime(df['Time'], unit='s')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

print(f"   نطاق التواريخ: من {df['Date'].min()} إلى {df['Date'].max()}")

print(f"\n   عدد التقييمات سنوياً:")
yearly_counts = df['Year'].value_counts().sort_index()
for year, count in yearly_counts.items():
    print(f"      {int(year)}: {count:,} تقييم")

print(f"\n   متوسط التقييم سنوياً:")
yearly_avg = df.groupby('Year')['Score'].mean()
for year, avg in yearly_avg.items():
    print(f"      {int(year)}: {avg:.2f}")

# ===== 9. إنشاء الرسوم البيانية =====
print("\n[9] إنشاء الرسوم البيانية...")

plt.rcParams['font.family'] = 'Arial'
sns.set_style("whitegrid")

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('تحليل شامل لتقييمات المنتجات', fontsize=16, fontweight='bold')

# 1. توزيع التقييمات
axes[0, 0].hist(df['Score'], bins=5, color='skyblue', edgecolor='black', rwidth=0.85)
axes[0, 0].set_title('توزيع التقييمات')
axes[0, 0].set_xlabel('النجوم')
axes[0, 0].set_ylabel('التكرار')

# 2. التقييمات السنوية
yearly_counts.plot(kind='bar', ax=axes[0, 1], color='coral', edgecolor='black')
axes[0, 1].set_title('عدد التقييمات سنوياً')
axes[0, 1].set_xlabel('السنة')
axes[0, 1].set_ylabel('العدد')

# 3. متوسط التقييم السنوي
yearly_avg.plot(kind='line', marker='o', ax=axes[1, 0], color='green')
axes[1, 0].set_title('متوسط التقييم سنوياً')
axes[1, 0].set_xlabel('السنة')
axes[1, 0].set_ylabel('المتوسط')
axes[1, 0].set_ylim(1, 5)

# 4. توزيع المشاعر
axes[1, 1].hist(df_sample['Sentiment'], bins=50, color='purple', edgecolor='black')
axes[1, 1].set_title('توزيع المشاعر')
axes[1, 1].set_xlabel('قيمة المشاعر')
axes[1, 1].set_ylabel('التكرار')

# 5. نسبة المساعدية
helpfulness_clean = df['HelpfulnessRatio'][(df['HelpfulnessRatio'] > 0) & (df['HelpfulnessRatio'] <= 1)]
axes[2, 0].hist(helpfulness_clean, bins=50, color='lightgreen', edgecolor='black')
axes[2, 0].set_title('توزيع نسبة المساعدية')
axes[2, 0].set_xlabel('النسبة')
axes[2, 0].set_ylabel('التكرار')

# 6. Box plot للتقييمات
axes[2, 1].boxplot(df['Score'], tick_labels=['التقييم'])
axes[2, 1].set_title('توزيع التقييمات (Box Plot)')
axes[2, 1].set_ylabel('النجوم')

plt.tight_layout()
plt.savefig(CHART_FILE, dpi=300, bbox_inches='tight')
print(f"   ✓ تم حفظ الرسوم البيانية: {CHART_FILE}")

# ===== 10. إنشاء تقرير PDF =====
print("\n[10] إنشاء تقرير PDF...")

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'تقرير تحليل تقييمات المنتجات', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

pdf = PDF()
pdf.add_page()

pdf.chapter_title('ملخص التحليل')
pdf.chapter_body(f'''
إجمالي التقييمات: {len(df):,}
متوسط التقييم: {df['Score'].mean():.2f}/5
عدد المستخدمين: {df['UserId'].nunique():,}
عدد المنتجات: {df['ProductId'].nunique():,}

التقييمات الإيجابية: {positive_reviews:,} ({positive_reviews/len(df)*100:.1f}%)
التقييمات السلبية: {negative_reviews:,} ({negative_reviews/len(df)*100:.1f}%)

دقة نموذج ML: {accuracy:.3f}
''')

pdf.chapter_title('تحليل المشاعر')
pdf.chapter_body(f'''
متوسط المشاعر: {df_sample['Sentiment'].mean():.3f}
المشاعر الإيجابية: {(df_sample['Sentiment'] > 0.1).sum()/len(df_sample)*100:.1f}%
المشاعر السلبية: {(df_sample['Sentiment'] < -0.1).sum()/len(df_sample)*100:.1f}%
''')

pdf.chapter_title('التحليل الزمني')
pdf.chapter_body(f'''
نطاق التواريخ: {df['Date'].min()} - {df['Date'].max()}
أكثر السنوات نشاطاً: {yearly_counts.idxmax()} ({yearly_counts.max():,} تقييم)
''')

pdf.output(PDF_FILE)
print(f"   ✓ تم حفظ التقرير: {PDF_FILE}")

# ===== الملخص النهائي =====
print("\n" + "=" * 60)
print("✨ الملخص النهائي")
print("=" * 60)
print(f"""
📊 إجمالي التقييمات: {len(df):,}
⭐ متوسط التقييم: {df['Score'].mean():.2f}/5
👍 تقييمات إيجابية: {positive_reviews/len(df)*100:.1f}%
👎 تقييمات سلبية: {negative_reviews/len(df)*100:.1f}%

👥 عدد المستخدمين: {df['UserId'].nunique():,}
📦 عدد المنتجات: {df['ProductId'].nunique():,}

🧠 دقة نموذج ML: {accuracy:.3f}
😊 متوسط المشاعر: {df_sample['Sentiment'].mean():.3f}

📈 الرسوم البيانية محفوظة في: {CHART_FILE}
📄 التقرير PDF محفوظ في: {PDF_FILE}

✅ اكتمل التحليل الشامل بنجاح!
""")

print("=" * 60)