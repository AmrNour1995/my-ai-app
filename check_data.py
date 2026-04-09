import pandas as pd

# 1. تحميل ملف البيانات
# تأكد أن الملف في نفس الفولدر اللي أنت واقف فيه
df = pd.read_csv('Reviews.csv')

# 2. عرض أسماء الأعمدة فقط
print("Columns Names:")
print(df.columns)

# 3. عرض أول 5 صفوف بشكل منظم (اختياري)
print("\nFirst 5 rows:")
print(df.head())