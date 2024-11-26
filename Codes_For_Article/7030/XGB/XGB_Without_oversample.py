# بهترین کد برای افزایش دقت مقاله مرجع

import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from google.colab import drive

# اتصال به Google Drive برای دسترسی به داده‌ها
drive.mount('/content/drive')

# بارگذاری داده‌ها
data = pd.read_csv('/content/drive/MyDrive/data_Article/diabetes.csv')

# حذف مقادیر NaN از y
data = data.dropna(subset=['Outcome'])  # حذف سطرهایی که دارای NaN در 'Outcome' هستند

# جدا کردن ویژگی‌ها (X) و برچسب‌ها (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# تقسیم داده‌ها به مجموعه‌های آموزشی (70%) و تست (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# تعریف تابع هدف برای بهینه‌سازی
def objective(trial):
    # فضای جستجوی هایپریپارامترها
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),               # تعداد درخت‌ها
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True), # نرخ یادگیری
        'max_depth': trial.suggest_int('max_depth', 3, 10),                        # عمق حداکثری درخت‌ها
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),          # حداقل وزن نمونه در یک برگ
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),                           # پارامتر برای کاهش انشعابات
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),                   # نسبت نمونه‌های استفاده شده برای آموزش هر درخت
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),     # نسبت ویژگی‌های استفاده شده برای هر درخت
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),                   # منظم‌سازی L1
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0)                  # منظم‌سازی L2
    }

    # تعریف مدل XGBoost با پارامترهای پیشنهادی
    xgb_model = XGBClassifier(**param, random_state=42, use_label_encoder=False)

    # آموزش مدل روی داده‌های آموزشی
    xgb_model.fit(X_train, y_train)

    # پیش‌بینی روی داده‌های تست
    y_test_pred = xgb_model.predict(X_test)

    # محاسبه دقت مدل روی داده‌های تست
    accuracy = accuracy_score(y_test, y_test_pred)

    return accuracy  # تابع هدف دقت را برمی‌گرداند

# ایجاد مطالعه Optuna برای جستجوی بهترین هایپریپارامترها
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# نمایش بهترین پارامترها
print("Best parameters:", study.best_params)

# آموزش مدل با بهترین پارامترهای پیدا شده توسط Optuna
best_params = study.best_params
best_model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False)
best_model.fit(X_train, y_train)

# ارزیابی مدل با داده‌های تست
y_test_pred = best_model.predict(X_test)

# محاسبه و نمایش معیارهای ارزیابی
test_accuracy = accuracy_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

# چاپ نتایج
print("Test Accuracy with best parameters:", test_accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
print("ROC AUC Score:", roc_auc)
