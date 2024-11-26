import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from google.colab import drive

# اتصال به Google Drive برای دسترسی به داده‌ها
drive.mount('/content/drive')

# بارگذاری داده‌ها
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/project_machinelearning/diabetes.csv')

# جدا کردن ویژگی‌ها (X) و برچسب‌ها (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# تقسیم داده‌ها به مجموعه‌های آموزشی (70%) و تست (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# اعمال SMOTE فقط روی داده‌های آموزشی
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# استاندارد کردن داده‌ها
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# تابع هدف برای بهینه‌سازی Optuna
def objective(trial):
    param = {
        'iterations': trial.suggest_int('iterations', 100, 500),            # تعداد تکرارها
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),   # نرخ یادگیری
        'depth': trial.suggest_int('depth', 4, 10),                         # عمق حداکثری درخت‌ها
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0),      # منظم‌سازی L2
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),  # پارامتر برای bagging
        'border_count': trial.suggest_int('border_count', 32, 255),         # تعداد مرزها در ویژگی‌های عددی
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0), # قدرت تصادفی‌سازی
    }

    # تعریف مدل CatBoost
    model = CatBoostClassifier(**param, random_seed=42, verbose=0)

    # آموزش مدل
    model.fit(X_train_resampled, y_train_resampled)

    # پیش‌بینی روی داده‌های تست
    y_test_pred = model.predict(X_test)

    # محاسبه دقت مدل روی داده‌های تست
    accuracy = accuracy_score(y_test, y_test_pred)
    
    return accuracy  # تابع هدف دقت را برمی‌گرداند

# ایجاد مطالعه Optuna برای جستجوی بهترین هایپریپارامترها
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # تنظیم تعداد تلاش‌ها (n_trials)

# نمایش بهترین پارامترها
print("Best parameters:", study.best_params)

# آموزش مدل با بهترین پارامترهای پیدا شده توسط Optuna
best_params = study.best_params
best_model = CatBoostClassifier(**best_params, random_seed=42, verbose=0)
best_model.fit(X_train_resampled, y_train_resampled)

# پیش‌بینی روی داده‌های تست با بهترین مدل
y_test_pred = best_model.predict(X_test)

# ارزیابی مدل بر روی داده‌های تست
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test Accuracy with best parameters:', test_accuracy)

# سایر معیارهای ارزیابی
print('Confusion Matrix:\n', confusion_matrix(y_test, y_test_pred))
print('Classification Report:\n', classification_report(y_test, y_test_pred))

# نمایش نتایج پیش‌بینی و مقایسه با برچسب‌های واقعی
results = pd.DataFrame(X_test, columns=X.columns)
results['Real'] = y_test.reset_index(drop=True)
results['Predicted'] = y_test_pred
results['Correct'] = results['Real'] == results['Predicted']

# نمایش نتایج
print(results)
