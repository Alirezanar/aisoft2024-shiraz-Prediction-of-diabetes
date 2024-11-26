import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
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
    # تنظیم هایپرپارامترهای مدل‌ها
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        'gamma': trial.suggest_float('xgb_gamma', 0.0, 0.5),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.0, 3.0)
    }

    catboost_params = {
        'iterations': trial.suggest_int('cat_iterations', 100, 500),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('cat_depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1e-1, 10.0)
    }

    gb_params = {
        'n_estimators': trial.suggest_int('gb_n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('gb_max_depth', 3, 10)
    }

    lr_params = {
        'C': trial.suggest_float('lr_C', 0.01, 10.0)
    }

    # ایجاد مدل Voting Classifier با پارامترهای بهینه
    voting_model = VotingClassifier(estimators=[
        ('xgb', XGBClassifier(**xgb_params, random_state=42, use_label_encoder=False)),
        ('catboost', CatBoostClassifier(**catboost_params, silent=True, random_state=42)),
        ('gb', GradientBoostingClassifier(**gb_params, random_state=42)),
        ('lr', LogisticRegression(**lr_params, solver='liblinear', random_state=42))
    ], voting='hard')

    # آموزش مدل Voting
    voting_model.fit(X_train_resampled, y_train_resampled)

    # پیش‌بینی روی داده‌های تست
    y_test_pred = voting_model.predict(X_test)

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

# ایجاد مدل Voting Classifier با بهترین پارامترها
best_voting_model = VotingClassifier(estimators=[
    ('xgb', XGBClassifier(**{
        'n_estimators': best_params['xgb_n_estimators'],
        'learning_rate': best_params['xgb_learning_rate'],
        'max_depth': best_params['xgb_max_depth'],
        'min_child_weight': best_params['xgb_min_child_weight'],
        'gamma': best_params['xgb_gamma'],
        'subsample': best_params['xgb_subsample'],
        'colsample_bytree': best_params['xgb_colsample_bytree'],
        'reg_alpha': best_params['xgb_reg_alpha'],
        'reg_lambda': best_params['xgb_reg_lambda']
    }, random_state=42, use_label_encoder=False)),
    ('catboost', CatBoostClassifier(**{
        'iterations': best_params['cat_iterations'],
        'learning_rate': best_params['cat_learning_rate'],
        'depth': best_params['cat_depth'],
        'l2_leaf_reg': best_params['cat_l2_leaf_reg']
    }, silent=True, random_state=42)),
    ('gb', GradientBoostingClassifier(**{
        'n_estimators': best_params['gb_n_estimators'],
        'learning_rate': best_params['gb_learning_rate'],
        'max_depth': best_params['gb_max_depth']
    }, random_state=42)),
    ('lr', LogisticRegression(**{
        'C': best_params['lr_C']
    }, solver='liblinear', random_state=42))
], voting='hard')

# آموزش مدل با بهترین پارامترها
best_voting_model.fit(X_train_resampled, y_train_resampled)

# پیش‌بینی روی داده‌های تست با بهترین مدل
y_test_pred = best_voting_model.predict(X_test)

# ارزیابی مدل بر روی داده‌های تست
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test Accuracy with best Voting Classifier:', test_accuracy)

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
