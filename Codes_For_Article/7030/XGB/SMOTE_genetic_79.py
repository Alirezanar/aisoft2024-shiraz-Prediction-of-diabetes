import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from deap import base, creator, tools, algorithms
from google.colab import drive

# اتصال به Google Drive برای دسترسی به داده‌ها
drive.mount('/content/drive')

# بارگذاری داده‌ها
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/project_machinelearning/diabetes.csv')

# جدا کردن ویژگی‌ها (X) و برچسب‌ها (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# تقسیم داده‌ها به مجموعه‌های آموزشی (70%) و تست (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# اعمال SMOTE برای مقابله با عدم توازن کلاس‌ها
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# محدوده پارامترها برای بهینه‌سازی
param_ranges = {
    'n_estimators': (100, 500),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 10),
    'min_child_weight': (1, 10),
    'gamma': (0.0, 0.5),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (0.0, 1.0),
    'reg_lambda': (0.0, 3.0)
}

# تعریف نوع ژنوم و ایجاد محیط تکاملی
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# تعریف ژن‌ها در قالب محدوده پارامترها
toolbox.register("attr_int", np.random.randint, param_ranges['n_estimators'][0], param_ranges['n_estimators'][1] + 1)
toolbox.register("attr_float", np.random.uniform, param_ranges['learning_rate'][0], param_ranges['learning_rate'][1])
toolbox.register("attr_depth", np.random.randint, param_ranges['max_depth'][0], param_ranges['max_depth'][1] + 1)
toolbox.register("attr_int_weight", np.random.randint, param_ranges['min_child_weight'][0], param_ranges['min_child_weight'][1] + 1)
toolbox.register("attr_gamma", np.random.uniform, param_ranges['gamma'][0], param_ranges['gamma'][1])
toolbox.register("attr_subsample", np.random.uniform, param_ranges['subsample'][0], param_ranges['subsample'][1])
toolbox.register("attr_colsample", np.random.uniform, param_ranges['colsample_bytree'][0], param_ranges['colsample_bytree'][1])
toolbox.register("attr_alpha", np.random.uniform, param_ranges['reg_alpha'][0], param_ranges['reg_alpha'][1])
toolbox.register("attr_lambda", np.random.uniform, param_ranges['reg_lambda'][0], param_ranges['reg_lambda'][1])

# ایجاد فرد (individual)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_int, toolbox.attr_float, toolbox.attr_depth, toolbox.attr_int_weight,
                  toolbox.attr_gamma, toolbox.attr_subsample, toolbox.attr_colsample, toolbox.attr_alpha,
                  toolbox.attr_lambda), n=1)

# تعریف جمعیت
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# تابع ارزیابی مدل
def eval_xgboost(individual):
    params = {
        'n_estimators': int(individual[0]),
        'learning_rate': individual[1],
        'max_depth': int(individual[2]),
        'min_child_weight': int(individual[3]),
        'gamma': individual[4],
        'subsample': individual[5],
        'colsample_bytree': individual[6],
        'reg_alpha': individual[7],
        'reg_lambda': individual[8],
        'random_state': 42,
        'use_label_encoder': False
    }

    # ایجاد و آموزش مدل
    model = XGBClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)

    # پیش‌بینی و ارزیابی مدل
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return (accuracy,)

# ثبت توابع ژنتیک
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_xgboost)

# محدود کردن مقادیر ژن‌ها به محدوده‌های تعریف شده
def check_bounds(individual):
    individual[0] = int(max(min(individual[0], param_ranges['n_estimators'][1]), param_ranges['n_estimators'][0]))
    individual[1] = max(min(individual[1], param_ranges['learning_rate'][1]), param_ranges['learning_rate'][0])
    individual[2] = int(max(min(individual[2], param_ranges['max_depth'][1]), param_ranges['max_depth'][0]))
    individual[3] = int(max(min(individual[3], param_ranges['min_child_weight'][1]), param_ranges['min_child_weight'][0]))
    individual[4] = max(min(individual[4], param_ranges['gamma'][1]), param_ranges['gamma'][0])
    individual[5] = max(min(individual[5], param_ranges['subsample'][1]), param_ranges['subsample'][0])
    individual[6] = max(min(individual[6], param_ranges['colsample_bytree'][1]), param_ranges['colsample_bytree'][0])
    individual[7] = max(min(individual[7], param_ranges['reg_alpha'][1]), param_ranges['reg_alpha'][0])
    individual[8] = max(min(individual[8], param_ranges['reg_lambda'][1]), param_ranges['reg_lambda'][0])
    return individual

# تکامل جمعیت
population = toolbox.population(n=20)
NGEN = 10
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    offspring = [check_bounds(ind) for ind in offspring]
    fits = list(map(toolbox.evaluate, offspring))

    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit

    population = toolbox.select(offspring, k=len(population))

# نمایش بهترین فرد
best_ind = tools.selBest(population, 1)[0]
print(f"Best individual is: {best_ind}, with accuracy: {best_ind.fitness.values[0]:.4f}")


# --------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score  # تغییر مکان cross_val_score به model_selection
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from deap import base, creator, tools, algorithms
from google.colab import drive

# اتصال به Google Drive برای دسترسی به داده‌ها
drive.mount('/content/drive')

# بارگذاری داده‌ها
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/project_machinelearning/diabetes.csv')

# جدا کردن ویژگی‌ها (X) و برچسب‌ها (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# تقسیم داده‌ها به مجموعه‌های آموزشی (70%) و تست (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# اعمال SMOTE برای مقابله با عدم توازن کلاس‌ها
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# محدوده پارامترها برای بهینه‌سازی
param_ranges = {
    'n_estimators': (100, 500),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 10),
    'min_child_weight': (1, 10),
    'gamma': (0.0, 0.5),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (0.0, 1.0),
    'reg_lambda': (0.0, 3.0)
}

# تعریف نوع ژنوم و ایجاد محیط تکاملی
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# تعریف ژن‌ها در قالب محدوده پارامترها
toolbox.register("attr_int", np.random.randint, param_ranges['n_estimators'][0], param_ranges['n_estimators'][1] + 1)
toolbox.register("attr_float", np.random.uniform, param_ranges['learning_rate'][0], param_ranges['learning_rate'][1])
toolbox.register("attr_depth", np.random.randint, param_ranges['max_depth'][0], param_ranges['max_depth'][1] + 1)
toolbox.register("attr_int_weight", np.random.randint, param_ranges['min_child_weight'][0], param_ranges['min_child_weight'][1] + 1)
toolbox.register("attr_gamma", np.random.uniform, param_ranges['gamma'][0], param_ranges['gamma'][1])
toolbox.register("attr_subsample", np.random.uniform, param_ranges['subsample'][0], param_ranges['subsample'][1])
toolbox.register("attr_colsample", np.random.uniform, param_ranges['colsample_bytree'][0], param_ranges['colsample_bytree'][1])
toolbox.register("attr_alpha", np.random.uniform, param_ranges['reg_alpha'][0], param_ranges['reg_alpha'][1])
toolbox.register("attr_lambda", np.random.uniform, param_ranges['reg_lambda'][0], param_ranges['reg_lambda'][1])

# ایجاد فرد (individual)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_int, toolbox.attr_float, toolbox.attr_depth, toolbox.attr_int_weight,
                  toolbox.attr_gamma, toolbox.attr_subsample, toolbox.attr_colsample, toolbox.attr_alpha,
                  toolbox.attr_lambda), n=1)

# تعریف جمعیت
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# تابع ارزیابی مدل
def eval_xgboost(individual):
    params = {
        'n_estimators': int(individual[0]),
        'learning_rate': individual[1],
        'max_depth': int(individual[2]),
        'min_child_weight': int(individual[3]),
        'gamma': individual[4],
        'subsample': individual[5],
        'colsample_bytree': individual[6],
        'reg_alpha': individual[7],
        'reg_lambda': individual[8],
        'random_state': 42,
        'use_label_encoder': False
    }

    # ایجاد و آموزش مدل
    model = XGBClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)

    # پیش‌بینی و ارزیابی مدل
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return (accuracy,)

# ثبت توابع ژنتیک
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_xgboost)

# محدود کردن مقادیر ژن‌ها به محدوده‌های تعریف شده
def check_bounds(individual):
    individual[0] = int(max(min(individual[0], param_ranges['n_estimators'][1]), param_ranges['n_estimators'][0]))
    individual[1] = max(min(individual[1], param_ranges['learning_rate'][1]), param_ranges['learning_rate'][0])
    individual[2] = int(max(min(individual[2], param_ranges['max_depth'][1]), param_ranges['max_depth'][0]))
    individual[3] = int(max(min(individual[3], param_ranges['min_child_weight'][1]), param_ranges['min_child_weight'][0]))
    individual[4] = max(min(individual[4], param_ranges['gamma'][1]), param_ranges['gamma'][0])
    individual[5] = max(min(individual[5], param_ranges['subsample'][1]), param_ranges['subsample'][0])
    individual[6] = max(min(individual[6], param_ranges['colsample_bytree'][1]), param_ranges['colsample_bytree'][0])
    individual[7] = max(min(individual[7], param_ranges['reg_alpha'][1]), param_ranges['reg_alpha'][0])
    individual[8] = max(min(individual[8], param_ranges['reg_lambda'][1]), param_ranges['reg_lambda'][0])
    return individual

# تکامل جمعیت
population = toolbox.population(n=20)
NGEN = 10
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    offspring = [check_bounds(ind) for ind in offspring]
    fits = list(map(toolbox.evaluate, offspring))

    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit

    population = toolbox.select(offspring, k=len(population))

# نمایش بهترین فرد
best_ind = tools.selBest(population, 1)[0]
print(f"Best individual is: {best_ind}, with accuracy: {best_ind.fitness.values[0]:.4f}")

# ساخت مدل بهینه با استفاده از بهترین فرد (best individual)
best_params = {
    'n_estimators': int(best_ind[0]),
    'learning_rate': best_ind[1],
    'max_depth': int(best_ind[2]),
    'min_child_weight': int(best_ind[3]),
    'gamma': best_ind[4],
    'subsample': best_ind[5],
    'colsample_bytree': best_ind[6],
    'reg_alpha': best_ind[7],
    'reg_lambda': best_ind[8],
    'random_state': 42,
    'use_label_encoder': False
}

best_model = XGBClassifier(**best_params)
best_model.fit(X_train_resampled, y_train_resampled)

# ارزیابی مدل با داده‌های تست
y_test_pred = best_model.predict(X_test)

# محاسبه و نمایش معیارهای ارزیابی
test_accuracy = accuracy_score(y_test, y_test_pred)
train_accuracy = accuracy_score(y_train_resampled, best_model.predict(X_train_resampled))  # دقت روی داده‌های آموزشی
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

# محاسبه دقت، فراخوانی، و امتیاز F1
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# محاسبه نرخ مثبت کاذب (FPR)
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
fpr_rate = fpr[1]  # نرخ مثبت کاذب

# Cross-Validation برای ارزیابی دقیق‌تر
cv_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

# چاپ نتایج
print("Test Accuracy with best parameters:", test_accuracy)
print("Train Accuracy:", train_accuracy)  # دقت روی داده‌های آموزشی
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
print("ROC AUC Score:", roc_auc)
print("Precision:", precision)
print("Recall (TPR/Sensitivity):", recall)
print("F1 Score:", f1)
print("False Positive Rate (FPR):", fpr_rate)
print("Cross-Validation Accuracy: {:.2f} (+/- {:.2f})".format(cv_mean, cv_std))
