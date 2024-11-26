# ابتدا کتابخانه های زیر را نصب کنید
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from deap import base, creator, tools, algorithms
from google.colab import drive

# میتوانید فایل داده ها را در پوشه کد ها ذخیره کنید که نیازی به ادرس طولانی نباشد
# اتصال به Google Drive 
drive.mount('/content/drive')


# بارگذاری داده‌ها
data = pd.read_csv('/content/drive/MyDrive/data_Article/diabetes.csv')

# حذف مقادیر NaN(از تارگت)
data = data.dropna(subset=['Outcome'])  # حذف سطرهایی که دارای NaN در 'Outcome' هستند

# جدا کردن ویژگی‌ها (X) و برچسب‌ها (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# تقسیم داده‌ها به مجموعه‌های آموزشی (90%) و تست (10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# اعمال SMOTE برای مقابله با عدم توازن کلاس‌ها
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# محدوده پارامترها برای بهینه‌سازی
param_ranges = {
    'iterations': (100, 500),
    'learning_rate': (0.01, 0.3),
    'depth': (3, 10),
    'l2_leaf_reg': (1, 10),
    'bagging_temperature': (0.0, 1.0),
    'random_strength': (1.0, 10.0)
}

# تابع برای محدود کردن مقادیر پارامترها به محدوده مجاز
def check_bounds(individual):
    for i, key in enumerate(param_ranges.keys()):
        if individual[i] < param_ranges[key][0]:
            individual[i] = param_ranges[key][0]
        elif individual[i] > param_ranges[key][1]:
            individual[i] = param_ranges[key][1]
    return individual

# تعریف creator برای تابع هدف
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# تعریف toolbox برای الگوریتم ژنتیک
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0, 1)
toolbox.register("individual", tools.initIterate, creator.Individual, lambda: [
    np.random.randint(param_ranges['iterations'][0], param_ranges['iterations'][1]),
    np.random.uniform(param_ranges['learning_rate'][0], param_ranges['learning_rate'][1]),
    np.random.randint(param_ranges['depth'][0], param_ranges['depth'][1]),
    np.random.uniform(param_ranges['l2_leaf_reg'][0], param_ranges['l2_leaf_reg'][1]),
    np.random.uniform(param_ranges['bagging_temperature'][0], param_ranges['bagging_temperature'][1]),
    np.random.uniform(param_ranges['random_strength'][0], param_ranges['random_strength'][1])
])

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# تابع ارزیابی
def evaluate(individual):
    params = {
        'iterations': int(individual[0]),
        'learning_rate': individual[1],
        'depth': int(individual[2]),
        'l2_leaf_reg': individual[3],
        'bagging_temperature': individual[4],
        'random_strength': individual[5],
        'random_seed': 42,
        'verbose': 0
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    return (accuracy_score(y_test, y_pred),)

toolbox.register("evaluate", evaluate)

# تعداد نسل‌ها و جمعیت اولیه
NGEN = 10
POP_SIZE = 20

# ایجاد جمعیت اولیه
population = toolbox.population(n=POP_SIZE)

# الگوریتم ژنتیک
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    offspring = [check_bounds(ind) for ind in offspring]
    
    # ارزیابی تناسب افراد
    fits = list(map(toolbox.evaluate, offspring))
    
    # اختصاص دادن تناسب به افراد
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    
    # انتخاب افراد برای نسل بعد
    population = toolbox.select(offspring, k=len(population))

# پیدا کردن بهترین فرد
best_ind = tools.selBest(population, 1)[0]
print(f"Best individual is {best_ind}")
print(f"With accuracy: {best_ind.fitness.values[0]}")

# ارزیابی بهترین مدل با پارامترهای بهینه‌شده
best_params = {
    'iterations': int(best_ind[0]),
    'learning_rate': best_ind[1],
    'depth': int(best_ind[2]),
    'l2_leaf_reg': best_ind[3],
    'bagging_temperature': best_ind[4],
    'random_strength': best_ind[5],
    'random_seed': 42,
    'verbose': 0
}

best_model = CatBoostClassifier(**best_params)
best_model.fit(X_train_resampled, y_train_resampled)
y_test_pred = best_model.predict(X_test)

# ارزیابی نهایی مدل بهینه‌شده
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Final Test Accuracy with optimized parameters: {test_accuracy}')
print('Confusion Matrix:\n', confusion_matrix(y_test, y_test_pred))
print('Classification Report:\n', classification_report(y_test, y_test_pred))
