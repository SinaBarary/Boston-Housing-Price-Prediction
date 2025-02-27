import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import joblib

# خواندن فایل CSV
data = pd.read_csv('D:\\A_I\\BostonHousing.csv')

# نمایش چند سطر اول از داده‌ها
#print(data.head())
#نمایش داده های گمشده
#print(data.isnull().sum())
# پر کردن داده‌های گمشده (در صورت وجود)
data = data.fillna(method='ffill')

# جداسازی ویژگی‌ها و برچسب‌ها
X = data.drop(columns=['medv'])  # ویژگی‌ها (به فرض اینکه 'medv' ستون هدف یا همان قیمت باشد)
y = data['medv']  # برچسب‌ها

# استانداردسازی ویژگی‌ها
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تقسیم‌بندی داده‌ها به بخش‌های آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
"""
print('x_train : ',X_train)
print('y_train : ',y_train)
print('x_test : ',X_test)
print('y_test : ',y_test)
"""

# ایجاد مدل رگرسیون خطی
model = LinearRegression()
# آموزش مدل با داده‌های آموزش
model.fit(X_train, y_train)

# پیش‌بینی با داده‌های تست
y_pred = model.predict(X_test)

# محاسبه خطا (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# ایجاد مدل درخت تصمیم‌گیری
tree_model = DecisionTreeRegressor(random_state=42)

# آموزش مدل با داده‌های آموزش
tree_model.fit(X_train, y_train)

# پیش‌بینی با داده‌های تست
y_pred_tree = tree_model.predict(X_test)

# محاسبه خطا (Mean Squared Error)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print(f'Mean Squared Error (Decision Tree): {mse_tree}')
from sklearn.model_selection import cross_val_score

# اعتبارسنجی متقابل با 5 بخش (5-fold cross-validation)
cv_scores = cross_val_score(tree_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')

# محاسبه میانگین و انحراف معیار خطاها
mean_cv_score = -cv_scores.mean()
std_cv_score = cv_scores.std()
print(f'Mean CV MSE: {mean_cv_score}')
print(f'Standard Deviation of CV MSE: {std_cv_score}')


# تعریف پارامترهای جستجو
param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# ایجاد مدل جستجوی شبکه‌ای
grid_search = GridSearchCV(estimator=tree_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# آموزش مدل با جستجوی شبکه‌ای
grid_search.fit(X_train, y_train)

# بهترین پارامترها و مدل
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# پیش‌بینی با بهترین مدل
y_pred_best = best_model.predict(X_test)

# محاسبه خطا (Mean Squared Error) با بهترین مدل
mse_best = mean_squared_error(y_test, y_pred_best)
print(f'Best Model Mean Squared Error: {mse_best}')
print(f'Best Parameters: {best_params}')

# ذخیره مدل آموزش‌دیده در فایل
joblib.dump(best_model, 'D:\\A_I\\best_model.pkl')


# بارگذاری مدل آموزش‌دیده از فایل
best_model = joblib.load('D:\\A_I\\best_model.pkl')

# اصلاح نام‌های ستون‌ها به حروف کوچک
data.columns = [col.lower() for col in data.columns]

# بررسی نام‌های ستون‌ها
print(data.columns)

# فرض کنید یک داده جدید داریم
new_data = pd.DataFrame({
    'crim': [0.1],
    'zn': [0],
    'indus': [8.14],
    'chas': [0],
    'nox': [0.538],
    'rm': [6.575],
    'age': [65.2],
    'dis': [4.0900],
    'rad': [1],
    'tax': [296],
    'ptratio': [15.3],
    'b': [396.9],
    'lstat': [4.98]
})

# استانداردسازی داده جدید
new_data_scaled = scaler.transform(new_data)

# پیش‌بینی قیمت با بهترین مدل درخت تصمیم‌گیری
predicted_price = best_model.predict(new_data_scaled)
print(f'Predicted Price: {predicted_price[0]}')
