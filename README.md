# Boston-Housing-Price-Prediction
این پروژه شامل یک مدل یادگیری ماشین برای پیش‌بینی قیمت مسکن در شهر بوستون است.
مراحل اصلی این پروژه عبارتند از:

آماده‌سازی داده‌ها: شامل خواندن داده‌ها از فایل CSV، پیش‌پردازش داده‌ها و استانداردسازی ویژگی‌ها.

آموزش مدل: شامل آموزش مدل‌های مختلف از جمله رگرسیون خطی و درخت تصمیم‌گیری و بهینه‌سازی هایپرپارامترها.

ارزیابی مدل: ارزیابی عملکرد مدل‌ها با استفاده از معیار Mean Squared Error و اعتبارسنجی متقابل.

استقرار مدل: ذخیره و بارگذاری مدل آموزش‌دیده برای استفاده‌های آینده.

فایل‌ها و پوشه‌ها:
BostonHousing.csv: فایل داده‌های مسکن بوستون.

predictHomeValue.py: فایل کد پایتون شامل مراحل پیش‌پردازش، آموزش و استقرار مدل.

best_model.pkl: فایل مدل ذخیره‌شده برای استفاده در پیش‌بینی‌های آینده.

نحوه استفاده:
نصب نیازمندی‌ها:

pip install -r requirements.txt

اجرای کد:

python predictHomeValue.py
پیش‌بینی قیمت‌های جدید: مدل ذخیره‌شده را بارگذاری کرده و از آن برای پیش‌بینی قیمت‌های جدید استفاده کنید.

منابع آموزشی:
این پروژه بر اساس مفاهیم پایه‌ای و پیشرفته یادگیری ماشین و هوش مصنوعی ایجاد شده است. برای یادگیری بیشتر، می‌توانید از منابع زیر استفاده کنید:

دوره‌های آنلاین: Coursera، edX، Udacity

کتاب‌های مرجع: "Deep Learning" نوشته Ian Goodfellow، "Pattern Recognition and Machine Learning" نوشته Christopher Bishop
