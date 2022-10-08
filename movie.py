import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
data = pd.read_csv(r'C:\me\[FreeCourseSite.com] Udemy - Complete 2022 Data Science & Machine Learning Bootcamp\02. Predict Movie Box Office Revenue with Linear Regression\03.2 cost_revenue_clean.csv')
x = DataFrame(data, columns=['production_budget_usd'])
y = DataFrame(data, columns=['worldwide_gross_usd'])
reg = LinearRegression()
reg.fit(x,y)
plt.plot(x,reg.predict(x), color="red")
plt.scatter(x, y, alpha=0.3)
plt.title('movie predictions')
plt.xlabel('budget')
plt.ylabel('revenue')
plt.ylim(0,3000000000)
plt.xlim(0,450000000)
plt.show()
