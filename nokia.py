import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
data = pd.read_csv(r'C:\Users\manje\Downloads\Nokia_Dataset.csv')
x = DataFrame(data, columns=['sales'])
y = DataFrame(data, columns=['income'])
reg = LinearRegression()
reg.fit(x, y)
plt.plot(x, reg.predict(x), color="red")
plt.scatter(x, y, alpha=0.5)
plt.title("Nokia")
plt.xlabel("sales")
plt.ylabel("income")
plt.show()