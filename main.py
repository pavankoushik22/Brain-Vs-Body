import pandas as pd
#for data frames or handling the csv data
from sklearn import linear_model
#completely baked estimator
import matplotlib.pyplot as plt
#visualize


dataframe = pd.read_fwf('data.txt')
x = dataframe[["Brain"]]
y = dataframe[["Body"]]

regress = linear_model.LinearRegression()
regress.fit(x,y)

plt.scatter(x,y)
plt.plot(x,regress.predict(x))
plt.show()

