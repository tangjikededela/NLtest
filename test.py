import numpy as np
from sklearn.linear_model import LinearRegression
from jinja2 import Template
import os
import scipy.stats
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
test1_tmpl_path = os.path.join(_here, 'templates', 'testLinear')
test2_tmpl_path = os.path.join(_here, 'templates', 'testStats')
test3_tmpl_path = os.path.join(_here, 'templates', 'testPoly')
test4_tmpl_path = os.path.join(_here, 'templates', 'testScipy')
TEST1_TMPL = Template(open(test1_tmpl_path).read())
TEST2_TMPL = Template(open(test2_tmpl_path).read())
TEST3_TMPL = Template(open(test3_tmpl_path).read())
TEST4_TMPL = Template(open(test4_tmpl_path).read())


class Summary:
    def __int__(self):
        self.summary_data = None

    def summary(self, m):

        if "LinearRegression()" in str(m):
            summary = TEST1_TMPL.render(
                model=m,
                R=r_sq,
                inter=model.intercept_,
                slope=model.coef_,
                y_pre=y_pred,
            )
            return summary
        elif "statsmodels" in str(m):
            summary = TEST2_TMPL.render(
                model=m,
                R=model.fit().rsquared,
                R2=model.fit().rsquared_adj,
                RC=model.fit().params,
                y_pre=model.fit().fittedvalues,
            )
            return summary
        elif "polyfit" in str(m):
            xp = np.linspace(-10, int(data.shape[0] * 1.1), int(data.shape[0] * 1.1))
            plt.plot(x, y, '.', label='original data')
            plt.plot(xp, p(xp), '-', label='fitted line')
            yhat = p(x)
            ybar = np.sum(y) / len(y)
            ssreg = np.sum((yhat - ybar) ** 2)
            sstot = np.sum((y - ybar) ** 2)
            Rsquared = ssreg / sstot
            y_pre = p(data.shape[0] + 180)
            plt.legend()
            plt.show()
            summary = TEST3_TMPL.render(
                model=m,
                R=Rsquared,
                y_pre=y_pre,
                deg=p.order,
            )
            return summary
        elif "LinregressResult" in str(m):
            slope, intercept, r_value, p_value, std_err = model
            plt.plot(x, y, '.', label='original data')
            plt.plot(x, model.intercept + model.slope * x, 'r', label='fitted line')
            plt.legend()
            plt.show()
            summary = TEST4_TMPL.render(
                model=m,
                slope=model.slope,
                p_value=p_value,
                std_err=std_err,
                R=(scipy.stats.pearsonr(x, y)[0]) * (scipy.stats.pearsonr(x, y)[0]),
                rho=scipy.stats.spearmanr(x, y)[0],
                tau=scipy.stats.kendalltau(x, y)[0],
            )
            return summary


# Example 1

# x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
# y = [4, 5, 20, 14, 32, 22, 38, 43]
# x, y = np.array(x), np.array(y)
# model = LinearRegression().fit(x, y)
# r_sq = model.score(x, y)
# y_pred = model.predict(x)
# example1 = Summary()
# print(example1.summary(model))

# Example 3
# import statsmodels.api as sm
# x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
# y = [4, 5, 20, 14, 32, 22, 38, 43]
# x, y = np.array(x), np.array(y)
# x = sm.add_constant(x)
# model = sm.OLS(y, x)
# example3 = Summary()
# print(example3.summary(model))

# Example 4
# x = np.array([0, 5, 15, 25, 35, 45, 55, 60])
# x = np.array(x, dtype=float)
# y = np.array([4, 5, 20, 14, 32, 22, 38, 43])
# y = np.array(y, dtype=float)
# model = scipy.stats.linregress(x, y)
# example4 = Summary()
# print(example4.summary(model))

# Example 5
# model = "numpy.polyfit"
# p = np.poly1d(np.polyfit(x, y, 3))
# example5 = Summary()
# print(example5.summary(model))

# Example 6
import math
import pandas as pd

data = pd.read_csv("GoldPrice.csv")

data["Date"] = pd.to_datetime(data["Date"])
data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(method='ffill')
data.head()
i = 0
y = pd.Series([])
for n in range(0, data.shape[0]):
    y[i] = data.loc[n]["Price"]
    i = i + 1

i = 0
j = data.shape[0] - 1
n = math.floor(data.shape[0] / 2)

for n in range(0, n):
    temp1 = y[i]
    temp2 = y[j]
    y[i] = temp2
    y[j] = temp1
    i = i + 1
    j = j - 1

x = pd.Series(range(data.shape[0]))

model = scipy.stats.linregress(x, y)
example6 = Summary()
print(example6.summary(model))


model = "numpy.polyfit"
p = np.poly1d(np.polyfit(x, y, 6))
example7 = Summary()
print(example7.summary(model))
