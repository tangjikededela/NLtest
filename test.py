import numpy as np
from sklearn.linear_model import LinearRegression
from jinja2 import Template
import os

_here = os.path.dirname(os.path.abspath(__file__))
test1_tmpl_path = os.path.join(_here, 'templates', 'testLinear')
test2_tmpl_path = os.path.join(_here, 'templates', 'testStats')
TEST1_TMPL = Template(open(test1_tmpl_path).read())
TEST2_TMPL = Template(open(test2_tmpl_path).read())


class Summary:
    def __int__(self):
        self.summary_data = None

    def summary(self, m):

        if str(m) == "LinearRegression()":
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
                y_pre= model.fit().fittedvalues,
            )
            return summary


#Example

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
y_pred = model.predict(x)
example1 = Summary()
print(example1.summary(model))

from sklearn.preprocessing import PolynomialFeatures
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])
transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
model = LinearRegression().fit(x_, y)
r_sq = model.score(x_, y)
y_pred = model.predict(x_)
example2 = Summary()
print(example2.summary(model))



import statsmodels.api as sm

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)
x = sm.add_constant(x)
model = sm.OLS(y, x)
# results = model.fit()
example3=Summary()
print(example3.summary(model))

