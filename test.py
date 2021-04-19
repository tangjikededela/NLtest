import numpy as np
from sklearn.linear_model import LinearRegression
from jinja2 import Template
import os
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from scipy import stats
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from GPyOpt.methods import BayesianOptimization


_here = os.path.dirname(os.path.abspath(__file__))
test1_l1_tmpl_path = os.path.join(_here, 'templates', 'testLinearL1')
test1_l2_tmpl_path = os.path.join(_here, 'templates', 'testLinearL2')
test1_l3_tmpl_path = os.path.join(_here, 'templates', 'testLinearL3')
test2_tmpl_path = os.path.join(_here, 'templates', 'testStats')
test3_tmpl_path = os.path.join(_here, 'templates', 'testPoly')
test4_tmpl_path = os.path.join(_here, 'templates', 'testScipy')
test5_tmpl_path = os.path.join(_here, 'templates', 'testARIMA')
test6_l1_tmpl_path = os.path.join(_here, 'templates', 'testLogL1')
test6_l2_tmpl_path = os.path.join(_here, 'templates', 'testLogL2')
test6_l3_tmpl_path = os.path.join(_here, 'templates', 'testLogL3')
test7_tmpl_path = os.path.join(_here, 'templates', 'testPiecewise')
test8_tmpl_path = os.path.join(_here, 'templates', 'testPiecewisePwlf')
test9_tmpl_path = os.path.join(_here, 'templates', 'testPiecewisePwlfSummary')
test10_tmpl_path = os.path.join(_here, 'templates', 'testPiecewisePwlfV2')
TEST1_l1_TMPL = Template(open(test1_l1_tmpl_path).read())
TEST1_l2_TMPL = Template(open(test1_l2_tmpl_path).read())
TEST1_l3_TMPL = Template(open(test1_l3_tmpl_path).read())
TEST2_TMPL = Template(open(test2_tmpl_path).read())
TEST3_TMPL = Template(open(test3_tmpl_path).read())
TEST4_TMPL = Template(open(test4_tmpl_path).read())
TEST5_TMPL = Template(open(test5_tmpl_path).read())
TEST6_l1_TMPL = Template(open(test6_l1_tmpl_path).read())
TEST6_l2_TMPL = Template(open(test6_l2_tmpl_path).read())
TEST6_l3_TMPL = Template(open(test6_l3_tmpl_path).read())
TEST7_TMPL = Template(open(test7_tmpl_path).read())
TEST8_TMPL = Template(open(test8_tmpl_path).read())
TEST9_TMPL = Template(open(test9_tmpl_path).read())
TEST10_TMPL = Template(open(test10_tmpl_path).read())


class Summary:
    def __int__(self):
        self.summary_data = None

    def summary(self, m):

        if "LinearRegression()" in str(m):
            modelfit = LinearRegression().fit(X, y)

            if "ndarray" in str(type(X)):
                plt.plot(X, y, '.', label='original data')
                plt.plot(X, modelfit.predict(X), 'r', label='linear fitted line')
                plt.legend()
                plt.show()
            if level == 1:
                lm = LinearRegression()
                lm.fit(X, y)
                params = np.append(lm.intercept_, lm.coef_)
                predictions = lm.predict(X)

                newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
                MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

                var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
                sd_b = np.sqrt(var_b)
                ts_b = params / sd_b

                # p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]
                sd_b = np.round(sd_b, 3)
                ts_b = np.round(ts_b, 3)
                # p_values = np.round(p_values, 3)
                X2 = sm.add_constant(X)
                est = sm.OLS(y, X2)
                est2 = est.fit()
                p_determine1 = 0
                p_determine2 = 0
                p_determine3 = 0
                p_determine4 = 0
                if all(i >= 0.07 for i in est2.pvalues) is True:
                    p_determine1 = 1
                elif all(i >= 0.05 for i in est2.pvalues) is True:
                    p_determine2 = 1
                elif any(i >= 0.07 for i in est2.pvalues) is True:
                    p_determine3 = 1
                elif any(i >= 0.05 for i in est2.pvalues) is True:
                    p_determine4 = 1

                print(est2.summary())

                summary = TEST1_l3_TMPL.render(
                    model=m,
                    R=modelfit.score(X, y),
                    inter=modelfit.intercept_,
                    slope=modelfit.coef_,
                    y_pre=modelfit.predict(X),
                    Xcol=XcolName,
                    ycol=ycolName,
                    sd=sd_b,
                    t=ts_b,
                    p=est2.pvalues,
                    p2=np.round(est2.pvalues, 3),
                    pd1=p_determine1,
                    pd2=p_determine2,
                    pd3=p_determine3,
                    pd4=p_determine4,
                )
                return summary
            elif level == 2:
                lm = LinearRegression()
                lm.fit(X, y)
                params = np.append(lm.intercept_, lm.coef_)
                predictions = lm.predict(X)

                newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
                MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

                var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
                sd_b = np.sqrt(var_b)
                ts_b = params / sd_b

                sd_b = np.round(sd_b, 3)
                ts_b = np.round(ts_b, 3)
                # p_values = np.round(p_values, 3)
                X2 = sm.add_constant(X)
                est = sm.OLS(y, X2)
                est2 = est.fit()
                p_determine1 = 0
                p_determine2 = 0
                p_determine3 = 0
                p_determine4 = 0
                if all(i >= 0.07 for i in est2.pvalues) is True:
                    p_determine1 = 1
                elif all(i >= 0.05 for i in est2.pvalues) is True:
                    p_determine2 = 1
                elif any(i >= 0.07 for i in est2.pvalues) is True:
                    p_determine3 = 1
                elif any(i >= 0.05 for i in est2.pvalues) is True:
                    p_determine4 = 1

                print(est2.summary())

                summary = TEST1_l3_TMPL.render(
                    model=m,
                    R=modelfit.score(X, y),
                    inter=modelfit.intercept_,
                    slope=modelfit.coef_,
                    y_pre=modelfit.predict(X),
                    Xcol=XcolName,
                    ycol=ycolName,
                    sd=sd_b,
                    t=ts_b,
                    p=est2.pvalues,
                    p2=np.round(est2.pvalues, 3),
                    pd1=p_determine1,
                    pd2=p_determine2,
                    pd3=p_determine3,
                    pd4=p_determine4,
                )
                return summary
            else:
                if "ndarray" in str(type(X)):
                    train_sizes, train_score, test_score = learning_curve(model, X, y,
                                                                          train_sizes=np.linspace(.5, 1, 8), cv=15,
                                                                          scoring='neg_mean_squared_error')
                    train_error = 1 - np.mean(train_score, axis=1)
                    test_error = 1 - np.mean(test_score, axis=1)
                    plt.plot(train_sizes, train_error, 'o-', color='r', label='training')
                    plt.plot(train_sizes, test_error, 'o-', color='g', label='testing')
                    plt.legend(loc='best')
                    plt.xlabel('traing examples')
                    plt.ylabel('error')
                    img = plt.imread('templates\pic.jpg')
                    fig = plt.figure('show picture')
                    plt.imshow(img)
                    plt.show()

                lm = LinearRegression()
                lm.fit(X, y)
                params = np.append(lm.intercept_, lm.coef_)
                predictions = lm.predict(X)

                newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
                MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

                var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
                sd_b = np.sqrt(var_b)
                ts_b = params / sd_b

                # p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]

                sd_b = np.round(sd_b, 3)
                ts_b = np.round(ts_b, 3)
                # p_values = np.round(p_values, 3)
                X2 = sm.add_constant(X)
                est = sm.OLS(y, X2)
                est2 = est.fit()
                p_determine1 = 0
                p_determine2 = 0
                p_determine3 = 0
                p_determine4 = 0
                if all(i >= 0.07 for i in est2.pvalues) is True:
                    p_determine1 = 1
                elif all(i >= 0.05 for i in est2.pvalues) is True:
                    p_determine2 = 1
                elif any(i >= 0.07 for i in est2.pvalues) is True:
                    p_determine3 = 1
                elif any(i >= 0.05 for i in est2.pvalues) is True:
                    p_determine4 = 1

                print(est2.summary())

                summary = TEST1_l3_TMPL.render(
                    model=m,
                    R=modelfit.score(X, y),
                    inter=modelfit.intercept_,
                    slope=modelfit.coef_,
                    y_pre=modelfit.predict(X),
                    Xcol=XcolName,
                    ycol=ycolName,
                    sd=sd_b,
                    t=ts_b,
                    p=est2.pvalues,
                    p2=np.round(est2.pvalues, 3),
                    pd1=p_determine1,
                    pd2=p_determine2,
                    pd3=p_determine3,
                    pd4=p_determine4,
                )
                return summary
        elif "statsmodels.api" in str(m):
            summary = TEST2_TMPL.render(
                model=m,
                R=model.fit().rsquared,
                R2=model.fit().rsquared_adj,
                RC=model.fit().params,
                y_pre=model.fit().fittedvalues,
            )
            return summary
        elif "polyfit" in str(m):

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
        elif "ARIMA" in str(m):
            residuals = DataFrame(model_fit.resid)
            print(residuals.describe())
            print(model_fit.summary())
            # evaluate forecasts
            rmse = sqrt(mean_squared_error(test, predictions))
            # plot forecasts against actual outcomes
            pyplot.plot(test)
            pyplot.plot(predictions, color='red')
            pyplot.show()
            summary = TEST5_TMPL.render(
                model=m,
                R=rmse,
                y_pre=predictions,
            )
            return summary
        elif "piecewise" in str(m):

            summary = TEST7_TMPL.render(
                model1=model1,
                model2=model2,
                model3=model3,
                Xcol=xcolname,
                ycol=ycolname,
                change1=change1,
                change2=change2,
                change3=change3,
                CD=ChangeDetermine,

            )
            return summary
        elif "PiecewiseLinFit" in str(m):
            my_pwlf=m
            def my_obj(x):
                # define some penalty parameter l
                # you'll have to arbitrarily pick this
                # it depends upon the noise in your data,
                # and the value of your sum of square of residuals
                l = y.mean() * 0.001
                f = np.zeros(x.shape[0])
                for i, j in enumerate(x):
                    my_pwlf.fit(j[0])
                    f[i] = my_pwlf.ssr + (l * j[0])
                return f

            # define the lower and upper bound for the number of line segments
            bounds = [{'name': 'var_1', 'type': 'discrete',
                       'domain': np.arange(2, 10)}]

            np.random.seed(12121)

            myBopt = BayesianOptimization(my_obj, domain=bounds, model_type='GP',
                                          initial_design_numdata=10,
                                          initial_design_type='latin',
                                          exact_feval=True, verbosity=True,
                                          verbosity_model=False)
            max_iter = 30

            # perform the bayesian optimization to find the optimum number
            # of line segments
            myBopt.run_optimization(max_iter=max_iter, verbosity=True)

            # print('\n \n Opt found \n')
            # print('Optimum number of line segments:', myBopt.x_opt)
            # print('Function value:', myBopt.fx_opt)
            myBopt.plot_acquisition()
            myBopt.plot_convergence()

            # perform the fit for the optimum
            BP = my_pwlf.fit(myBopt.x_opt)
            slopes = my_pwlf.calc_slopes()
            BPNumber = int(myBopt.x_opt[0])
            print(BPNumber)
            print(BP)
            print(slopes)

            # predict for the determined points
            xHat = np.linspace(min(x), max(x), num=1000)
            yHat = my_pwlf.predict(xHat)

            # plot the results
            plt.figure()
            plt.plot(x, y, 'o')
            plt.plot(xHat, yHat, '-')
            plt.show()
            increasePart=" "
            decreasePart=" "
            notchangePart=" "
            maxIncrease=" "
            maxDecrease=" "
            for i in range(BPNumber):
                # print("the x change is", BP[i + 1] - BP[i])
                # print("the y change is", int(my_pwlf.predict(BP[i + 1]) - my_pwlf.predict(BP[i])))
                # print("the slope is", slopes[i])
                if slopes[i] < 0:
                    decreasePart= decreasePart+"from "+str(np.round(BP[i], 2))+" till "+str(np.round(BP[i+1], 2)) + ", "
                    if slopes[i] == min(slopes):
                        maxDecrease = str(np.round(BP[i], 2))+" till "+str(np.round(BP[i+1], 2))
                elif slopes[i] > 0:
                    increasePart = increasePart +"from "+ str(np.round(BP[i], 2)) + " till "+str(np.round(BP[i+1], 2)) + ", "
                    if slopes[i] == max(slopes):
                        maxIncrease = str(np.round(BP[i], 2))+" till "+str(np.round(BP[i+1], 2))
                else:
                    notchangePart = notchangePart +"from "+ str(np.round(BP[i], 2)) + " till "+str(np.round(BP[i+1], 2)) + ", "
                summary = TEST8_TMPL.render(
                    ychange=abs(my_pwlf.predict(BP[i + 1]) - my_pwlf.predict(BP[i])),
                    Xchange=BP[i + 1] - BP[i],
                    slope=slopes[i],
                    Xcol=xcolname,
                    ycol=ycolname,
                    n=i+1,
                )
                print(summary)
            print("In summary, ")
            for i in range(BPNumber):
                CD=abs(max(y)-min(y))
                summary = TEST9_TMPL.render(
                    ychange=abs(my_pwlf.predict(BP[i + 1]) - my_pwlf.predict(BP[i])),
                    Xchange=BP[i + 1] - BP[i],
                    slope=slopes[i],
                    Xcol=xcolname,
                    ycol=ycolname,
                    n=i+1,
                    end=BPNumber,
                    CD=CD,
                )
                print(summary)
            summary = TEST10_TMPL.render(
                iP=increasePart,
                dP=decreasePart,
                nP=notchangePart,
                Xcol=xcolname,
                ycol=ycolname,
                n=BPNumber,
                mI=maxIncrease,
                mD=maxDecrease,
            )
            print(summary)

        elif "LogisticRegression" in str(m):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            # plt.plot(X, y, '.', label='original data')
            # plt.plot(X, model.predict(X), '+', label='linear fitted line')
            # plt.legend()
            # plt.show()
            if level !=1:
                # Heatmap
                class_names = [0, 1]  # name  of classes
                fig, ax = plt.subplots()
                tick_marks = np.arange(len(class_names))
                plt.xticks(tick_marks, class_names)
                plt.yticks(tick_marks, class_names)
                # Create heatmap
                sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
                ax.xaxis.set_label_position("top")
                plt.tight_layout()
                plt.title('Confusion matrix', y=1.1)
                plt.ylabel('Actual label')
                plt.xlabel('Predicted label')
                plt.show()
            # ROC
            y_pred_proba = model.predict_proba(X_test)[::, 1]
            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
            auc = metrics.roc_auc_score(y_test, y_pred_proba)
            if level == 3:
                plt.plot(fpr, tpr, label="data, auc=" + str(auc))
                plt.legend(loc=4)
                plt.show()

            COEF=np.hsplit(model.coef_, np.size(model.coef_))
            CoefFactor=""
            for i in range(np.size(model.coef_)):
                if COEF[i]<0:
                    CoefFactor = CoefFactor+XcolName[i]+" is negative correlation."+"\n"+"which means the higher the" + XcolName[i] +" is, the lower possibility of " +ycolName+"\n"
                else:
                    CoefFactor = CoefFactor+XcolName[i]+" is positive correlation."+"\n"+"which means the higher the" + XcolName[i] +" is, the higher possibility of " +ycolName+"\n"

            if level == 1:
                summary = TEST6_l1_TMPL.render(
                    model=m,
                    cof=model.coef_,
                    CF=CoefFactor,
                    Accuracy=metrics.accuracy_score(y_test, y_pred),
                    Precision=metrics.precision_score(y_test, y_pred),
                    Xcol=XcolName,
                    ycol=ycolName,
                    AUC=auc,
                )
                return summary
            elif level ==2:
                summary = TEST6_l2_TMPL.render(
                    model=m,
                    cof=model.coef_,
                    CF=CoefFactor,
                    Accuracy=metrics.accuracy_score(y_test, y_pred),
                    Precision=metrics.precision_score(y_test, y_pred),
                    Xcol=XcolName,
                    ycol=ycolName,
                    AUC=auc,
                )
                return summary
            elif level ==3:
                summary = TEST6_l3_TMPL.render(
                    model=m,
                    cof=model.coef_,
                    CF=CoefFactor,
                    Accuracy=metrics.accuracy_score(y_test, y_pred),
                    Precision=metrics.precision_score(y_test, y_pred),
                    Xcol=XcolName,
                    ycol=ycolName,
                    AUC=auc,
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
# xp = np.linspace(-10, int(data.shape[0] * 1.1), int(data.shape[0] * 1.1))
# example5 = Summary()
# print(example5.summary(model))

# Example 6
import math
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot

# model = scipy.stats.linregress(x, y)
# example6 = Summary()
# print(example6.summary(model))

# model = "numpy.polyfit"
# p = np.poly1d(np.polyfit(x, y, 6))
# example7 = Summary()
# print(example7.summary(model))

# Example 8
# from pandas import read_csv
# from pandas import datetime
# from pandas import DataFrame
# from matplotlib import pyplot
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error
# from pandas.plotting import autocorrelation_plot
# from math import sqrt
# # load dataset
# def parser(x):
#     return datetime.strptime('20' + x, '%Y-%m')
#
# series = read_csv('gold2.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
# series.index = series.index.to_period('M')
# # series.plot()
# # series.diff(1).plot()
# # autocorrelation_plot(series)
# # pyplot.show()
# # split into train and test sets
# X = series.values
# size = int(len(X) * 0.66)
# train, test = X[0:size], X[size:len(X)]
# history = [x for x in train]
# predictions = list()
# # walk-forward validation
# for t in range(len(test)):
#     model = ARIMA(history, order=(20, 1, 1))
#     model_fit = model.fit()
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test[t]
#     history.append(obs)
#     # print('predicted=%f, expected=%f' % (yhat, obs))
# example8 = Summary()
# print(example8.summary(model))

# Example 9
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix
# x = np.arange(10).reshape(-1, 1)
# y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
# model = LogisticRegression(solver='liblinear', random_state=0)
# model.fit(x, y)
# classes=model.classes_
# intercept=model.intercept_
# cof=model.coef_
# model.predict_proba(x)
# model.predict(x)

# # Example 10
#
# data = read_csv('HW.csv')
#
# colName = list(data.columns)
#
# X = data.Height
# XcolName = colName[0]
# y = data.Weight
# ycolName = colName[1]
# X, y = np.array(X).reshape(-1, 1), np.array(y)
# level = 3
# model = LinearRegression()
#
# example10 = Summary()
# print(example10.summary(model))
# print()

# # Example 11
# import pandas as pd
# import statsmodels.api as sm
#
# Stock_Market = {
#     'Year': [2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2016, 2016, 2016, 2016, 2016, 2016,
#              2016, 2016, 2016, 2016, 2016, 2016],
#     'Month': [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
#     'Interest_Rate': [2.75, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.25, 2.25, 2.25, 2, 2, 2, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
#                       1.75, 1.75, 1.75, 1.75, 1.75],
#     'Unemployment_Rate': [5.3, 5.3, 5.3, 5.3, 5.4, 5.6, 5.5, 5.5, 5.5, 5.6, 5.7, 5.9, 6, 5.9, 5.8, 6.1, 6.2, 6.1, 6.1,
#                           6.1, 5.9, 6.2, 6.2, 6.1],
#     'Stock_Index_Price': [1464, 1394, 1357, 1293, 1256, 1254, 1234, 1195, 1159, 1167, 1130, 1075, 1047, 965, 943, 958,
#                           971, 949, 884, 866, 876, 822, 704, 719]
# }
#
# df = pd.DataFrame(Stock_Market, columns=['Year', 'Month', 'Interest_Rate', 'Unemployment_Rate', 'Stock_Index_Price'])
#
# X = df[['Interest_Rate',
#         'Unemployment_Rate']]
# y = df['Stock_Index_Price']
# colName = list(df.columns)
#
# XcolName = colName[2] + " & " + colName[3]
# ycolName = colName[4]
# model = LinearRegression()
# level = 2
# example11 = Summary()
# print(example11.summary(model))

#Example 12
# from sklearn.linear_model import LogisticRegression
# col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# pima = pd.read_csv("diabetes.csv", header=None, names=col_names)
# feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
# X = pima[feature_cols] # Features
# y = pima.label # Target variable
# XcolName=feature_cols
# ycolName="get diabetes"
# model = LogisticRegression(solver='lbfgs', max_iter=1000)
# level = 3
# example12 = Summary()
# print(example12.summary(model))

# Example 13 对多段函数寻找断点，手动拟合
import matplotlib.pyplot as plt
import ruptures as rpt
data = np.genfromtxt('windspeed.csv',delimiter=',',skip_header=1)
data = data[data[:,1].argsort()]
signal = data[:,1:3]
x=data[:,1]
y=data[:,2]



# #algo = rpt.BottomUp(model="linear").fit(signal)
# #algo = rpt.KernelCPD(kernel="linear").fit(signal) one point correct
# #algo = rpt.Binseg(model="linear").fit(signal)
# #algo = rpt.Window(width=7600, model="linear").fit(signal)
# result = algo.predict(n_bkps=2)
# print(result)
# rpt.display(signal, result)
# plt.show()
# first = signal[:result[0],:2]
# second = signal[result[0]:result[1]:2]
# thrid=signal[result[1]:,:2]
# x1,y1=first[:,:1],first[:,1:2]
# x2,y2=second[:,:1],second[:,1:2]
# x3,y3=thrid[:,:1],thrid[:,1:2]
#
# model2="numpy.polyfit"
# plt.plot(x, y, '.', label='original data')
# model1="LinearRegression()"
# modelfit1 = LinearRegression().fit(x1, y1)
# plt.plot(x1, modelfit1.predict(x1), 'r', label='Linear fitted line 1')
# p = np.poly1d(np.polyfit(x2.ravel(), y2.ravel(), 3))
# xp = np.linspace(x2[0], x2[x2.size-1] , x2.size)
# plt.plot(xp, p(xp), '-', label='Polynomial fitted line 2')
# model3="LinearRegression()"
# modelfit3 = LinearRegression().fit(x3, y3)
# plt.plot(x3, modelfit3.predict(x3), 'y', label='Linear fitted line 3')
#
# plt.legend()
# plt.show()
#
# ChangeDetermine=(max(y)-min(y))/100
# change1=y1[y1.size-1]-y1[0]
# change2=y2[y2.size-1]-y2[0]
# change3=y3[y3.size-1]-y3[0]
# xcolname="Wind Speed"
# ycolname="Theoretical Energy"
# model="piecewise"
#
# example13 = Summary()
# print(example13.summary(model))

# Example 14 对多段函数寻找断点，自动拟合
import pwlf
# x0 = np.array([min(x), x[result[0]],x[result[1]], max(x)])
#
# # initialize piecewise linear fit with your x and y data
# my_pwlf = pwlf.PiecewiseLinFit(x, y)
#
# # fit the data with the specified break points
# # (ie the x locations of where the line segments
# # will terminate)
# my_pwlf.fit_with_breaks(x0)
#
# # predict for the determined points
# xHat = np.linspace(min(x), max(x), num=10000)
# yHat = my_pwlf.predict(xHat)
#
# # plot the results
# plt.figure()
# plt.plot(x, y, 'o')
# plt.plot(xHat, yHat, '-')
# plt.show()
#
# my_pwlf = pwlf.PiecewiseLinFit(x, y)
#
# # fit the data for four line segments
# res = my_pwlf.fit(len(result))
#
# # predict for the determined points
# xHat = np.linspace(min(x), max(x), num=5000)
# yHat = my_pwlf.predict(xHat)
#
# # plot the results
# plt.figure()
# plt.plot(x, y, 'o')
# plt.plot(xHat, yHat, '-')
# plt.show()

import numpy as np
import pwlf
col_names = ['month', 'AvocadoAveragePrice']
data = pd.read_csv("avocado.csv", header=None, names=col_names)
x = data.month # Features
y = data.AvocadoAveragePrice # Target variable
# x = np.array([4., 5., 6., 7., 8.,9.,10.,11.])
# y = np.array([11., 13., 16., 28.92, 42.81,52.8,62.8,72.8])
# xcolname="day"
# ycolname="money"
xcolname="month"
ycolname="Avocado Average Price"

# my_pwlf = pwlf.PiecewiseLinFit(x, y)
# breaks = my_pwlf.fit_guess([6.0])
# xHat = np.linspace(min(x), max(x), num=5000)
# yHat = my_pwlf.predict(xHat)
# plt.figure()
# plt.plot(x, y, 'o')
# plt.plot(xHat, yHat, '-')
# plt.show()


my_pwlf = pwlf.PiecewiseLinFit(x, y)

example14 = Summary()
print(example14.summary(my_pwlf))

