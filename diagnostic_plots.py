import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as smf
from statsmodels.graphics.gofplots import ProbPlot


def gen_data(n):
    x = np.arange(-n / 2, n / 2, 1, dtype=np.float64)
    m = np.random.uniform(0.3, 0.5, (n,))
    b = np.random.uniform(5, 10, (n,))
    y = x * m + b
    return x, y


def graph(formula, x_range, label=None):
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')


def resvsfit(X, y, model_fit=None):
    if not model_fit:
        model_fit = smf.OLS(y, smf.add_constant(X)).fit()

    df = pd.concat([X, y], axis=1)
    model_fitted_y = model_fit.fittedvalues
    model_residuals = model_fit.resid
    model_abs_resid = np.abs(model_residuals)
    plot_lm_1 = plt.figure()
    plot_lm_1.axes[0] = sns.residplot(model_fitted_y, model_residuals, data=df,
                                      lowess=True,
                                      scatter_kws={'alpha': 0.5},
                                      line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plot_lm_1.axes[0].set_title('Residuals vs Fitted')
    plot_lm_1.axes[0].set_xlabel('Fitted values')
    plot_lm_1.axes[0].set_ylabel('Residuals')
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_5 = abs_resid[:5]
    for i in abs_resid_top_5.index:
        plot_lm_1.axes[0].annotate(i, xy=(model_fitted_y[i], model_residuals[i]))

    plt.show()


def qqplot(X, y, model_fit=None):
    if not model_fit:
        model_fit = smf.OLS(y, smf.add_constant(X)).fit()
    model_fitted_y = model_fit.fittedvalues
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

    QQ = ProbPlot(model_norm_residuals)
    plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
    plot_lm_2.axes[0].set_title('Normal Q-Q')
    plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
    plot_lm_2.axes[0].set_ylabel('Standardized Residuals')
    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_5 = abs_norm_resid[:5]
    for r, i in enumerate(abs_norm_resid_top_5):
        plot_lm_2.axes[0].annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], model_norm_residuals[i]))


def scale_fitted(X, y, model_fit=None):
    if not model_fit:
        model_fit = smf.OLS(y, smf.add_constant(X)).fit()
    model_fitted_y = model_fit.fittedvalues
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    plot_lm_3 = plt.figure()
    plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm_3.axes[0].set_title('Scale-Location')
    plot_lm_3.axes[0].set_xlabel('Fitted values')
    plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$')

    # annotations
    abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
    abs_sq_norm_resid_top_5 = abs_sq_norm_resid[:5]
    for i in abs_sq_norm_resid_top_5:
        plot_lm_3.axes[0].annotate(i, xy=(model_fitted_y[i], model_norm_residuals_abs_sqrt[i]))

    plt.show()


def cook(X, y, model_fit=None):
    if not model_fit:
        model_fit = smf.OLS(y, smf.add_constant(X)).fit()

    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    model_leverage = model_fit.get_influence().hat_matrix_diag
    model_cooks = model_fit.get_influence().cooks_distance[0]

    plot_lm_4 = plt.figure()
    plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(model_leverage, model_norm_residuals,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm_4.axes[0].set_xlim(0, max(model_leverage) + 0.01)
    plot_lm_4.axes[0].set_ylim(-3, 5)
    plot_lm_4.axes[0].set_title('Residuals vs Leverage')
    plot_lm_4.axes[0].set_xlabel('Leverage')
    plot_lm_4.axes[0].set_ylabel('Standardized Residuals')

    leverage_top_5 = np.flip(np.argsort(model_cooks), 0)[:5]
    for i in leverage_top_5:
        plot_lm_4.axes[0].annotate(i, xy=(model_leverage[i], model_norm_residuals[i]))

    p = len(model_fit.params)
    graph(lambda x: np.sqrt((.5 * p * (1 - x)) / x), np.linspace(.001, max(model_leverage), 50), 'Cook\'s distance')
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), np.linspace(.001, max(model_leverage), 50), 'Cook\'s distance')
    plot_lm_4.legend(loc='upper right')

    plt.show()


def diag_plot(X, y, model_fit=None):
    if not model_fit:
        model_fit = smf.OLS(y, smf.add_constant(X)).fit()

    df = pd.concat([X, y], axis=1)
    model_fitted_y = model_fit.fittedvalues
    model_residuals = model_fit.resid
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    model_abs_resid = np.abs(model_residuals)
    model_leverage = model_fit.get_influence().hat_matrix_diag
    model_cooks = model_fit.get_influence().cooks_distance[0]
    plot_lm_1 = plt.figure()
    plot_lm_1.axes[0] = sns.residplot(model_fitted_y, model_residuals, data=df,
                                      lowess=True,
                                      scatter_kws={'alpha': 0.5},
                                      line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plot_lm_1.axes[0].set_title('Residuals vs Fitted')
    plot_lm_1.axes[0].set_xlabel('Fitted values')
    plot_lm_1.axes[0].set_ylabel('Residuals')
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_5 = abs_resid[:5]
    for i in abs_resid_top_5.index:
        plot_lm_1.axes[0].annotate(i, xy=(model_fitted_y[i], model_residuals[i]))

    plt.show()

    QQ = ProbPlot(model_norm_residuals)
    plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
    plot_lm_2.axes[0].set_title('Normal Q-Q')
    plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
    plot_lm_2.axes[0].set_ylabel('Standardized Residuals')
    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_5 = abs_norm_resid[:5]
    for r, i in enumerate(abs_norm_resid_top_5):
        plot_lm_2.axes[0].annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], model_norm_residuals[i]))

    plot_lm_3 = plt.figure()
    plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm_3.axes[0].set_title('Scale-Location')
    plot_lm_3.axes[0].set_xlabel('Fitted values')
    plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$')

    # annotations
    abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
    abs_sq_norm_resid_top_5 = abs_sq_norm_resid[:5]
    for i in abs_sq_norm_resid_top_5:
        plot_lm_3.axes[0].annotate(i, xy=(model_fitted_y[i], model_norm_residuals_abs_sqrt[i]))

    plt.show()

    plot_lm_4 = plt.figure()
    plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(model_leverage, model_norm_residuals,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm_4.axes[0].set_xlim(0, max(model_leverage) + 0.01)
    plot_lm_4.axes[0].set_ylim(-3, 5)
    plot_lm_4.axes[0].set_title('Residuals vs Leverage')
    plot_lm_4.axes[0].set_xlabel('Leverage')
    plot_lm_4.axes[0].set_ylabel('Standardized Residuals')

    leverage_top_5 = np.flip(np.argsort(model_cooks), 0)[:5]
    for i in leverage_top_5:
        plot_lm_4.axes[0].annotate(i, xy=(model_leverage[i], model_norm_residuals[i]))

    p = len(model_fit.params)
    graph(lambda x: np.sqrt((.5 * p * (1 - x)) / x), np.linspace(.001, max(model_leverage), 50), 'Cook\'s distance')
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), np.linspace(.001, max(model_leverage), 50), 'Cook\'s distance')
    plot_lm_4.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    x, y = gen_data(50)
    X_ = pd.DataFrame(x)
    Y_ = pd.DataFrame(y)
    # diag_plot(X_, Y_)
    resvsfit(X_, Y_)
    qqplot(X_, Y_)
    scale_fitted(X_, Y_)
    cook(X_, Y_)
