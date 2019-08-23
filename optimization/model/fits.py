import numpy as np
from scipy import stats
import utils
from scipy.optimize import curve_fit

def func1(x, a, b):
    return a * x ** b


def func10(x, a, b):
    return a + b * x


def func2(x, a, b, c):
    return a * x[0] ** b * x[1] ** c


def d_func1(x, cov, coeffs):
    a, b = coeffs[0], coeffs[1]
    y = func1(x, a, b)
    # print 'd_func1'
    # print coeffs,y,cov
    # dy = y*x**b * np.sqrt(cov[0,0] + (a*np.log(x))**2 * cov[1,1] + a*np.log(x)*cov[0,1]) #+ 0.05
    dy = y * np.sqrt(cov[0, 0] / a ** 2 + cov[1, 1] * (b / x) ** 2 + cov[0, 1] * b / (x * a))
    return dy


def d_func2(x, cov, coeffs):
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    B, CPV = x[0], x[1]
    y = func2(x, a, b, c)
    dy = y * np.sqrt(
        cov[0, 0] / a ** 2 + cov[1, 1] * (b / B) ** 2 + cov[2, 2] * (c / CPV) ** 2 + cov[0, 1] * b / (B * a) + cov[
            0, 2] * c / (CPV * a) + cov[1, 2] * b * c / (B * CPV))
    return dy


def get_chi2_prob(fit, data, sigma):
    vals = ((fit - data) / sigma) ** 2
    chi2 = np.sqrt(np.cumsum(vals))
    ndf = len(vals) - 1
    prob = 1.0 - stats.chi2.cdf(chi2[-1], ndf)

    return (chi2[-1], ndf, prob)


def fit_scores(df_oneday, df_sel, ViewRate_imp, Npoints_min):
    # Fits dependence of campaigns scores vs Budget and CPV
    # Stores results in coeffs_all
    # Fit Scores vs Budget
    Model_accuracy = 0.03
    Prob_lower_limit = 0.10
    # Set fitting limits for Alpha and Beta coeffs
    Alpha_min, Alpha_max = 1.e-6, 1.e6
    Beta_min, Beta_max = 0.50, 0.98

    coeffs_all = {}

    # Add a numeric day:
    df_sel['D'] = df_sel['day'].apply(lambda x: x.split('-')[2])
    df_sel['D'] = df_sel['D'].apply(lambda x: eval(x) if x[0] != '0' else eval(x[1]))

    # get min score_cost:
    min_score_to_cost = np.nanmin(df_sel.query('views>=1').Score_to_cost.values)
    # print 'min_score_to_cost ', min_score_to_cost
    # print 'set(df_oneday.campaign_id)',set(df_oneday.campaign_id)

    for cid, tmp in df_sel.groupby('campaign_id'):

        mean_slope = min_score_to_cost

        # if cid is not in the df_oneday, use a historical mean; otherwise use overall min score/cost
        if len(df_oneday.query('campaign_id == %d' % cid)) < 1 or sum(tmp.views.values) < 1:
            sc_local = df_sel.query('campaign_id == %d' % cid).Score_to_cost.values
            if len(sc_local) > 0:
                mean_slope = np.nanmean(sc_local)
                # print cid,' is not in df_oneday, mean_slope = ',mean_slope
                # print 'sc_local', sc_local
            coeffs_all[cid] = np.array([mean_slope, Beta_max])
            continue

        # provide initial values for coefficients
        tmp_oneday_cid = df_oneday.query('campaign_id == %d' % cid)

        if float(tmp_oneday_cid['cost']) > 0:
            mean_slope = float(tmp_oneday_cid['Score_to_cost'].values)
        coeffs = coeffs_def = np.array([mean_slope, Beta_max])
        tmp = tmp.query('impressions>=1 and views>=1 and VR<1 and Score>0')
        tmp = tmp.dropna(subset=['Score', 'cost'])

        # Score only campaigns with al least 3 day data (use default average values otherwise)
        if len(tmp) >= Npoints_min:
            # Make curve fit (Score vs cost)
            vr = tmp.VR.values
            views = tmp.views.values
            cost = tmp.cost.values
            scores = tmp.Score.values
            score_to_cost = scores / cost

            # Calculate stat. and syst. uncertainties
            dErr = np.sqrt((1. + ViewRate_imp ** 2 * (1 - vr)) / views)  # np.sqrt(1./views)
            dErr = np.sqrt(dErr ** 2 + Model_accuracy ** 2)
            # Suppress by error older dates
            time_diff = max(tmp.D.values) - tmp.D.values
            dErr = dErr / (0.9 ** time_diff)
            Err = scores * dErr + 0.1

            Alpha_init = sum(scores) / sum(cost)
            Beta_init = np.nanmean(cost)
            Alpha_min, Alpha_max = np.nanmin(score_to_cost), np.nanmax(score_to_cost)
            Prob = 0.
            try:
                # print 'Inside try' #p0=[Alpha_init,Beta_init]
                coeffs, cov = curve_fit(func1, cost, scores, bounds=([Alpha_min, Beta_min], [Alpha_max, Beta_max]),
                                        sigma=Err, absolute_sigma=True, method='dogbox', ftol=1.e-2, xtol=1.e-2)
                # Estimate upper and bottom limits for the Score fits:
                # df0, scale  = utils.make_df(tmp, coeffs, cov, dErr, 1.0,'Score')
                # Use default values if the fit probability is low:
                # Chi2, Ndf, Prob = get_chi2_prob(df0.fit.values, df0.Score.values, df0.Err.values)
                # self.plot.plot_scores(df0,cid,'Score',Chi2, Ndf, Prob,1,coeffs)
                # if Prob < Prob_lower_limit or np.isnan(Prob) or Ndf<1:
                #    coeffs = coeffs_def
                #    cov = cov_def
            except:
                coeffs = coeffs_def

        coeffs_all[cid] = list(coeffs)

    #print 'coeffs_all', coeffs_all

    return coeffs_all
