import pandas as pd
import numpy as np
import fits


def ucb(mean, n, nj, conf_fact):
  return mean + np.sqrt(conf_fact * np.log(n) / (nj + 0.1))
  
def Good_Budget(b_ratio):
    #return abs(1.0 - b_ratio) < 0.01 
    #print 'Good_Budget: b_ratio', b_ratio
    return b_ratio > 0.995 and b_ratio < 1.02

def Good_CPV(b,cpv,cpv_lim): 
    cpv_res = sum(b*cpv)/sum(b)
    #print 'Good_CPV (obs, lim, obs/lim):',cpv_res, cpv_lim, cpv_res/cpv_lim
    return (cpv_lim < 1.e-6 or cpv_res/cpv_lim < 1.01)

def Good_VR(b,vr,vr_lim): 
    vr_res = sum(b*vr)/sum(b)
    #print 'Good_VR (obs, lim, obs/lim):',vr_res, vr_lim, vr_res/vr_lim
    return (vr_lim < 1.e-6 or vr_res/vr_lim > 0.98)


def empty_df(message):
    df_empty = pd.DataFrame({"Status": [message]})
    df_empty.index = ['']
    return df_empty


def outlier_cut(x, x_lo, x_up, coef=1.5):
    x = np.array(x)
    x = x[x > x_lo]
    x = x[x < x_up]
    irq1 = np.percentile(x, [25])[0]
    irq3 = np.percentile(x, [75])[0]
    irq = irq3 - irq1
    cut1 = irq1 - coef * irq
    cut2 = irq3 + coef * irq
    return cut1, cut2


def hour_group(day, h, nhg, Day_today, last_hour):
    if day == Day_today:
        return '0'
    nh = int((23 - last_hour) / nhg) + 1
    return str((h - last_hour) / nh)


def get_vr_importance(CPV_ratio, VR_ratio, VR_imp_max=2.0):
    if VR_ratio > 1.95:
        VR_ratio = 1.95
    if CPV_ratio > 1:
        CPV_ratio = 0.99
    CPV_ratio_critical = 0.99
    VR_imp = VR_imp_max - (VR_imp_max / CPV_ratio_critical) * CPV_ratio ** (VR_imp_max - VR_ratio)
    return min(VR_imp_max, max(0.05, VR_imp))


def max_budget_fraction(dailyBudget, nli):
    a, b, c = [0.47117431,  0.10613777, -0.48182186]
    return a * dailyBudget ** b * nli ** c

def make_df_cpv(tmp, coeffs, cov, Nsigma, var):
    fitted_Y = fits.func1(tmp.cost.values, *coeffs)
    dY = fits.d_func1(tmp.cost.values, cov, coeffs)
    Err = np.sqrt(dY**2 + tmp['Err'].values**2)
    dY_full = np.sqrt(dY**2 + (0.01 * fitted_Y)**2)
    Chi2, Ndf, Prob = fits.get_chi2_prob(fitted_Y, tmp[var].values, Err)

    scale = 1.0
    if Chi2 / Ndf > 1.0:
        scale = np.sqrt(Chi2 / Ndf)
    Err_fit = dY_full * scale
    corr_pos = 1. + Nsigma * Err_fit / fitted_Y

    return corr_pos[-1]


def gamma(min_corr_fact, frac):
    max_corr_fact = 1.0
    slope = (max_corr_fact - min_corr_fact) / np.log(100.)
    return max_corr_fact - slope * np.log(frac + 1.)


def budget_corr_fact_max(ndays_passed, ndays_left, VR_imp):
    ndays = ndays_passed + ndays_left
    max_corr_fact = 2.0 * np.log(2.72 + VR_imp)
    min_corr_fact = 1.2
    slope = (max_corr_fact - min_corr_fact) / np.log(ndays)
    return max_corr_fact - slope * np.log(ndays_passed + 1)


def make_df(tmp, coeffs, cov, dErr, Nsigma, var):
    Reg_uncert = 0.02
    fitted_Y = fits.func1(tmp.cost.values, *coeffs)
    dY = fits.d_func1(tmp.cost.values, cov, coeffs) + Reg_uncert
    Err_data = tmp[var].values * np.sqrt(dErr ** 2 + Reg_uncert ** 2)
    df = pd.DataFrame(
        {'cost': tmp.cost.values, var: tmp[var].values, 'fit': fitted_Y, 'Err_fit': dY, 'Err_data': Err_data})
    df['Err'] = np.sqrt(df['Err_fit'] ** 2 + df['Err_data'] ** 2)
    df = df.sort_values(by=['cost'], ascending=True)
    Chi2, Ndf, Prob = fits.get_chi2_prob(df.fit.values, df[var].values, df.Err.values)
    scale = 1.0
    if Chi2 / Ndf > 0.0:
        scale = np.sqrt(Chi2 / Ndf)
    df['Err_fit'] = df['Err_fit'] * scale
    df['fit_pos'] = df['fit'] + Nsigma * df['Err_fit']
    df['fit_neg'] = df['fit'] - Nsigma * df['Err_fit']
    df['Err'] = np.sqrt(df['Err_fit'] ** 2 + df['Err_data'] ** 2)

    return (df, scale)
