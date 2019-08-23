import numpy as np
import pandas as pd
import fits
import utils
from scipy.stats import t
from scipy.optimize import curve_fit
import scipy
import scipy.integrate as integrate
from numpy import pi, exp
import time
from optimization import constants
from dateutil import parser
from datetime import timedelta

const = constants.Const()

def get_CPV_bid(df_oneday, df_camp_factor_in, Day_today, CPV_max, CPV_project, ratio_prj,
                cost_increase_max, dow_today, last_hour, nhg, DBM, opt_type, network, bumper_ad):

    def get_cpvs(tmp_day):
        tmp_day = tmp_day.query('views>0 and CPV<%f' % (const.K_CPV_MAX * CPV_max))
        if len(tmp_day) < 4:
            return (0, 0, 0)
        cpvs = tmp_day.CPV.values
        costs = tmp_day.cost.values
        views = tmp_day.views.values
        cpv_mean = np.nansum(costs) / np.nansum(views)
        cpv_std = np.nanstd(cpvs)
        n_std = t.ppf(0.90, len(cpvs))  # max(1.645,t.ppf(0.90,len(cpvs)))
        return (cpv_mean, cpv_std, n_std)

    if opt_type == 0:
        last_hour = 0

    # consider only last 4 days + today
    day_min = parser.parse(Day_today) - timedelta(days=const.DAYS_WINDOW)
    day_min = day_min.strftime("%Y-%m-%d")
    df_camp_factor = df_camp_factor_in.query('views > 0 and day>"%s" and (day=="%s" or hour_of_day>=%d)' % (day_min, Day_today, last_hour))

    # Create a dictionary with CPV mean and std values for today with campaign_id as a key
    df_today = df_camp_factor.query("day=='%s'" % Day_today)
    g = df_today[['CPV', 'cost', 'views', 'campaign_id']].groupby('campaign_id')
    cpv_today_dict = g.apply(get_cpvs).to_dict()

    df_camp_factor['day_hour_group'] = np.array([str(d) + '_' + utils.hour_group(d, h, nhg, Day_today, last_hour) for d, h
                                                 in zip(df_camp_factor.day.values, df_camp_factor.hour_of_day.values)])
    tmp_sum1 = df_camp_factor[['day','day_hour_group', 'dow_factor', 'campaign_id', 'cost', 'views']].groupby(
        ['day','dow_factor', 'day_hour_group', 'campaign_id']).sum()
    tmp_sum1.reset_index(inplace=True)
    tmp_sum2 = df_camp_factor[['campaign_id', 'campaign_state', 'day_hour_group', 'CPV']].groupby(
        ['campaign_id', 'campaign_state', 'day_hour_group']).max()
    tmp_sum2.reset_index(inplace=True)
    tmp_sum = pd.merge(tmp_sum1, tmp_sum2, on=['campaign_id', 'day_hour_group'])
    tmp_sum = tmp_sum.query('CPV>0')

    def make_fit(tmp):
        # https://github.com/scipy/scipy/blob/master/scipy/optimize/_lsq/least_squares.py
        if len(tmp) < 2:
            return (np.array([0.9 * CPV_max, 0.10]), 1.)

        tmp = tmp.sort_values('cost')
        dErr = 1. / np.sqrt(tmp.views)
        # Redefine errors by a day order: larger for older days
        days = map(lambda x: parser.parse(x), tmp.day.values)
        days = parser.parse(max(tmp.day.values)) - np.array(days)
        time_diff = np.array(map(lambda x: x.days, days))
        tmp['dErr'] = dErr / (0.80 ** time_diff)
        tmp['dErr'] = np.sqrt(tmp.dErr.values ** 2 + 0.03 ** 2)
        tmp['Err'] = tmp.dErr * tmp.CPV

        cpv_max = max(tmp.CPV) * cost_increase_max ** const.COST_POWER_MAX
        cpv_min = 0.01 if cpv_max > 0.01 else 0.5 * cpv_max
        corr_pos_max = 1.3
        cost_power_min = 0.01
        if bumper_ad:
            cpv_min = 1.0 if cpv_max > 1 else 0.5 * cpv_max
            corr_pos_max = 1.07
            cost_power_min = 0.001

        coeffs, cov = curve_fit(fits.func1, tmp.cost, tmp.CPV, bounds=([cpv_min,cost_power_min],[cpv_max,const.COST_POWER_MAX]), method='dogbox')
        # sigma=tmp.Err, absolute_sigma=True, ftol=1.e-2, xtol=1.e-2)
        corr_pos = utils.make_df_cpv(tmp, coeffs, cov, 1, 'CPV')

        return coeffs, min(corr_pos_max, corr_pos)


    coeffs_corr_dict = tmp_sum.groupby(['campaign_id']).apply(make_fit).to_dict()  # .max().to_dict()

    big_project_cpv = CPV_project / CPV_max > 0.95
    prj_bottom_offset_min = -0.10
    budget_opt_d = df_oneday.set_index('campaign_id')['Budget'].to_dict()
    cpv_max_abs = 0.9 * CPV_max

    cpv_bid_arr = {}
    cpv_max_arr = {}
    for cid, tmp in tmp_sum.groupby('campaign_id'):
        # Get max CPV for now (now is today for enabled campaigns and last day for paused)
        today_loc = max(tmp.day)
        tmp_now = tmp.query('day == "%s" and CPV < %f' % (today_loc, const.K_CPV_MAX * CPV_max))
        cost_today = 0
        if today_loc == Day_today and len(tmp_now) > 0:
            cost_today = sum(tmp_now.cost)

        if len(tmp_now) < 1 or cid not in budget_opt_d or 'paused' in tmp.iloc[0].campaign_state:
            cpv_bid_arr[cid] = 1.0
            cpv_max_arr[cid] = cpv_max_abs
            continue

        # sigma is chosen in the fair price definition to deliver daily budget
        n_sigma = get_nsigma(ratio_prj)

        if const.PRINT_DETAILS > 1:
            print '\nget_CPV_bid: cid, n_sigma, ratio_prj',cid, n_sigma, ratio_prj

        if const.PRINT_DETAILS > 1:
            print tmp[['campaign_id', 'campaign_state', 'day_hour_group', 'cost','views','CPV']]

        # Optimized budget for today
        budget_max_new = float(budget_opt_d[cid]) + 0.01
        if opt_type == 0:
            budget_max_new = max(0.25 * budget_max_new, budget_max_new - cost_today)
            #if ratio_prj < const.BUDGET_RATIO_CRITICAL:
            #    n_sigma += 1

        #Increase CPV for inSearch projects 
        #n_sigma += apply_network_handicap(network)

        cpv_max_all = max(tmp['CPV']) * cost_increase_max ** const.COST_POWER_MAX
        cpv_max_fit_new = 1.1 * np.nanmax(tmp.CPV.values)
        if len(tmp) > 3:
            coeffs, corr = coeffs_corr_dict[cid]
            cpv_max_fit_new = get_fair_CPV(coeffs, corr, budget_max_new, cpv_max_all, n_sigma)

        cpv_corr_tow = 1.0
        #Correction of bids for time & day of week:
        # if opt_type == 1:
        #    cpv_corr_tow = get_tow_corr(tmp, coeffs, CPV_max, dow_today)

        # Find mean and variation of today's CPVs
        rule = 0
        if cid in cpv_today_dict:
            cpv_mean_now, cpv_std_now, n_std = cpv_today_dict[cid]
            max_crit = cpv_mean_now + n_std * cpv_std_now
            max_expd = cpv_mean_now + 2.0 * cpv_std_now
            # increase CPV bid to have all data within the new limit with C.L.=CL if
            # (a) the campaign spends too slow, and (b) CPV is not high
            if max_crit / cpv_max_fit_new > 1.05 and (
                ratio_prj - 1.) < prj_bottom_offset_min and not big_project_cpv and not DBM:
                cpv_max_fit_new = max_expd
                rule = 1
        cpv_bid_camp = cpv_max_fit_new / cpv_max_abs

        # Overall bid
        bid = cpv_corr_tow * cpv_bid_camp
        if const.PRINT_DETAILS > 1:
            print 'get_CPV_bid: bid', cid, bid, cpv_corr_tow, cpv_bid_camp, cpv_max_fit_new, cpv_max_abs,rule
        cpv_bid_arr[cid] = round(bid, 4)
        cpv_max_arr[cid] = round(bid * cpv_max_abs, 4)

    return (cpv_bid_arr, cpv_max_arr)


def apply_network_handicap(network):
    """
    Returns integer to be added to nsigma
    :param network: String name of network
    :return:
    """
    if network == "InSearchYouTube" or network == "InDisplayYouTube":
        return 2
    else:
        return 0


def get_fair_CPV(coeffs, corr_pos, new_cost, cpv_max_all, n_sigma):
    cpv_mean_new_cost = min(fits.func1(new_cost, *coeffs), cpv_max_all)
    cpv_max_new_cost = min(cpv_mean_new_cost * corr_pos, cpv_max_all)
    min_std = max(0.003, 0.05*cpv_mean_new_cost)
    #min_std = max(0.002, 0.05 * cpv_mean_new_cost)
    cpv_std = max(min_std,cpv_max_new_cost - cpv_mean_new_cost)
    if const.PRINT_DETAILS > 1:
        print 'get_CPV_bid: new_cost, cpv_max_new_cost, cpv_mean_new_cost', new_cost, cpv_max_new_cost, cpv_mean_new_cost, cpv_max_all, coeffs
    cpv_max_new_cost = cpv_mean_new_cost + n_sigma * cpv_std
    if const.PRINT_DETAILS > 1:
        print 'get_CPV_bid: n_sigma, cpv_std, cpv_max_new_cost', n_sigma, cpv_std, cpv_max_new_cost
    return cpv_max_new_cost


def get_tow_corr(tmp_in, coeffs, CPV_max, dow_today):
    tmp = tmp_in.query('CPV<%f' % (K_cpvmax * CPV_max))
    if len(tmp) < 3:
        return 1.0

    def cpv_budget_aver(dow):
        tmp_dow = tmp.query('dow_factor == "%s"' % (dow))
        if len(tmp_dow) < 3:
            tmp_dow = tmp
        ndays = len(set(tmp_dow.D))
        tot_views = sum(tmp_dow.views.values)
        CPV_aver = sum(tmp_dow.cost.values) / tot_views
        CPV_aver_err = CPV_aver / np.sqrt(tot_views + 0.5)
        B_aver = sum(tmp_dow.cost.values) / ndays
        return CPV_aver, CPV_aver_err, B_aver

    CPV_all, CPV_all_err, B_all = cpv_budget_aver('')
    CPV_tow, CPV_tow_err, B_tow = cpv_budget_aver(dow_today)
    tow_cpv_corr = (CPV_tow / CPV_all) * (B_all / B_tow) ** coeffs[1]
    tow_cpv_corr_err = tow_cpv_corr * np.sqrt((CPV_all_err / CPV_all) ** 2 + (CPV_tow_err / CPV_tow) ** 2)
    if abs(tow_cpv_corr - 1) < tow_cpv_corr_err:
        tow_cpv_corr = 1.0
    if tow_cpv_corr > 1.10:
        tow_cpv_corr = 1.10
    elif tow_cpv_corr < 0.90:
        tow_cpv_corr = 0.90

    return tow_cpv_corr


def get_nsigma(r_prj):
    nsigma = 24.5*np.exp(-3.22*r_prj) #nsigma(0.5) = 5, nsigma(1) = 1, nsigma(2) = 0.04
    #nsigma = 3 - (r_prj - 0.5) * (2. / 0.5)
    # nsigma = np.sign(nsigma)*nsigma**2
    return min(5, max(0, nsigma))
