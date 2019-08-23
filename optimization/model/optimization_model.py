# pylint: skip-file
import os
import time

from datetime import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings

from CPV_bids import get_CPV_bid, get_fair_CPV
from budget_optimizer import budget_optimize
import fits
from fits import fit_scores
from optimization import constants
from optimization.model.order import budget_order, view_order
import utils

warnings.filterwarnings("ignore")
const = constants.Const()

MAIN_OPT_FACTORS = ['video_id', 'age_range', 'gender', 'device']
MAIN_METRICS = ['impressions', 'views', 'completed_views', 'clicks', 'earned_subscribers', 'earned_views','cost','conversions']

class OptimizationModel:
    def __init__(self, ccf_data, optimization_params, adwords_metrics, project_metrics, project_camps_miss_active,
                 summary_metrics):
        # Static initialization values for stateful processing
        self.Budget_per_day = 1.
        self.VR_target_hist = -1
        self.VR_target = -1
        self.ViewRate_imp = 0.5
        self.optimize_by_factors = False
        self.CPV_opt = True
        self.Dynamic_scores = True
        self.coeffs_tot = {}
        self.grouping_metrics = ['campaign_id', 'campaign', 'campaign_state']
        self.df_camp_fail = pd.DataFrame()
        self.df_camp_toenable = pd.DataFrame()
        self.budget_link = ''
        self.pause_link = ''
        self.enable_link = ''
        self.Budget_frac_diff_today = 0
        self.budget_by_video = False
        self.budget_by_age = False
        self.budget_by_gender = False
        self.budget_by_device = False
        self.budget_by_placement = False
        self.factor_name = ''
        self.factors = ['']
        self.b_frac = [1.0]
        self.instream_frac = 1.
        self.hour_factor = 1.04
        self.Budget_to_expect = {}
        self.ccf_prj_expected = 1.

        # unpack params passed in
        self.data_cost_vs_hour = ccf_data

        self.Strike_id = project_metrics['strike_id']
        self.Account_id = project_metrics['account_id']
        self.start_date = project_metrics['start_date']
        self.end_date = project_metrics['end_date']
        self.update_time = project_metrics['update_time']
        self.project_camps_miss_active = project_camps_miss_active

        self.Budget_total_daily = optimization_params['daily_budget_remaining']
        self.VR_min = optimization_params['view_rate_min']
        self.VR_choice = optimization_params['view_rate_choice']
        self.CPV_max_form = optimization_params['cpv_max_form']
        self.second_KPI = project_metrics['df_utopsa']['kpi'].iloc[0]
        self.ViewRate_imp_form = optimization_params['view_rate_imp']
        self.day_of_data = optimization_params['day_of_data']
        self.budget_dict_by_factors = optimization_params['budget_dict_by_factors']
        self.opt_type = optimization_params['opt_type']
        self.enable_paused =  False     #optimization_params['enable_paused']
        self.monthly_transition = False #optimization_params['monthly_transition']
        self.DBM = optimization_params['dbm']
        self.opt_factors = optimization_params['budget_by_factors']

        self.df_camp = adwords_metrics['df_campaign']
        #replace NaN with 0
        self.df_camp.fillna(0, inplace=True)

        self.camp_status_dict = adwords_metrics['campaign_status_dict']

        self.b_max_dict = adwords_metrics['b_max_dict']
        self.cpv_max_dict = adwords_metrics['cpv_max_dict']

        self.df_sum = summary_metrics['df_summary']
        self.hours_since_CPV_update = int(self.df_sum.hours_since_cpv_update)
        self.views_since_CPV_update = int(self.df_sum.views_since_cpv_update)
        self.last_cpv_bid_hour = self.df_sum.last_cpv_bid_hour.values[0]
        self.last_cpv_bid_day = self.df_sum.last_cpv_bid_day.values[0]
        self.media_spend_total = float(self.df_sum.media_spend_total)
        self.media_spend_today = float(self.df_sum.media_spend_today)
        self.Budget_spent_frac_by_now = float(self.df_sum.budget_spend_fraction)
        self.Budget_spent_yesterday = float(self.df_sum.budget_spent_yesterday)
        self.Budget_exp_for_yesterday = float(self.df_sum.budget_exp_for_yesterday)
        self.days_left = int(self.df_sum['days_left'])
        self.days_passed = int(self.df_sum['days_passed'])
        self.CPV_max = float(self.df_sum.max_cpv)
        self.CPV_project = float(self.df_sum.cpv_so_far)
        self.min_budget = 3 * self.CPV_project
        self.last_hour = int(self.df_sum.last_hour)

        self.today = self.df_sum.today.values[0]
        self.yesterday = self.df_sum.yesterday.values[0]
        self.dow_today = self.df_sum.dow_today.values[0]
        self.media_order = float(self.df_sum.media_order)
        self.Vertical = self.df_sum.vertical.values[0]
        D_today = self.today.split('-')[2]
        self.D_today = eval(D_today) if D_today[0] != '0' else eval(D_today[1])
        self.Gamma = const.GAMMA
        self.max_corr = 1.0
        self.network = project_metrics['network']
        self.currency = project_metrics['df_utopsa']['currency'].iloc[0]

        self.CPV_max_cap = self.CPV_max * const.K_CPV_MARGIN
        self.CPV_max_limit = self.CPV_max
        self.key_var = 'views'

        df_today = self.df_camp.query('day == "%s"'%self.today)
        self.view_order = float(summary_metrics['view_order'])
        self.Views_total_daily = optimization_params['daily_views_remaining']
        self.views_delivered_total = float(self.df_sum.views_so_far) 
        self.views_seen_today = sum(df_today.views)
        self.VO_optimization = summary_metrics['performance_goal'] == 'ViewOrder'

        # and set to True for flights with bumper ads: see QUAN-679 for more details.
        self.bumper_ad = False #project_metrics['bumper_ad']
        if self.bumper_ad:
            self.df_camp['CPV'] = 1000.*self.df_camp['cost']/(self.df_camp['impressions']+const.EPS**2)
            self.df_camp['views'] = self.df_camp['completed_views']
            self.df_camp['VR'] = self.df_camp['VCR']
            self.CPV_project =  sum(self.df_camp['cost'])/(sum(self.df_camp['impressions'])+const.EPS)
            self.min_budget = self.CPV_project

        # Order calculation initialization
        self.order_calculator = self._get_order_calculator()

        #Map Kevin Secondary KPI names to OE internal names:
        if self.second_KPI == 'Clicks':
            self.second_KPI = 'CTR'
            self.key_var = 'clicks'
        elif self.second_KPI == 'VideoCompletionRate':
            self.second_KPI = 'VCR'
            self.key_var = 'completed_views'
        elif self.second_KPI == 'EarnedSubscribers':
            self.second_KPI = 'SR'
            self.key_var = 'earned_subscribers'
        elif self.second_KPI == 'EarnedViews':
            self.second_KPI = 'EVR'
            self.key_var = 'earned_views'
        elif self.second_KPI == 'Conversions':
            self.second_KPI = 'CR'
            self.key_var = 'conversions'
        else: #ViewRate
            self.second_KPI = 'VR'
            self.key_var = 'views'

        self.allow_sec_kpi = False

    def run(self):
        df_empty = pd.DataFrame()

        if const.ADDITIONAL_SEC_KPI == 1 or (const.ADDITIONAL_SEC_KPI >= 0 and self.second_KPI != 'VR'):
            return (utils.empty_df("Additional secondary KPI feature is not tested yet. "
                                   "Please set ADDITIONAL_SEC_KPI to 0, second_KPI to VR and re-run again."),
                    df_empty, df_empty, df_empty, df_empty, df_empty)
        elif const.ADDITIONAL_SEC_KPI == -1:
            self.allow_sec_kpi = True  # redefine it in case ADDITIONAL_SEC_KPI is -1 (needed for the research OE)

        if len(self.df_camp) < 1:
            return (utils.empty_df(
                "Empty dataframe. No active campaigns for today or yesterday."
                "Wait till campaigns will be active and then run again."),
                    df_empty, df_empty, df_empty, df_empty, df_empty)

        #Filter out paused campaigns if the flag enable_paused is set to False:
        if (not self.enable_paused):
            self.df_camp = self.df_camp.query('campaign_state == "enabled"')

        #Remove audience_group from mandatory budget factos
        if 'audience_group' in self.budget_dict_by_factors:
            del self.budget_dict_by_factors['audience_group']

        self.ccf_prj_expected = 1.0 #float(self.df_sum.budget_frac_diff_today)

        self.Budget_ratio_prj = self.order_calculator.get_budget_ratio(
            self.days_passed,
            self.days_left,
            self._get_order_today(),
            self._get_order_delivered(),
            self._get_order_total()
        )

        self.Budget_total_daily = self.order_calculator.get_total_daily_budget(
            self.Budget_ratio_prj,
            self.df_camp,
            self.CPV_project,
            self._get_limiting_factor()
        )
        
        if self.Budget_ratio_prj < 0.95:
            self.enable_paused = True

        self.Budget_for_flight_rem = self.Budget_total_daily * self.days_left
        self.Budget_for_flight_tot = self.media_spend_total + self.Budget_for_flight_rem
        self.df_sum['budget_per_day'] = self.Budget_total_daily
        self.df_sum['Budget_for_flight_rem'] = self.Budget_for_flight_rem
        self.df_sum['Budget_for_flight_tot'] = self.Budget_for_flight_tot        

        self.Budget_per_day = {'': self.Budget_total_daily}
        self.df_arr = {'': self.df_camp}
        # Get mandatory split factors and form new Budget_per_day
        factors = self.budget_dict_by_factors.keys()

        if len(factors) > 0:
            if len(self.opt_factors) == 0:
                self.opt_factors = factors
                self.Budget_per_day, self.df_arr = self.get_kevin_factors()
            else:
                self.optimize_by_factors = True
                self.opt_factors = list(set(factors) & set(self.opt_factors))
                if len(self.opt_factors) == 0:
                    self.opt_factors = factors
                self.grouping_metrics += self.opt_factors
                self.grouping_metrics = list(set(self.grouping_metrics))
        else:
            self.opt_factors = []

        if self.DBM and 'video_id' not in self.grouping_metrics:
            self.grouping_metrics.append('video_id')

        if len(self.Budget_per_day) < 1:
            return utils.empty_df("Empty Budget_per_day dictionary. Wrong factors are chosen"), [df_empty, df_empty,
                                                                                                 df_empty, df_empty]
        if (self.Budget_spent_frac_by_now < 0.10):
            self.CPV_opt = False

        if self.monthly_transition:
            self.CPV_opt = True
            self.opt_type = 0  # Greedy optimization

        # Optimize budget
        def optimization_in_loop(df_oneday):
            num = 4
            cfact = np.linspace(1.0, 2, num=num)
            if not self.Dynamic_scores:
                cfact = np.array([1.])
            status = -1
            df_result = pd.DataFrame()
            min_budget = 1.e-2
            if self.monthly_transition:
                min_budget = 2 * float(self.CPV_project)
                cfact = [1.0]
            for ix, cf in enumerate(cfact, start=0):
                df_oneday_en = df_oneday.query('campaign_state == "enabled"')
                df_oneday_toen = df_oneday.query('campaign_state != "enabled"')
                status, good_cpv, good_vr, df_result = budget_optimize(df_oneday_en, cf, self.Dynamic_scores, self.days_passed, self.coeffs_tot,
                                                        self.VR_target, self.CPV_max_limit, self.monthly_transition, self.Budget_total_max)

                df_result = pd.concat([df_result, df_oneday_toen]).fillna(const.EPS)

                if self.Dynamic_scores and status != 0 and ix < num:
                    continue

                b_ratio = sum(df_result['Budget']) / self.Budget_total_max
                good_budget =  utils.Good_Budget(b_ratio)

                if const.PRINT_DETAILS > 0:
                    print 'Target_daily_budget, Total opt. budget', self.Budget_total_max, sum(df_result['Budget'])
                    print 'optimization_in_loop: good_budget, good_cpv, good_vr', b_ratio, good_budget,good_cpv, good_vr

                if (not good_budget or not good_vr or not good_cpv) and ix < num:
                    continue

                break

            return status, df_result

        df_combo = pd.DataFrame()
        self.df_camp_factor = pd.DataFrame()
        status = -1
        for fn, B_pd in self.Budget_per_day.iteritems():

            self.Dynamic_scores = True
            self.CPV_opt = True

            self.df_camp_factor = self.df_arr[fn]
            self.Budget_total_max = B_pd
            self.Budget_spent_today = np.sum(self.df_camp_factor.query('day=="%s"' % self.today)['cost'])
            self.Budget_spent_fraction = min(0.9999, self.Budget_spent_today / self.Budget_total_max)
            self.Gamma = utils.gamma(0.5, 100. * self.Budget_spent_fraction)
            if self.second_KPI == 'CTR' or self.second_KPI == 'SR' or self.second_KPI == 'EVR' or self.second_KPI == 'CR':
                self.Gamma = 1

            # skip campaigns if budget-per-day is small or nothing is spent today, or no active campaigns
            if B_pd < 0.01:
                continue

            df_active = self.df_camp_factor.query('campaign_state == "enabled"')
            if len(df_active) < 1:
                continue

            # calculate scores
            df_camp_grp_sc, df_camp_fail, df_oneday_toenable = self.calculate_scores(self.df_camp_factor)

            if len(df_camp_grp_sc) < 1:
                continue

            # run budget optimization
            status, df_camp_grp = optimization_in_loop(df_camp_grp_sc)
            if self.Dynamic_scores and status > 0:
                self.Dynamic_scores = False
                status, df_camp_grp = optimization_in_loop(df_camp_grp_sc)

            # get CPV bids
            df_camp_grp = self.CPV_bidder(df_camp_grp)

            # append to the overall campaign list
            df_combo = df_combo.append(df_camp_grp)

            self.df_camp_fail = self.df_camp_fail.append(df_camp_fail)

            self.df_camp_toenable = self.df_camp_toenable.append(df_oneday_toenable)

        if len(df_combo) < 1:
            return (utils.empty_df("Empty optimization dataframe."),
                    df_empty, df_empty, df_empty, df_empty, df_empty)

        # report final results
        self.df_out, df_opt_res = self.report_results(df_combo, self.df_sum)

        # create three independent data frames: main(w/ enabled LIs only), 'to_pause' and 'to_enable'.
        self.make_three_dfs()

        df_fact, df_exp_b = df_empty, df_empty
        # if self.optimize_by_factors:
        if len(self.opt_factors) > 0:
            df_fact, df_exp_b = self.make_factor_tables(df_combo)

        return (self.df_out, df_fact, df_exp_b, self.df_camp_fail, self.df_camp_toenable, df_opt_res)

    def make_three_dfs(self):
        columns_keep = self.df_out.columns
        # Make DF with campaigns to enable:
        if len(self.df_camp_toenable) > 0:
            camps_main_only = list(set(self.df_out.campaign_id.values) - set(self.df_camp_toenable.campaign_id.values))
            camps_main_and_toenable = list(
                set(self.df_out.campaign_id.values) & set(self.df_camp_toenable.campaign_id.values))
            self.df_camp_toenable = self.df_camp_toenable.append(
                self.df_out[self.df_out['campaign_id'].isin(camps_main_and_toenable)])
            self.df_camp_toenable = self.df_camp_toenable.drop_duplicates('campaign_id', keep='last')[columns_keep]
            self.df_out = self.df_out[self.df_out.campaign_id.isin(camps_main_only)]

        # Make DF with campaigns to pause:
        if len(self.df_camp_fail) > 0:
            cols_toadd = list(set(self.df_out.columns) - set(self.df_camp_fail.columns))
            for col in cols_toadd:
                self.df_camp_fail[col] = ''
            self.df_camp_fail['Budget'] = 0.01
            self.df_camp_fail['CPV_max'] = self.CPV_max


    def CPV_bidder(self, df_camp_grp):
        # Get CPV bids:
        cpv_bids = [1.] * len(df_camp_grp)
        cpv_max_vals = [-1] * len(df_camp_grp)
        cpv_rules = [''] * len(df_camp_grp)
        t0 = time.time()
        if self.CPV_opt:  # and self.days_passed>0: # and self.last_hour>=self.CPV_bid_hour_min and self.last_hour<=self.CPV_bid_hour_max:
            nhg = self.get_nhour_groups()
            camps_list = list(set(df_camp_grp.campaign_id.values))
            self.df_camp_factor = self.df_camp_factor[self.df_camp_factor['campaign_id'].isin(camps_list)]

            cpv_bids, cpv_max_vals = get_CPV_bid(df_camp_grp, self.df_camp_factor, self.today, self.CPV_max, self.CPV_project,
                                                 self.Budget_ratio_prj, self.max_corr, self.dow_today, self.last_hour, nhg, self.DBM,
                                                 self.opt_type, self.network, self.bumper_ad)

        df_camp_grp['CPV_bid'] = df_camp_grp.apply(
            lambda r: cpv_bids[r['campaign_id']] - 1.0 if r['campaign_id'] in cpv_bids else 0.0, axis=1)
        df_camp_grp['CPV_bid'] = df_camp_grp['CPV_bid'].apply(lambda x: x if x > -0.89 else -0.89)
        df_camp_grp['CPV_bid'] = df_camp_grp['CPV_bid'].apply(lambda x: x if x < 8.9 else 8.9)
        df_camp_grp['CPV_rule'] = cpv_rules

        #LI CPV max should not be more that const.K_CPV_MAX * CPV_max_project
        CPV_max_abs = const.K_CPV_MAX * self.CPV_max
        cpv_max_vals = {k: (v if v <= CPV_max_abs else CPV_max_abs) for k,v in cpv_max_vals.items()}

        CPV_max = 0.9*self.CPV_max
        df_camp_grp['CPV_max'] = df_camp_grp.apply(
            lambda r: cpv_max_vals[r['campaign_id']] if r['campaign_id'] in cpv_max_vals else CPV_max, axis=1)

        #CPV_max should be <= Budget
        df_camp_grp['CPV_max'] = df_camp_grp.apply(lambda r: r['CPV_max'] if r['CPV_max'] <= r['Budget'] else r['Budget'], axis=1)

        if const.PRINT_DETAILS > 1:
            print 'CPV_bidder: assigned CPV_max values:'
            print df_camp_grp[['campaign_id','Budget','CPV_max']]

        def get_new_cpv_max(x):
            int_cents = int(x * 100) / 100.
            if x - int_cents < 0.001:
                x = int_cents
            else:
                x = int_cents + 0.01
            if self.Budget_ratio_prj < const.BUDGET_RATIO_CRITICAL:
               x += 0.01
            return max(0.01, x)
            #return x

        if self.DBM:
            df_camp_grp['CPV_max'] = df_camp_grp['CPV_max'].apply(lambda x: get_new_cpv_max(x))

        # make bid=1, if a bidding correction is small (<=2%)
        df_camp_grp['CPV_bid'] = df_camp_grp['CPV_bid'].apply(
            lambda x: round(100 * x, 0) if abs(1 - x) > 0.02 else 100.)
        df_camp_grp['CPV_rule'] = df_camp_grp.apply(lambda row: row['CPV_rule'] if abs(100 - row['CPV_bid']) > 2 else 0,
                                                    axis=1)

        return df_camp_grp


    def get_kevin_factors(self):
        factors = self.budget_dict_by_factors.keys()
        self.grouping_metrics += factors
        df_arr = {}
        Budget_per_day = {}
        nfact = len(factors)
        b_to_spend = self.media_order - (self.media_spend_total - self.media_spend_today)
        all_factor_values = [self.budget_dict_by_factors[k].keys() for k,v in self.budget_dict_by_factors.items()]
        all_factor_values = set([f for e in all_factor_values for f in e])
        for f, tmp in self.df_camp.groupby(factors):
            # Budget_spent_by_now_total[f] = sum(tmp['cost'])
            f_t = f if np.size(f) > 1 else (f,)
            if len(set(f_t) & all_factor_values) < len(f_t):
                if const.PRINT_DETAILS > 0:
                    print 'get_kevin_factors: Some factors from', f_t, 'are not in our request',  all_factor_values
                continue
            if f == 'None':
                if const.PRINT_DETAILS > 0:
                    print 'get_kevin_factors: None', sum(tmp['cost']), set(tmp.campaign_id)
                continue
            frac = 1.
            for i in range(nfact):
                frac *= self.budget_dict_by_factors[factors[i]][f_t[i]]
            df_arr[f] = tmp
            Budget_per_day[f] = frac * b_to_spend / self.days_left

        # we have to normalize b_frac to have a total sum unity
        f_norm = self.Budget_total_daily / sum(Budget_per_day.values())
        Budget_per_day = {f: max(1.01e-2, b * f_norm) for f, b in Budget_per_day.iteritems()}

        return (Budget_per_day, df_arr)


    def get_max_budget(self, df_oneday, n_toenable):
        if self.days_passed > 0 or self.monthly_transition:
            # print self.df_camp.groupby(['campaign_id','day'])['cost'].sum()
            df_camp_sel = self.df_camp.query('hour_of_day>%d' % self.last_hour)
            df_camp_sum = df_camp_sel[['campaign_id', 'campaign_state', 'day', 'cost', 'views']].groupby(
                ['campaign_id', 'campaign_state', 'day']).sum()
            df_camp_sum.reset_index(inplace=True)
            df_camp_sum = df_camp_sum.sort_values('day')
            # max per-campaign budget increase factor
            self.max_corr = utils.budget_corr_fact_max(self.days_passed, self.days_left, self.ViewRate_imp)
            if self.monthly_transition:
                self.max_corr = 1.2

            #Create a dictionary of per-LI promotional budget scaling factors.
            if self.days_passed > 3: 
                max_promo_corr = 1.5*self.max_corr
                min_promo_corr = 1. / np.sqrt(self.max_corr)
                df_oneday_sel = df_oneday.query(self.key_var+'> 0')
                df_oneday['w'] = df_oneday.Score_to_cost/(np.percentile(df_oneday.query(self.key_var+'> 0').Score_to_cost, [50])[0] + 1.e-3)
                df_oneday['w'] = df_oneday.apply(lambda r: max(min_promo_corr, min(max_promo_corr, r['w'])) if r[self.key_var]>0 else min_promo_corr, axis=1)
            else:
                df_oneday['w'] = 1.0
            weights_dict = df_oneday.set_index('campaign_id')['w'].to_dict()

            frac_days_passed = 100. * self.days_passed / (self.days_passed+self.days_left)
            Gamma_b = utils.gamma(0.5, frac_days_passed)

            #Allowed max budget fraction per LI:
            nli = len(set(df_camp_sum.campaign_id)) + 1
            MAX_B_FRAC = min(const.MAX_B_FRAC, 2.*utils.max_budget_fraction(self.Budget_total_max, nli))

            b_max_tospend_dict = {}
            b_yest_dict = {}
            for cid, tmp in df_camp_sum.groupby('campaign_id'):
                b_max_tospend_dict[cid] = 0.01

                if len(tmp) == 0:
                    continue

                tmp_sorted = tmp.sort_values('day', ascending=False)
                costs = tmp_sorted.cost.values
                cost_yest = costs[0]

                #weighted mean cost:
                w = Gamma_b ** np.arange(len(costs))
                sum_w = sum(w)

                if sum_w < const.EPS or sum(costs) < const.EPS:
                    continue

                w = w / sum_w
                cost_wt_mean = sum(costs * w)
                cost_hist = max(cost_wt_mean, self.min_budget)

                tmp_t = self.df_today.query('campaign_id == %d' % cid)
                cost_today = 0
                if len(tmp_t) > 0:
                    cost_today = tmp_t.cost.values[0]

                #Rescale max budget factors
                max_corr = self.max_corr
                if cid in weights_dict:
                    max_corr *= weights_dict[cid]

                cost_max_fr = max(self.CPV_max, MAX_B_FRAC * self.Budget_total_max - cost_today)
                b_max = min(cost_max_fr, self.max_corr * cost_hist)
                b_min = 0.75*cost_hist

                if const.PRINT_DETAILS > 1:
                	print '\ncid',cid
                	print tmp_sorted
                	print tmp_t[['campaign_id', 'campaign_state', 'day', 'cost', 'views']]
                	print cid, map(lambda x: round(x,2), [cost_hist, cost_yest, cost_today, cost_max_fr, b_max, b_min])

                if b_max < b_min:
                    b_max = b_min

                b_max_tospend_dict[cid] = b_max
                b_yest_dict[cid] = cost_hist

            # Budget yesterday
            df_oneday['B_yest_h'] = df_oneday['campaign_id'].apply(
                lambda x: b_yest_dict[x] if x in b_yest_dict else self.min_budget)
            # Total campaign max budget for the rest of a day
            df_oneday['B_max_h'] = df_oneday['campaign_id'].apply(
                lambda x: b_max_tospend_dict[x] if x in b_max_tospend_dict else self.min_budget)
            # Total campaign max budget for a whole day:
            df_oneday['B_max_n'] = df_oneday['B_min'] + df_oneday['B_max_h']
        else:
            b_max_today_dict = {cid: sum(tmp.cost.values) / self.ccf_prj_expected for cid, tmp in
                                self.df_today.groupby('campaign_id')}
            df_oneday['B_max_n'] = df_oneday['campaign_id'].apply(lambda x: b_max_today_dict[x])
            df_oneday['B_max_h'] = df_oneday['B_max_n']
            df_oneday['B_yest_h'] = 0

        from IPython import embed; embed()
        
        # Overspend: special treatment
        if self.Budget_spent_today > self.Budget_total_max:
            # we should stop spending today and distribute campaign budget within [0,Cost_today]
            df_oneday['B_max_n'] = df_oneday[['B_max_n', 'B_min']].min(axis=1)
            df_oneday['B_min'] = 0

        if self.monthly_transition:
            df_oneday['B_min'] = 0

        # Renormalize (scale up) max budget to have it equal to daily max if the latter is > sum(B_max_n)
        # or we underdeliver the daily budget
        B_max_tot = sum(df_oneday['B_max_n'])
        if self.Budget_total_max > B_max_tot or self.Budget_ratio_prj < const.BUDGET_RATIO_CRITICAL:
            df_oneday['B_max_n'] *= self.Budget_total_max / B_max_tot
        # Check for min bubget
        df_oneday['B_max_n'] = df_oneday['B_max_n'].apply(lambda x: x if x > 0.01 else 0.01)

        # This part is for Static optimization:
        # Set default budget is proportional to Score/cost, but not more than hist. max value.
        if self.Budget_spent_today < self.Budget_total_max and self.Budget_ratio_prj > const.BUDGET_RATIO_CRITICAL:
            df_oneday['Budget'] = df_oneday[['B_max','B_max_n']].min(axis=1)
            df_oneday['Budget'] = df_oneday[['B_min','Budget']].max(axis=1)
        else:
            df_oneday['Budget'] = df_oneday['B_max_n']

        # Check for min bubget for paused campaigns
        if n_toenable > 0:
            min_frac = max(0.05, 1 - self.Budget_ratio_prj)
            mean_budget = max(self.min_budget, min_frac * self.Budget_total_max / n_toenable)
            df_oneday['Budget'] = df_oneday[['Budget','campaign_state']].apply(lambda row: mean_budget if 'paused' in row['campaign_state'] else row['Budget'], axis=1)

        #Overall normalization
        df_oneday['Budget'] *= self.Budget_total_max / sum(df_oneday['Budget'])

        if const.PRINT_DETAILS > 0:
            print('get_max_budget: self.Budget_spent_today, self.Budget_total_max, sum(B_max_n), sum(B_max)',
                self.Budget_spent_today, self.Budget_total_max, B_max_tot, sum(df_oneday['B_max']))
            cols = ['campaign_id','campaign_state','Score','cost','Score_to_cost','B_min','B_max_h','B_max_n','B_max','Budget']
            print(df_oneday[cols].sort_values('Score_to_cost',ascending=False).query('B_min>-1'))

        return df_oneday

    def make_data(self, df_in):
        # Defines main variables used in the optimization (with some regularization)
        EPS = 1.e-2
        if len(df_in) > 0:
            df_in['impressions'] = df_in.apply(lambda row: row['impressions'] if row['impressions'] > 0 else EPS,
                                               axis=1)
            df_in['views'] = df_in.apply(lambda row: row['views'] if row['views'] >= 1 else EPS * row['impressions'],
                                         axis=1)
            df_in['cost'] = df_in.apply(lambda row: row['cost'] if row['cost'] > 0 else const.EPS, axis=1)
            df_in['CPV'] = np.array([1. * c / v if c > 0.0099 and v > 0.99 else 1.001 * self.CPV_max for c, v in
                                     zip(df_in['cost'].values, df_in['views'].values)])
            df_in['VR'] = df_in['views'] / df_in['impressions']
            if self.allow_sec_kpi:
                df_in['completed_views'] = df_in.apply(
                    lambda row: row['completed_views'] if row['completed_views'] >= 1 else EPS * row['impressions'], axis=1)
                df_in['clicks'] = df_in.apply(lambda row: row['clicks'] if row['clicks']>=1 else const.EPS, axis=1)
                df_in['earned_subscribers'] = df_in.apply(lambda row: row['earned_subscribers'] if row['earned_subscribers']>0 else 0*const.EPS, axis=1)
                df_in['VCR'] = df_in['completed_views'] / df_in['impressions']
                df_in['CTR'] = df_in['clicks'] / df_in['impressions']
                df_in['SR'] = 1. * df_in['earned_subscribers'] / df_in['views']

        return df_in

    def get_nhour_groups(self):
        nhour_groups = 3
        if self.days_passed <= 3:
            nhour_groups = 5
        elif self.days_passed > 3 and self.days_passed <= 6:
            nhour_groups = 4
        return nhour_groups

    def get_target_vr(self, df_in):
        VR = df_in.VR.values
        Views = df_in.views.values
        Impressions = df_in.impressions.values
        Costs = df_in.cost.values
        VR_mean = 1. * np.sum(Views) / np.sum(Impressions)
        VR_cwt = np.sum(Costs * VR) / np.sum(Costs)
        #self.VR_mean_to_cwt = VR_mean / VR_cwt
        # new VR target value should be always cost-weighted
        VR_increase_factor = const.VR_INCREASE_FACTOR
        if len(set(df_in.campaign_id)) <= 5:
            VR_increase_factor = 1.
        else:
            if self.VR_target < 0 and self.VR_choice == 'Daily Non-decreasing':
                pass
            elif self.VR_target > 0:
                VR_increase_factor = max(const.VR_INCREASE_FACTOR,
                                         1. + min(0.01, (self.VR_target / VR_mean - 1) / (self.days_left + 0.1)))

        self.VR_target = VR_increase_factor * VR_cwt

        # get VR importance:
        if self.ViewRate_imp_form < 0:
            self.ViewRate_imp = utils.get_vr_importance(self.CPV_project / self.CPV_max, 0.01, 2.)
        else:
            self.ViewRate_imp = self.ViewRate_imp_form


    def get_cpvmax_limit(self,df_in):
        CPV = df_in.CPV.values
        Costs = df_in.cost.values
        Views = df_in.views.values
        CPV_cwt = np.sum(Costs * CPV) / np.sum(Costs)
        CPV_project = sum(Costs) / (sum(Views)+const.EPS)
        if sum(Costs) < 0.02:
            CPV_project = self.CPV_project
        #CPV_mean_to_cwt = CPV_project/CPV_cwt
        CPV_decrease_factor = const.CPV_DECREASE_FACTOR
        if CPV_project > self.CPV_max_cap and self.CPV_max > 0.01:
            CPV_decrease_factor = self.CPV_max_cap / CPV_project

        self.CPV_max_limit = CPV_decrease_factor * CPV_cwt

        if const.PRINT_DETAILS > 0:
            print 'self.CPV_max_limit', self.CPV_max_limit, self.CPV_max
            print 'initial CPV_cwt', CPV_cwt
            print 'CPV_project, self.CPV_max_cap, CPV_DECREASE_FACTOR', CPV_project, self.CPV_max_cap, CPV_decrease_factor


    def get_scores(self, df_in, gamma=1.0):
        #Let's get a larger relative Scores for low-stat. campaigns:
        conf_fact = 1. + min(const.N_SIGMA_MAX, max(0,const.N_SIGMA_MAX*(1.-float(self.days_passed)/const.N_EVALUATION_DAYS)))

        CPV = df_in.CPV.values
        CPV_factor = (CPV/self.CPV_max)**(2*gamma-self.ViewRate_imp)

        key_var_prob = self.key_var+'_ul'

        #Rare events like Subs, Conversions are treated separately with some regularization applied
        if self.allow_sec_kpi and (self.second_KPI == 'SR' or self.second_KPI == 'CR' or self.second_KPI == 'EVR' or self.second_KPI == 'CTR'):
            tot_views = sum(df_in.views)
            df_in[key_var_prob] = utils.ucb(df_in[self.key_var], tot_views, df_in.views, conf_fact)
            df_in['Score'] = 1.e4*df_in[key_var_prob]/CPV_factor
        elif self.second_KPI == 'VR' or self.second_KPI == 'VCR':
            tot_impressions = sum(df_in.impressions)
            df_in[key_var_prob] = utils.ucb(df_in[self.key_var], tot_impressions, df_in.impressions, conf_fact)
            KPI_prob = df_in[key_var_prob] / df_in.impressions
            KPI_prob[KPI_prob >= 1] = 0.999
            df_in['Score'] = df_in['views'] * KPI_prob**self.ViewRate_imp/CPV_factor

        #print df_in[['campaign_id','campaign','cost','impressions','views','CPV','Score']+[self.key_var, key_var_prob]].round(3).sort_values('cost',ascending=False)

        # define score if it is nan:
        dfs = df_in.query('views >= 1')
        min_score, max_cost = 1.e-3, 1.e6
        if len(dfs) > 0:
            min_score = np.nanmin(df_in.query('views >= 1').Score.values)
            max_cost  = np.nanmax(df_in.query('views >= 1').cost.values)
        df_in['Score'] = df_in['Score'].apply(lambda x: x if ~np.isnan(x) else 0.3 * min_score)
        df_in['Score_to_cost'] = np.array([1. *s / c if (c > 1.e-3 and v >= 1) else min_score / max_cost
                                for s, c, v in zip(df_in['Score'].values,df_in['cost'].values,df_in['views'].values)])

        return df_in

    def get_cut_values(self, df_in):
        VR_min, Score_min, CPV_max = -const.EPS, const.EPS, const.K_CPV_MAX * self.CPV_max
        VR_min_outl, Score_min_outl, CPV_max_outl = VR_min, Score_min, CPV_max
        if len(df_in) >= 5:
            VR = np.array(df_in[self.second_KPI].values, dtype=np.float64)
            Impr = np.array(df_in.impressions.values, dtype=np.float64)
            if self.allow_sec_kpi and (self.second_KPI == 'CTR' or self.second_KPI == 'SR'):
                key_var_selection = self.key_var + '>0'
                df_in_tmp = df_in.query(key_var_selection)
                if len(df_in_tmp) == 0:
                    return VR_min, Score_min, CPV_max
                df_in = df_in_tmp
            Cost = np.array(df_in.cost.values, dtype=np.float64)
            Score = df_in.Score.values
            Score_to_cost = Score / Cost
            VR_min_outl = utils.outlier_cut(VR, VR_min, 0.999)[0]
            Score_min_outl = utils.outlier_cut(Score_to_cost, Score_min, 1.e6)[0]
            if self.CPV_project > const.K_CPV_MARGIN * self.CPV_max:
                CPV = df_in.CPV.values
                CPV_max = max(const.K_CPV_MAX * self.CPV_max, utils.outlier_cut(CPV, 0.001, 2 * self.CPV_max)[1])

        VR_min, Score_min = VR_min_outl, Score_min_outl
        if self.VR_min > 0:
            VR_min = self.VR_min
        if self.CPV_max_form > 0:
            CPV_max = self.CPV_max_form

        return VR_min, Score_min, CPV_max

    def aggregate_by_hours(self, df_in, nhg, last_hour):
        # Aggregates data by hour groups
        df_in['day_hour_group'] = np.array([str(d) + '_' + utils.hour_group(d, h, nhg, self.today, last_hour) for d, h in
                                            zip(df_in.day.values, df_in.hour_of_day.values)])
        groupby = list(set(['day', 'D', 'day_hour_group'] + self.grouping_metrics))
        dfs = df_in.groupby(groupby).sum()[MAIN_METRICS]
        dfs.reset_index(inplace=True)
        dfs = self.make_data(dfs)
        return dfs

    # todo model, break this up into multiple methods, maybe its own class
    def calculate_scores(self, df_in, makeFit=True):
        # Calculates and fits daily campaign Scores, and project CPV and VR(VCR) vs Budget for each campaign
        # This info is needed for
        # (a) dynamic budget allocation, and
        # (b) estimating upper limits on the daily budgets.

        def resum_to_day(df_resum, groupby=['day', 'campaign_state']):
            # Groups main campaign metrics to a day level
            groupby = list(set(groupby + self.grouping_metrics))
            df_resum = df_resum[groupby + MAIN_METRICS].groupby(groupby).sum()
            df_resum.reset_index(inplace=True)
            df_resum = self.make_data(df_resum)
            return df_resum

        def add_data(df_day, df_all, day, today=False):
            if not today:
                df_all = resum_to_day(df_all)
            if len(set(df_sel.campaign_id.values)) > len(df_day.campaign_id.values):
                for cid, row in df_all.groupby(['campaign_id']):
                    # cid,state = Id
                    if cid not in df_day.campaign_id.values:
                        tmp = row.iloc[-1].to_frame().T
                        tmp['day'] = day
                        if today:
                            tmp['cost'] = 0.0
                        df_day = df_day.append(tmp)
            return df_day

        def metrics_per_day(tmp_cmp_in):
            tmp_cmp_sum = tmp_cmp_in[MAIN_METRICS + ['day']].groupby('day').sum()
            tmp_cmp = tmp_cmp_sum.reset_index().sort_values('day', ascending=False)[MAIN_METRICS]
            if len(tmp_cmp) > 0 and self.Gamma < 0.99:
                tmp_cmp.iloc[0] /= self.ccf_prj_expected
            gamma = self.Gamma ** np.arange(len(tmp_cmp))
            m_gamma = np.matrix(gamma)
            sum_gamma = sum(gamma)
            if self.Gamma > 0.99:
                sum_gamma = 1.
            m_aver = m_gamma.dot(tmp_cmp)/sum_gamma
            return pd.DataFrame(m_aver, columns=MAIN_METRICS)

        def get_average_data(dfp):
            df_all_p = pd.DataFrame()
            grouping_metrics = list(set(self.grouping_metrics) - set(['campaign_state']))
            df_all_p = dfp.groupby(grouping_metrics).apply(metrics_per_day).reset_index()
            df_all_p['campaign_state'] = df_all_p['campaign_id'].apply(lambda cid: self.camp_status_dict[cid]
                                         if cid in self.camp_status_dict else 'paused')
            df_all_p['CPV'] = df_all_p['cost'] / df_all_p['views']
            df_all_p['VR'] = df_all_p['views'] / df_all_p['impressions']
            if self.bumper_ad:
                df_all_p['CPV'] = 1000. * df_all_p['cost'] / df_all_p['impressions']

            if self.allow_sec_kpi:
                df_all_p['VCR'] = df_all_p['completed_views'] / df_all_p['impressions']
                df_all_p['CTR'] = df_all_p['clicks'] / df_all_p['impressions']
                df_all_p['SR'] = df_all_p['earned_subscribers'] / df_all_p['views']
                df_all_p['EVR'] = df_all_p['earned_views'] / df_all_p['impressions']
                df_all_p['CR'] = df_all_p['conversions'] / df_all_p['views']
            df_all_p['day'] = self.today

            return df_all_p

        min_days = max(2, 0.1 * (self.days_passed + self.days_left))
        self.pause_allowed = self.days_passed >= min_days and self.days_left > 1 and \
                             self.Budget_spent_fraction > 0.10  and self.Budget_ratio_prj > const.BUDGET_RATIO_CRITICAL

        df_sel = df_in.copy()

        # Filter out paused campaigns:
        video_id_current = list(set(df_sel.query('campaign_state == "enabled"').video_id))

        #if (not self.enable_paused or not self.pause_allowed):  # and not self.monthly_transition:
        #    df_sel = df_sel.query('campaign_state == "enabled"')

        if not self.optimize_by_factors:
            nhg = self.get_nhour_groups()
            last_hour = 0
            if self.opt_type == 1:
                df_sel = df_sel.query('day=="%s" or hour_of_day>=%d' % (self.today, self.last_hour))
                nhg -= 1
                last_hour = self.last_hour
            df_sel = self.aggregate_by_hours(df_sel, nhg, last_hour)
        else:
            groupby = ['day', 'D'] + self.grouping_metrics
            dfs = df_sel.groupby(groupby).sum()[MAIN_METRICS]
            dfs.reset_index(inplace=True)
            df_sel = self.make_data(dfs)

        self.df_today = df_sel.query('day=="%s"' % self.today)
        self.df_today = resum_to_day(self.df_today, ['day'])

        cost_yesterday = {}
        if self.days_passed > 0 or self.monthly_transition:
            df_yesterday = df_sel.query('day=="%s"' % self.yesterday)
            cost_yesterday = df_yesterday[['campaign_id', 'cost']].groupby('campaign_id').sum()['cost'].to_dict()

        df_sel_window = df_sel.query('D>=%d' % (self.D_today - const.DAYS_WINDOW))

        #Get averaged data for scores
        df_oneday = get_average_data(df_sel_window)
        # Extend data frames with missing campaigns
        df_oneday = add_data(df_oneday, df_sel, self.today, False)

        # Get target (min) VR
        self.get_target_vr(df_oneday)
        # Get (max) CPV limit
        self.get_cpvmax_limit(df_oneday)

        # Calculate scores:
        self.df_oneday = self.get_scores(df_oneday)
        df_sel = self.get_scores(df_sel)

        df_oneday_pass = self.df_oneday.copy()
        # Suggest pausing campaigns
        if self.pause_allowed and self.Budget_ratio_prj > 0.95:  # and not self.monthly_transition:
            # Apply cuts on VR, Score and CPV
            VR_min, Score_min, CPV_max = self.get_cut_values(df_oneday.query('campaign_state=="enabled"'))
            q_kpi = self.second_KPI + '>=' + str(0.995 * VR_min)
            q = 'views>=%f and CPV<%f and Score_to_cost>%f and ' % (const.MIN_VIEWS, CPV_max, Score_min) + q_kpi
            q_all = 'campaign_state == "enabled" and (' + q + ') or campaign_state == "paused"'
            df_oneday_tmp = df_oneday.query(q_all)
            if len(df_oneday_tmp)>1:
                df_oneday_pass = df_oneday_tmp
                # df_oneday_pass = df_oneday_pass.append(df_p)

        camps_all = set(self.df_oneday.campaign_id.values)
        camps_sel = set(df_oneday_pass.campaign_id.values)
        camps_fail = list(set(camps_all) - set(camps_sel))
        df_oneday_fail = df_oneday[df_oneday['campaign_id'].isin(camps_fail)]

        # enable paused:
        df_oneday_toenable = pd.DataFrame()
        if self.enable_paused:  # and self.pause_allowed or self.monthly_transition:
            camps_paused_allopt = set(df_oneday_pass.query('campaign_state == "paused"').campaign_id)
            camps_paused_failed = set(df_oneday_fail.query('campaign_state == "paused"').campaign_id)
            camps_paused_toenable = list(camps_paused_allopt - camps_paused_failed)
            df_oneday_toenable = self.df_oneday[self.df_oneday['campaign_id'].isin(camps_paused_toenable)]
            camps_pass = list(camps_sel - camps_paused_failed)
            df_oneday_pass = df_oneday_pass[df_oneday_pass['campaign_id'].isin(camps_pass)]
            # they also should have valid video_id
            if 'video_id' in df_oneday_toenable.columns:
                df_oneday_toenable = df_oneday_toenable[df_oneday_toenable['video_id'].isin(video_id_current)]
            if 'video_id' in df_oneday_pass.columns:
                df_oneday_pass = df_oneday_pass[df_oneday_pass['video_id'].isin(video_id_current)]

        # We suggest to pause only campaigns which are currently enabled:
        df_oneday_fail = df_oneday_fail.query('campaign_state == "enabled"')
        df_oneday_fail['campaign_state'] = 'enabled (to pause)'
        # If a campaign is paused and passed the selections, its status changes to "paused (to enable)"
        df_oneday_pass['campaign_state'] = df_oneday_pass['campaign_state'].apply(
            lambda x: 'paused (to enable)' if x == 'paused' else x)
        # if (x == 'paused' and not self.monthly_transition) else x)
        # ... and it will be shown in a separate table
        df_oneday_toenable['campaign_state'] = 'paused (to enable)'

        # Combine pass and fail sets (we show failed campaigns in a separate table, but do not remove them)
        df_oneday = df_oneday_pass  # pd.concat([df_oneday_pass, df_oneday_fail])

        if len(df_oneday) < 1:
            return df_oneday, df_oneday_fail, df_oneday_toenable

        # Data frame for today should contain all the campaigns from  df_oneday set:
        self.df_today = add_data(self.df_today, df_oneday, self.today, True)

        self.df_today['B_min'] = self.df_today['cost']
        self.df_today['campaign_id'] = self.df_today['campaign_id'].astype(np.int64)
        df_oneday = pd.merge(df_oneday, self.df_today[['campaign_id', 'B_min']], on='campaign_id')
        df_oneday['Budget_max'] = df_oneday['campaign_id'].apply(
            lambda cid: self.b_max_dict[cid] if cid in self.b_max_dict else -1)

        # Dictionary with a number of days in the campaigns active today
        camp_len = {cid: len(tmp) for cid, tmp in df_sel.groupby('campaign_id') if (self.today in tmp.day.values)}
        # Switch to Static optimization if a fraction of new campaigns (#days < MIN_NDAY_PARTS) is > than some critical value
        frac_new = 1. * sum(np.array(camp_len.values()) < const.MIN_NDAY_PARTS) / (len(camp_len)+1)
        # or expected average #views is small or we spend>50% of today's budget
        CPV_project = float(self.df_sum['cpv_so_far'])
        mean_views_tosee_today = (self.Budget_total_max - sum(self.df_today.cost)) / len(df_oneday) / CPV_project
        if frac_new > const.CAMP_FRAC_MIN or self.Budget_spent_fraction > 0.9 or mean_views_tosee_today < 15 or len(df_oneday)>400 or\
                      self.second_KPI == 'CR' or self.second_KPI == 'EVR' or self.second_KPI == 'SR' or self.monthly_transition:
            self.Dynamic_scores = False
            #print ' ##### Dynamic_scores is False',frac_new, self.Budget_spent_fraction, mean_views_tosee_today, len(df_oneday),\
            #     self.second_KPI,self.monthly_transition
        #print df_oneday[['campaign_id','views','cost','CPV','VR','earned_subscribers','Score','Score_to_cost','B_min']].sort_values('Score_to_cost',ascending=False).head(20)

        # Fit scores vs budget for a dynamic optimization
        if self.Dynamic_scores:
            self.coeffs_tot = fit_scores(df_oneday, df_sel, self.ViewRate_imp, const.MIN_NDAY_PARTS)
        #else:
        #    df_oneday['Score'] = df_oneday['Score_to_cost']
        # Calculate Score_tot and correct B_max untill estimated opt. CPV is below project max CPV:
        df_oneday['B_max'] = df_oneday['Score_to_cost'] * self.Budget_total_max / sum(df_oneday['Score_to_cost'].values)

        df_oneday = self.get_max_budget(df_oneday, len(df_oneday_toenable))

        return df_oneday, df_oneday_fail, df_oneday_toenable

    def report_results(self, df_camp_grp, d_sum):
        # Prepares main optimization results and some other metrics for reporting tables
        if const.PRINT_DETAILS > 0:
            print 'report_results:'
            #df_camp_grp['CPV'] *= 100
            #df_camp_grp['CPV_max'] *= 100
            #df_camp_grp['Budget'] *= 100
            df_camp_grp['impressions'] = [int(v) for v in df_camp_grp['impressions'].values]
            df_camp_grp['views'] = [int(v) for v in df_camp_grp['views'].values]
            df_camp_grp['CPA'] = [cost/(conv+0.001) if type(conv)==float else '' for cost, conv in zip(df_camp_grp['cost'], df_camp_grp['conversions'])]
            df_camp_grp['conversions'] = [int(v) for v in df_camp_grp['conversions'].values]
            df_camp_grp['AG'] = df_camp_grp['campaign'].apply(lambda x: '-'.join(x.split('-')[6:]) if len(x.split('-'))>=8 else '-'.join(x.split('-')[5:]))
            cols = ['campaign_id','campaign','impressions','views','conversions','conversions_ul','cost','CPV','CPA','VR','Score','Score_to_cost','B_max_h','Budget','CPV_max']
            print df_camp_grp[cols].sort_values('Budget',ascending=False).round(3)
            #df_grp = df_camp_grp.groupby('AG').sum()[['cost','impressions','views','conversions']].reset_index()
            #print 'Groupped by Audience Groups:'
            #print df_grp

        if self.Dynamic_scores:
            df_camp_grp['Score_d0'] = df_camp_grp.apply(
                lambda row: self.coeffs_tot[row['campaign_id']][1] * fits.func1(row['cost'],
                                                                                *self.coeffs_tot[row['campaign_id']])
                if row['campaign_id'] in self.coeffs_tot else const.EPS, axis=1)
            df_camp_grp['EffViews_new'] = df_camp_grp[
                                              'Score'] + 0.1  # np.array([round(i,1) for i in Score_d1]) #df_camp_grp['Score_d']
            df_camp_grp['EffViews_old'] = df_camp_grp['Score_d0'] + 0.1  # np.array([round(i,1) for i in Score_d0])
        else:
            df_camp_grp['EffViews_new'] = df_camp_grp['Budget'] * df_camp_grp['Score'] + 0.1
            df_camp_grp['EffViews_old'] = df_camp_grp['cost'] * df_camp_grp['Score'] + 0.1


        # Round floats
        min_budget = self.min_budget #2 * self.CPV_project
        df_camp_grp['CPV_max'] = df_camp_grp.apply(lambda row: 0.9*self.CPV_max if row['Budget'] < min_budget else row['CPV_max'], axis=1)
        df_camp_grp['Budget'] = np.array([i if i > min_budget else min_budget for i in df_camp_grp['Budget'].values])
        if not const.ALLOW_HIGH_LI_BMAX:
            df_camp_grp['Budget'] *= self.Budget_total_daily / sum(df_camp_grp['Budget'])
        df_camp_grp['Budget'] = df_camp_grp['Budget'].round(2)
        df_camp_grp['cost'] = np.array([round(i, 2) for i in df_camp_grp['cost'].values])
        df_camp_grp['impressions'] = np.array([round(i, 1) for i in df_camp_grp['impressions'].values])
        df_camp_grp['views'] = np.array([round(i, 1) for i in df_camp_grp['views'].values])
        df_camp_grp['CPV'] = np.array([round(i, 4) for i in df_camp_grp['CPV']])
        df_camp_grp['VR'] = np.array([round(i, 3) for i in df_camp_grp['VR'].values])
        if self.allow_sec_kpi:
            df_camp_grp['clicks'] = np.array([round(i, 1) for i in df_camp_grp['clicks'].values])
            df_camp_grp['earned_subscribers'] = np.array(
                [round(i, 2) for i in df_camp_grp['earned_subscribers'].values])
            df_camp_grp['completed_views'] = np.array([round(i, 1) for i in df_camp_grp['completed_views']])
            df_camp_grp['VCR'] = np.array([round(i, 3) for i in df_camp_grp['VCR'].values])
            df_camp_grp['CTR'] = np.array([round(i, 4) for i in df_camp_grp['CTR']])
            df_camp_grp['SR'] = np.array([round(i, 4) for i in df_camp_grp['SR']])
        if 'Score_to_B_opt' in df_camp_grp.columns:
            df_camp_grp['Score'] = df_camp_grp['Score_opt']
            df_camp_grp['Score_to_cost'] = df_camp_grp['Score_to_B_opt']
        df_camp_grp['Score'] = np.array([round(i, 2) for i in df_camp_grp['Score']])
        df_camp_grp['Score_to_cost'] = np.array([round(i, 3) for i in df_camp_grp['Score_to_cost']])
        #Normalize scores to be between 0-100%:
        #Score_max = max(df_camp_grp.query('views>=1').Score)+1.e-6
        #df_camp_grp['Score'] = 100.*df_camp_grp['Score']/Score_max

        # Look for ordered campaigns that are missing in Adwords
        if len(self.project_camps_miss_active) > 0:
            # CPV_max = np.nanmax(df_camp_grp.CPV_max.values)
            self.project_camps_miss_active['Budget'] = self.min_budget  # float(self.df_sum['CPV'].values)
            self.project_camps_miss_active['CPV_max'] = 0.9 * self.CPV_max
            self.project_camps_miss_active['Score'] = const.EPS
            self.project_camps_miss_active['campaign_state'] = 'enabled (not visible)'
            if self.pause_allowed:
                self.df_camp_fail = self.df_camp_fail.append(self.project_camps_miss_active)
            else:
                df_camp_grp = df_camp_grp.append(self.project_camps_miss_active).fillna(0)

        # Pause also campaigns that have Budget=0.01 and <=0.5% percentile(Score_to_cost)
        if self.pause_allowed:
            scores = df_camp_grp.Score_to_cost.values
            Score_to_cost_05p = np.percentile(scores[~pd.isnull(scores)],[1.0])[0] + 1.e-6
            df_pause = df_camp_grp.query('campaign_state == "enabled" and ((Score_to_cost<=%f and Budget<%f) or Budget<=%f)'
                                        %(Score_to_cost_05p, self.min_budget, self.CPV_project))

            if len(df_pause) > 0:
                df_pause['campaign_state'] = 'enabled (to pause)'
                #print '%d campaigns will be paused.'%len(df_pause), set(df_pause.campaign_id)
                self.df_camp_fail = self.df_camp_fail.append(df_pause)
                camps_pass = list(set(df_camp_grp.campaign_id.values) - set(df_pause.campaign_id.values))
                df_camp_grp = df_camp_grp[df_camp_grp['campaign_id'].isin(camps_pass)]


        #Round up to integer Budget and CPV values for JPY & KRW
        if self.currency in ['JPY','KRW']:
            df_camp_grp['Budget'] = np.array([int(round(i+0.25,0)) if i>1 else 1 for i in df_camp_grp['Budget'].values])
            df_camp_grp['CPV_max'] = np.array([int(round(i+0.25,0)) for i in df_camp_grp['CPV_max']])

        #Show only (most optimal) campaigns with an assigned budget
        if self.currency in ['JPY','KRW']:
            df_rep = df_camp_grp.query('Budget>1. and cost>1. and views>=2')
        else:
            df_rep = df_camp_grp.query('Budget>0.01 and cost>0.01 and views>=2 and CPV>0.001')

        Budget_opt = np.sum(df_camp_grp['Budget'])

        # Calculate summary metrics
        df_rep['cost'] = df_rep['cost'].apply(lambda x: x if x > 0 else const.EPS)
        Views_opt = np.sum(df_rep['Budget'] / df_rep['CPV'])
        Impr_opt = np.sum((df_rep['Budget'] / df_rep['CPV']) / df_rep['VR'])
        if self.bumper_ad:
            Impr_opt  = 1.e3 * np.sum(df_rep['Budget']/df_rep['CPV'])
            Views_opt = 1.e3 * np.sum( (df_rep['Budget']/df_rep['CPV'])*df_rep['VR'])

        VR_opt = Views_opt / Impr_opt
        CPV_opt = np.sum(df_rep['Budget']) / Views_opt
        if self.bumper_ad:
            CPV_opt = 1.e3 * np.sum(df_rep['Budget']) / Impr_opt

        if self.allow_sec_kpi:
            Views_compl_opt = np.sum((df_rep['Budget'] / df_rep['CPV']) * (df_rep['VCR'] / df_rep['VR']))
            VCR_opt = Views_compl_opt / Impr_opt
            CTR_opt = round(1. * np.sum(df_rep['clicks']) / np.sum(df_rep['impressions']), 4)
            SR_opt = np.sum(df_rep['Budget'] * df_rep['SR']) / np.sum(df_rep['Budget'])

        Max_budget_increase = utils.budget_corr_fact_max(self.days_passed, self.days_left,
                                                         self.ViewRate_imp)
        EffViews_ratio = round(sum(df_rep['EffViews_new']) / sum(df_rep['EffViews_old']), 2)
        EffViews_ratio_str = str(EffViews_ratio) + ' (' + str(int(sum(df_rep['EffViews_new']))) + '/' + str(
            int(sum(df_rep['EffViews_old']))) + ')'
        Cost_ratio = round(np.sum(df_rep['Budget']) / np.sum(df_rep['cost']), 2)
        Cost_ratio_str = str(Cost_ratio) + ' (' + str(round(sum(df_rep['Budget']), 2)) + '/' + str(
            round(sum(df_rep['cost']), 2)) + ')'

        VR_opt = str(round(VR_opt, 3)) + ' +- ' + str(round(2 * np.sqrt(VR_opt * (1 - VR_opt) / Impr_opt), 3))
        if self.allow_sec_kpi:
            VCR_opt = str(round(VCR_opt, 3)) + ' +- ' + str(round(2 * np.sqrt(VCR_opt * (1 - VCR_opt) / Impr_opt), 3))
            CTR_opt = str(round(CTR_opt, 4)) + ' +- ' + str(round(2 * np.sqrt(CTR_opt * (1 - CTR_opt) / Impr_opt), 4))

        if self.currency in ['JPY','KRW']:
            Budget_opt_report = [round(Budget_opt, 0)]
            CPV_opt = str(int(round(CPV_opt+0.25,0)))+' +- '+str(round(2*CPV_opt/np.sqrt(Views_opt),1))
        else:
            Budget_opt_report = [round(Budget_opt, 2)]
            CPV_opt = str(round(CPV_opt,3))+' +- '+str(round(2*CPV_opt/np.sqrt(Views_opt),3))

        df_opt_res = pd.DataFrame({'Budget': Budget_opt_report, 'Views': [int(Views_opt)], 'ViewRate': [VR_opt],
                                   'ViewCompletionRate': [''], 'CTR': [''], 'CPV': [CPV_opt], 'SR': [''],
                                   'Cost_ratio': [Cost_ratio_str], 'EffViews_ratio': [EffViews_ratio_str],
                                   'MaxBudgetIncrease': [round(Max_budget_increase, 2)],
                                   'VR_importance': [round(self.ViewRate_imp, 2)],
                                   })

        show_metrics = ['campaign_id', 'campaign', 'campaign_state', 'impressions', 'views', 'cost', 'Budget_max',
                        'CPV', 'VR',
                        'Score_to_cost', 'Score', 'EffViews_new', 'Budget', 'B_min', 'CPV_bid', 'CPV_rule', 'CPV_max',
                        'B_max_h', 'B_max']

        if self.allow_sec_kpi:
            show_metrics += ['completed_views', 'clicks', 'earned_subscribers', 'earned_views','conversions','VCR','CTR', 'SR','EVR','CR']
            df_opt_res['ViewCompletionRate'] = VCR_opt
            df_opt_res['CTR'] = CTR_opt
            df_opt_res['CPV'] = CPV_opt
            df_opt_res['SR'] = round(SR_opt, 4)

        if self.optimize_by_factors:
            show_metrics += self.opt_factors
            show_metrics = list(set(show_metrics))

        if self.monthly_transition:
            df_camp_grp['CPV_bid'] = 0
            df_camp_grp['CPV_max'] = df_camp_grp['CPV_max'].apply(lambda x: round(x, 2) + 0.01)
            df_camp_grp['Budget'] = df_camp_grp['Budget'].apply(lambda x: round(x, 2))

        df_show = df_camp_grp.sort_values(by=['Score'], ascending=False)[show_metrics]
        df_show.index = range(1, len(df_show) + 1)

        df_show = self.sum_stat(df_show)

        #cols_show = ['campaign_id','campaign_state','impressions','views','cost','CPV','VR','Score','Score_to_cost','B_min','B_max_h','B_max','Budget','CPV_max']
        #print 'df_show'
        #print df_show[cols_show].sort_values('Score_to_cost',ascending=False)

        for col in MAIN_OPT_FACTORS:
            if col not in df_show.columns:
                df_show[col] = ''

        df_show['impressions'] = ''
        if self.DBM:
            df_show['CPV_bid'] = ''
        if not self.CPV_opt:
            df_show['CPV_bid'] = ''
            df_show['CPV_max'] = ''
            df_show['CPV_rule'] = ''
        if self.second_KPI != 'VCR' and not self.bumper_ad:
            df_show['completed_views'] = ''
            df_show['VCR'] = ''
        if self.second_KPI != 'CTR' and self.second_KPI != 'SR':
            df_show['clicks'] = ''
            df_show['CTR'] = ''
        if self.second_KPI != 'SR':
            df_show['earned_subscribers'] = ''
            df_show['SR'] = ''
        if self.second_KPI != 'EVR':
            df_show['earned_views'] = ''
            df_show['EVR'] = ''
        if self.bumper_ad:
            df_show['CPM'] = df_show['CPV']
            df_show['VCR'] = df_show['VR']
            df_show['CPM_max'] = df_show['CPV_max']
            df_show['views'] = ''
            df_show['VR'] = ''
            df_show['CPV'] = ''
            df_show['CPV_max'] = ''
            df_opt_res['CPM'] = df_opt_res['CPV']
            df_opt_res['CPV'] = ''
            df_opt_res['ViewCompletionRate'] = df_opt_res['ViewRate']
            df_opt_res['ViewRate'] = ''
            df_opt_res['SR'] = ''
            df_opt_res['Views'] = ''

        # Create links to CSV files at S3
        ext = ''
        if self.monthly_transition:
            today = parser.parse(self.today)
            next_month_day = today + relativedelta(months=+1)
            ext = '_for_' + next_month_day.strftime("%B")

        csv_dir = "static/Account_%s_Strike_%s/Results/csv" % (self.Account_id, self.Strike_id)
        if not os.path.isdir(csv_dir):
            os.system("mkdir -p %s" % csv_dir)

        # show camps to pause
        if len(self.df_camp_fail) > 0:
            self.df_camp_fail.index = range(len(self.df_camp_fail))

        # show camps to enable
        if len(self.df_camp_toenable) > 0:
            self.df_camp_toenable.index = range(len(self.df_camp_toenable))

        return df_show, df_opt_res


    def sum_stat(self, df_in):
        # Some summary statistics
        df_main = df_in.copy()  # copy.deepcopy(df_in.query('Budget>0'))
        tot_cost = sum(df_main['cost'].values)
        tot_views = sum(df_main['views'].values)
        tot_impr = sum(df_main['impressions'].values)
        tot_budget = sum(df_main['Budget_max'].values)
        tot_cpv = tot_cost/tot_views
        if self.bumper_ad:
            tot_cpv = 1000. * tot_cost/tot_impr

        if self.currency in ['JPY','KRW']:
            Budget_tot = int(round(sum(df_main['Budget'].values), 0))
        else:
            Budget_tot =  round(sum(df_main['Budget'].values), 2)

        df_sum = pd.DataFrame({'campaign_id': [''], 'campaign': ['Summary metrics'],
                               'campaign_state': [''],
                               'impressions': [int(tot_impr)],
                               'views': [int(tot_views)],
                               'completed_views': [''],
                               'clicks': [''],
                               'earned_subscribers': [''],
                               'cost': [tot_cost],
                               'CPV': [round(tot_cost / tot_views, 3)],
                               'VR': [round(1. * tot_views / tot_impr, 3)],
                               'VCR': [''],
                               'CTR': [''],
                               'SR': [''],
                               'Score': [''],  # [round(sum(df_main['Score'].values),1)],
                               'Budget': [Budget_tot],
                               'B_min': [round(sum(df_main['B_min'].values), 2)],
                               'CPV_bid': [''],
                               'CPV_max': [''],
                               'CPV_rule': [''],
                               'EffViews_new': [''],
                               'Budget_max': [tot_budget],
                               'Score_to_cost': ['']
                               })

        if self.allow_sec_kpi:
            tot_full_views = sum(df_main['completed_views'].values)
            tot_clicks = sum(df_main['clicks'].values)
            tot_subscriptions = sum(df_main['earned_subscribers'].values)
            df_sum['completed_views'] = int(tot_full_views)
            df_sum['clicks'] = int(tot_clicks)
            df_sum['conversions'] = int(sum(df_main['conversions'].values))
            df_sum['earned_subscribers'] = round(tot_subscriptions, 1)
            df_sum['VCR'] = round(1. * tot_full_views / tot_impr, 3)
            df_sum['CTR'] = round(1. * tot_clicks / tot_impr, 4)
            df_sum['VCR'] = round(1. * tot_subscriptions / tot_views, 4)

        # remove duplicate columns if any
        df_main.drop_duplicates(inplace=True)

        factor_names_to_exclude = list(set(df_main.columns) - set(df_sum.columns))

        for f in factor_names_to_exclude:
            df_sum[f] = ''

        df_main['CPV'] = np.array([round(i, 3) for i in df_main['CPV']])
        df_main['CPV_bid'] = np.array([round(i, 2) for i in df_main['CPV_bid']])
        df_main['CPV_max'] = np.array([round(i, 3) for i in df_main['CPV_max']])

        df_sum = pd.concat([df_main, df_sum])
        df_sum = df_sum[df_main.columns]

        return df_sum

    def make_factor_tables(self, df_camp_grp):
        # Prepares factor tables with optimization results aggregated by opt. factors
        def make_df(df_fact):
            df_fact['B_frac'] = np.array([round(r, 3) for r in df_fact['Budget'] / sum(df_fact['Budget'])])
            df_fact['CPV'] = np.array(
                [round(c / v, 3) if c > 0 else -1 for c, v in zip(df_fact['cost'].values, df_fact['views'].values)])
            df_fact['VR'] = np.array([round(1. * v / i, 3) if i > 1 else -1 for v, i in
                                      zip(df_fact['views'].values, df_fact['impressions'].values)])
            # if self.allow_sec_kpi:
            df_fact['SR'] = np.array([round(1. * s / v, 4) if i > 1 else -1 for v, s in
                                      zip(df_fact['views'].values, df_fact['earned_subscribers'].values)])
            return df_fact

        if not self.allow_sec_kpi:
            df_camp_grp['earned_subscribers'] = 0
        df_fact_all = df_camp_grp.groupby(self.opt_factors).sum()[
            ['cost', 'impressions', 'views', 'earned_subscribers', 'Budget']]
        df_fact_all.reset_index(inplace=True)
        df_fact_all = make_df(df_fact_all)

        # Make a table with expected final budgets
        # optimized budget per day
        B_pd = df_fact_all.set_index(self.opt_factors)['Budget'].to_dict()

        Budget_to_expect = {}
        cpv_max_to_expect = {}
        nhg = self.get_nhour_groups()
        df_sel = self.aggregate_by_hours(self.df_camp, nhg, 0)
        for f, tmp in df_sel.groupby(self.opt_factors):
            Budget_spent_by_now = np.sum(tmp['cost'])
            Budget_spent_today = np.sum(tmp.query('day=="%s"' % self.today)['cost'])
            Budget_spent_by_today = Budget_spent_by_now - Budget_spent_today

            # get uncertainties on expected budget
            tmp_sum = tmp[['cost', 'views', 'day_hour_group']].groupby(['day_hour_group']).sum()
            tmp_sum.reset_index(inplace=True)
            tmp_sum['CPV'] = np.array(
                [1. * c / v if v > 0 else 1.e3 for c, v in zip(tmp_sum['cost'].values, tmp_sum['views'].values)])
            if self.bumper_ad:
                tmp_sum['CPV'] *= 1.e3

            tmp_sum = tmp_sum.query('CPV<1.e3').sort_values('cost')
            if f in B_pd:
                Budget_to_expect[f] = B_pd[f] * self.days_left + Budget_spent_by_today

                tmp_sum['Err'] = tmp_sum.CPV / np.sqrt(tmp_sum.views)
                CPV_min = 0.01 if self.CPV_max>0.01 else 0.5 * self.CPV_max
                coeffs, cov = curve_fit(fits.func1, tmp_sum.cost, tmp_sum.CPV,
                                        bounds=([CPV_min, 0.02], [2 * self.CPV_max, 0.25]),
                                        sigma=tmp_sum.Err, absolute_sigma=True, method='dogbox', ftol=1.e-2, xtol=1.e-2)
                corr_pos = min(1.02, utils.make_df_cpv(tmp_sum, coeffs, cov, 1, 'CPV'))
                cost_max = float(tmp_sum.iloc[-1].cost)
                cpv_max_all = float(tmp_sum.iloc[-1].CPV) * (B_pd[f] / cost_max) ** 0.2
                cpv_min = min(tmp_sum.CPV.values)
                cpv_max_to_expect[f] = max(cpv_min, get_fair_CPV(coeffs, corr_pos, B_pd[f], cpv_max_all, 2))
            else:
                Budget_to_expect[f] = Budget_spent_by_now
                cpv_max_to_expect[f] = max(tmp_sum.CPV.values)

        f_norm = self.media_order / sum(Budget_to_expect.values())
        Budget_to_expect = {f: b * f_norm for f, b in Budget_to_expect.iteritems()}

        df_exp = pd.DataFrame()
        columns = self.opt_factors + ['Budget', 'CPV_max']
        for f, row in df_fact_all.groupby(self.opt_factors):
            tmp = row.iloc[-1].to_frame().T
            tmp['Budget'] = round(Budget_to_expect[f], 2)
            tmp['CPV_max'] = round(cpv_max_to_expect[f], 3)
            df_exp = df_exp.append(tmp[columns])
        df_exp['B_frac'] = np.array([round(r, 2) for r in df_exp['Budget'] / sum(df_exp['Budget'])])

        for f in MAIN_OPT_FACTORS:
            if f not in df_fact_all.columns:
                df_fact_all[f] = ''
                df_exp[f] = ''

        return (df_fact_all.query('B_frac>0.001'), df_exp)

    def get_vr_target(self):
        return self.VR_target

    def get_cpv_bid_time(self):
        t = str(self.last_cpv_bid_day) + ',' + str(self.last_cpv_bid_hour) + 'h'
        return t

    def get_budget_link(self):
        return self.budget_link

    def get_pause_link(self):
        return self.pause_link

    def get_enable_link(self):
        return self.enable_link

    def get_summary_df(self):
        return self.df_sum

    def _get_order_calculator(self):
        """Determine which type of order we should be optimizing for"""
        if self.VO_optimization:
            return view_order.ViewOrder()
        return budget_order.BudgetOrder()

    def _get_limiting_factor(self):
        """Get the order value which is our limiting factor for optimization"""
        if self.VO_optimization:
            return self.Views_total_daily
        return self.Budget_total_daily

    def _get_order_delivered(self):
        """Get the amount of whatever order we're doing delivered so far."""
        if self.VO_optimization:
            return self.views_delivered_total
        return self.media_spend_total

    def _get_order_today(self):
        if self.VO_optimization:
            return self.views_seen_today
        return self.media_spend_today

    def _get_order_total(self):
        if self.VO_optimization:
            return self.view_order
        return self.media_order
