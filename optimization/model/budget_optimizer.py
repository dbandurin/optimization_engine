import fits
from scipy.optimize import minimize
from utils import Good_Budget, Good_CPV, Good_VR
import numpy as np
import pandas as pd
from collections import Counter
import time
from optimization import constants

const = constants.Const()

def budget_optimize(df_in, cf, Dynamic_scores, days_passed, coeffs_tot, VR_target, CPV_max_limit, monthly_transition, Budget_total_max):
    df_camp_grp = df_in.copy()
    Camps = df_camp_grp['campaign_id'].values
    VR = df_camp_grp['VR'].values
    Cost = df_camp_grp['cost'].values.copy() 
    Cost *= Budget_total_max/sum(Cost)
    N =  len(Camps)

    CPV = df_camp_grp['CPV'].values
    good_cpv = False  # Good_CPV(Cost,CPV,CPV_max_limit)
    good_vr  = False  # Good_VR(Cost,VR,VR_target)

    return_tuple = (100, good_cpv, good_vr, df_camp_grp)

    #from IPython import embed; embed()

    if not Dynamic_scores:
        if const.PRINT_DETAILS >= 1:
            print '\n===> Static optimization', sum(df_camp_grp['Budget'])

        if days_passed > 3 and const.STATIC_OPT_VERSION == 2 and sum(df_camp_grp['B_min']) > 0.01:
            #Start of direct solution
            df_camp_grp = df_camp_grp.sort_values('Score_to_cost',ascending=False)
            #assuming we are 2-3 hours behind the current data
            df_camp_grp['B_min2'] = 1.10 * df_camp_grp['B_min']
            B_max = list(df_camp_grp.B_min2.values)
            df_camp_grp.index = range(N)
            for ix, row in df_camp_grp.iterrows():
                b_max = row['B_max_h'] 
                Sum = sum(B_max) - row['B_min2']
                if Sum <= Budget_total_max:
                    if Sum + b_max <= Budget_total_max:
                        B_max[ix] += b_max
                    else:
                        #Assign remaining balance (LIs in the end of the ordered Score/cost list)
                        B_max[ix] = Budget_total_max - Sum        

            df_camp_grp['B_max2'] = np.array(B_max)
            if not const.ALLOW_HIGH_LI_BMAX:
                df_camp_grp['B_max2'] *= Budget_total_max/sum(df_camp_grp['B_max2'])
            df_camp_grp['B_max2'] = df_camp_grp['B_max2'].apply(lambda x: x if x>0.01 else 0.01)
            
            #print 'Budget_total_max-1', Budget_total_max,sum(df_camp_grp.B_max_h),sum(df_camp_grp.Budget),sum(df_camp_grp.B_max2) 
            #print df_camp_grp[['campaign_id','Score','Score_to_cost','B_min','B_min2','B_max_h','B_max','B_max_n','Budget','B_max2']].sort_values(
            #    'Score_to_cost', ascending=False).round(2)
            
            df_camp_grp['Budget'] = df_camp_grp['B_max2']
            #End of direct solution

        return (100, good_cpv, good_vr, df_camp_grp)

    #
    #The rest is a dynamic optimization
    #
    #Dynamic objective
    NormScale = 0
    for i in range(N):
        cid = Camps[i]
        NormScale += fits.func1(Cost[i], *coeffs_tot[cid])

    def ScoreOpt(costs):
        all_scores = []
        for i in range(N):
            cid = Camps[i]
            f = fits.func1(costs[i], *coeffs_tot[cid])
            all_scores.append(f)
        return -1.0 * sum(all_scores) / NormScale

    #Dynamic derivative
    def ScoreOpt_der(costs):
        all_diffs = []
        for i in range(N):
            cid = Camps[i]
            diff_scores = fits.func1(costs[i], *coeffs_tot[cid]) * coeffs_tot[cid][1] / costs[i]
            all_diffs.append(diff_scores)
        return -1.0 * np.array(all_diffs) / NormScale

    def make_fit(x0, bnds, cons):
        disp = const.PRINT_DETAILS
        if const.PRINT_DETAILS > 0:
            print '===> Dynamic optimization'
        res = minimize(ScoreOpt, x0, method='SLSQP', jac=ScoreOpt_der,  
                                   bounds=bnds, constraints=cons,
                                   options={'disp': disp,'maxiter':200,'iprint':1, #})
                                            'ftol':1.e-5}) 
        return res

    #Min budget should not be less than B_min + 0.25*B_yest_h
    #if const.ALLOW_HIGH_LI_BMAX:
    #    Budget_total_max = sum(df_camp_grp.B_max_n.values)
    B_to_deliver = max(0.01*N, Budget_total_max - sum(df_camp_grp.B_min.values))
    B_yest_h_tot = sum(df_camp_grp.B_yest_h.values)
    scale_yest = 0.25*min(B_to_deliver, B_yest_h_tot) / B_yest_h_tot 
    B_max_h_min = scale_yest * df_camp_grp.B_yest_h.values

    B_camp_max = cf*(df_camp_grp.B_max_n.values - df_camp_grp.B_min.values+0.10) - B_max_h_min # == B_max_h, NEW Style
    B_camp_min = np.zeros(N) + 0.01
    B_camp_max = np.array([j if j>i else i+0.10 for i,j in zip(B_camp_min,B_camp_max)])

    Budget_total_max_opt = B_to_deliver - sum(B_max_h_min)

    #Inequalities
    #min Budget
    cons =  ({'type': 'ineq','fun' : lambda x: np.array([sum(x) - 1.005*Budget_total_max_opt]),
                     'jac' : lambda x: np.ones(N)},
    #max Budget
            {'type': 'ineq','fun' : lambda x: np.array([-sum(x) + 1.01*Budget_total_max_opt]),
                     'jac' : lambda x: -1.*np.ones(N)},
    #min VR
            {'type': 'ineq','fun' : lambda x: np.array([sum(x*VR) - sum(x)*VR_target]),
                     'jac' : lambda x: np.array(list(VR - VR_target))},
    #max CPV 
            {'type': 'ineq','fun' : lambda x: np.array([sum(x)*CPV_max_limit*0.99 - sum(x*CPV)]),
                     'jac' : lambda x: np.array(list(CPV_max_limit*0.99 - CPV))}
           )

    #bounds for budget fractions
    bnds = zip(B_camp_min, 1.1*B_camp_max)

    #Create a few starting seeds:
    x_seed = []
    n_rand_max = 2
    for ir in range(n_rand_max):
        x_seed.append(np.random.random(N))
    x_seed = [list(Budget_total_max_opt*x/sum(x)) for x in x_seed]
    x_seed.append([Budget_total_max_opt/N]*N)

    #Find optimal solution
    res_best = make_fit(x_seed[0], bnds, cons)
    res_score_best = abs(res_best.fun)
    for x in x_seed[1:]:
        res = make_fit(x, bnds, cons)
        if res.status == 0:
            good_cpv = Good_CPV(res.x,CPV,CPV_max_limit)
            good_vr = Good_VR(res.x,VR,VR_target)
            b_tot = sum(res.x + df_camp_grp.B_min.values + B_max_h_min)
            good_b = Good_Budget(b_tot/Budget_total_max) 
            if good_cpv and good_vr and good_b and abs(res.fun) > res_score_best:
                res_score_best = abs(res.fun)
                res_best = res

    if res_best.status == 0:
        return_tuple = (res_best.status, good_cpv, good_vr, df_camp_grp)
        df_camp_grp['B_opt'] = res_best.x
        df_camp_grp['Budget'] = res_best.x + df_camp_grp.B_min.values + B_max_h_min
        df_camp_grp['Score_opt'] = df_camp_grp.apply(lambda row: fits.func1(row['B_opt'], *coeffs_tot[row['campaign_id']])
                                                             if row['campaign_id'] in coeffs_tot else 1.e-6, axis=1)
        df_camp_grp['Score_to_B_opt'] = df_camp_grp['Score_opt']/df_camp_grp['B_opt']

        if const.PRINT_DETAILS >= 1:
            cols = ['campaign_id','campaign_state','views','cost','CPV','VR','Score','Score_to_cost','Score_opt','B_opt','Score_to_B_opt',
                    'B_min','B_max_h','B_yest_h','B_max','Budget']        
            print df_camp_grp[cols].sort_values('Score_to_B_opt',ascending=False).round(3)
            print 'sum(df_camp_grp.Score_opt)',sum(df_camp_grp.Score_opt), sum(df_camp_grp.Budget)

    return return_tuple 
