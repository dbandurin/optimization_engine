import errno
import json
import os
import web
import cPickle as pickle

import numpy as np

from optimization import engine as optimization_engine
from ui import config
import redis


d_opt_type = {'Greedy': 0, 'Optimal for Today': 1}


class Index:

    def __init__(self):
        self.redis_client = redis.StrictRedis(
            host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB
        )
        strikes = sorted(set(self.redis_client.keys('*')))

        self.optimization_form = web.form.Form(
            web.form.Dropdown('strike_id', strikes, description='Project Strike ID', value=strikes[0]),
            web.form.Textbox('total_max_budget', web.form.notnull, description='Max total budget', value='-1'),
            web.form.Textbox('view_rate_importance', description='VR (VCR, CTR) importance', value='-1'),
            web.form.Textbox('view_rate_min', web.form.notnull, description='Min VR (VCR, CTR)', value='-1'),
            web.form.Textbox('max_cpv', web.form.notnull, description='Max CPV', value='-1'),
            web.form.Dropdown('second_kpi', ['VR', 'VCR', 'CTR'], description='Type of user interaction', value='VR'),
            web.form.Dropdown('Opt_type', d_opt_type.keys(),description='Optimization Type',value='Optimal for Today'),
            web.form.Checkbox('Enable_campaigns', checked=False, value=True, description='Add paused campaigns'),
            web.form.Checkbox('Monthly_transition', checked=False, value=True, description='Optimization for EOM transition'),
            web.form.Button('Submit', class_="btn btn-primary submit"),
            validators=[web.form.Validator("Wrong Strike ID format!", lambda i: (len(i.strike_id) >= 8))]
        )

        self.save_plan_form = web.form.Form(
            web.form.Button('CPV bids applied', id="save_plan_to_db", class_="btn btn-primary save-to-DB submit"),
            web.form.Hidden('strike_id'),
            web.form.Hidden('day'),
            web.form.Hidden('hour')
        )

    # GET method is used when there is no form data sent to the server
    def GET(self):
        form = self.optimization_form()
        return web.template.render('ui/templates').form(content=form)

    def POST(self):
        form = self.optimization_form()
        if not form.validates():
            return web.template.render('ui/templates').form(content=form)

        # form values
        budget_factors = 'None'
        day_of_data = 'Today and Yesterday'
        max_cpv = eval(form.d.max_cpv)
        second_kpi = form.d.second_kpi
        strike_id = form.d.strike_id
        total_max_budget = eval(form.d.total_max_budget)
        view_rate_choice = 'No Target VR'
        view_rate_importance = eval(form.d.view_rate_importance)
        view_rate_min = eval(form.d.view_rate_min)
        opt_type = form.d.Opt_type
        enable_paused = form.d.Enable_campaigns
        monthly_transition = form.d.Monthly_transition

        data_string = self.redis_client.get(strike_id)
        if data_string == 'error processing data':
            return None

        data_dict = pickle.loads(data_string)
        strike_id = strike_id.split(':')[1]

        ccf_data = data_dict['ccf_data']
        optimization_params = data_dict['optimization_params']

        optimization_params['budget_by_factors'] = budget_factors
        optimization_params['day_of_data'] = day_of_data
        optimization_params['cpv_max_form'] = max_cpv
        optimization_params['second_kpi'] = second_kpi
        optimization_params['view_rate_choice'] = view_rate_choice
        optimization_params['view_rate_imp'] = view_rate_importance
        optimization_params['view_rate_min'] = view_rate_min
        optimization_params['opt_type'] = opt_type
        optimization_params['enable_paused'] = enable_paused
        optimization_params['monthly_transition'] = monthly_transition
        if total_max_budget > 0:
            optimization_params['total_max_project_budget'] = total_max_budget

        adwords_metrics = data_dict['adwords_metrics']
        project_metrics = data_dict['project_metrics']
        project_camps_miss_active = data_dict['project_camps_miss_active']
        summary_metrics = data_dict['summary_metrics']

        optimization_results = optimization_engine.run(strike_id, optimization_params, ccf_data, adwords_metrics,
                                                       project_metrics, summary_metrics, project_camps_miss_active)

        df_res = optimization_results['df_res']
        df_fact = optimization_results['df_fact']
        df_pause = optimization_results['df_pause']
        df_enable = optimization_results['df_enable']
        df_opt_res = optimization_results['df_opt_res']
        df_sum = optimization_results['df_sum']

        self.save_plan_form.d.strike_id = strike_id
        self.save_plan_form.d.day = ''
        self.save_plan_form.d.hour = ''
        df_age, df_dev, df_gen, df_video = df_fact

        df_pause['N'] = range(1, len(df_pause) + 1)
        if len(df_pause) > 0:
            df_pause = df_pause.dropna(subset=['Score', 'impressions', 'views'])
            df_pause = self._make_pretty(df_pause)
        df_enable['N'] = range(1, len(df_enable) + 1)
        if len(df_enable) > 0:
            df_enable = df_enable.dropna(subset=['Score', 'impressions', 'views'])
            df_enable = self._make_pretty(df_enable)

        form_put_plan = web.form.Form(
            web.form.Button('CPV bids applied', id="save_plan_to_db", class_="btn btn-primary save-to-DB submit"),
            web.form.Hidden('Strike_id', value=strike_id),
            web.form.Hidden('day', value=None),
            web.form.Hidden('hour', value=str(None))
        )

        plots = None
        control_plots = None
        budget_link = optimization_results['budget_link']
        optimization_type = optimization_results['opt_type']
        account_id = 'FakeAccountIdForNow'
        strike_id = optimization_results['strike_id']
        make_control_plots = False
        cpv_bid_time = optimization_results['cpv_bid_time']
        cpv_opt = True,
        vr_target = optimization_results['vr_target']

        location = 'static/Account_' + account_id + '_Strike_' + strike_id + '/Results/json'
        self._store_as_json(df_opt_res, location, 'opt_res.json')
        self._store_as_json(df_res, location, 'budget_opt.json')
        self._store_as_json(df_sum, location, 'summary.json')

        if len(df_res) == 1:
            return df_res['Status'].values

        return web.template.render('ui/templates').form2(account_id, strike_id, make_control_plots, plots,
                                                         control_plots, budget_link, df_age, df_dev,
                                                         df_gen, df_video, df_pause, df_enable, optimization_type,
                                                         day_of_data, cpv_bid_time, cpv_opt, round(vr_target, 3),
                                                         content=form, content2=form_put_plan, flag=1)

    @staticmethod
    def _store_as_json(df_tmp, file_location, file_name):
        file_location = '{0}/{1}'.format(file_location, file_name)
        if not os.path.exists(os.path.dirname(file_location)):
            try:
                os.makedirs(os.path.dirname(file_location))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        d = df_tmp.T.to_dict()
        data = [d[i] for i in d.keys()]
        d = {"data": data}
        json.dump(d, open(file_location, 'w'))
        return d

    @staticmethod
    def _make_pretty(df_in):
        df_in['cost'] = np.array([round(i, 2) if str(i) != 'nan' else 0 for i in df_in['cost'].values])
        df_in['campaign_id'] = np.array([int(i) for i in df_in['campaign_id'].values])
        df_in['clicks'] = np.array([round(i, 2) if str(i) != 'nan' else 0 for i in df_in['clicks'].values])
        df_in['views'] = np.array([round(i, 1) if str(i) != 'nan' else 0 for i in df_in['views'].values])
        df_in['impressions'] = np.array([round(i, 1) if str(i) != 'nan' else 0 for i in df_in['impressions'].values])
        df_in['CPV'] = np.array([round(i, 3) if str(i) != 'nan' else -1 for i in df_in['CPV']])
        df_in['VR'] = np.array(
            [round(i, 3) if str(i) != 'nan' else -1 for i in df_in['VR'].values])  # if i>0.01 else 0
        df_in['VCR'] = np.array(
            [round(i, 3) if str(i) != 'nan' else -1 for i in df_in['VCR'].values])  # if i>0.01 else 0
        df_in['CTR'] = np.array([round(i, 3) if str(i) != 'nan' else -1 for i in df_in['CTR']])  # if i>0.0001 else 0
        df_in['Score'] = np.array([round(i, 2) if str(i) != 'nan' else -1 for i in df_in['Score']])
        df_in['Score_to_cost'] = np.array([round(i, 3) if str(i) != 'nan' else -1 for i in df_in['Score_to_cost']])
        return df_in.sort_values(['campaign_state', 'Score'])


class Health:
    def __init__(self):
        pass

    def GET(self):
        return "healthy"


class ResearchUI(object):
    def __init__(self):
        self.urls = ('/', 'Index',
                     '/health', 'Health')

    def serve(self):
        app = web.application(self.urls, globals())
        app.wsgifunc()
        app.run()
