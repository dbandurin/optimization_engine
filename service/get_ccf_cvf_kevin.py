import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import extras
import pickle
pd.options.display.width=200
import config

def get_data(q, db_cfg):
	try:
	   conn = psycopg2.connect(
	       "dbname=%s user=%s host=%s password=%s port=%s"%\
	       (db_cfg['database'], db_cfg['user'], db_cfg['host'], db_cfg['password'], db_cfg['port'])
	   )
	except:
	  print("\nI am unable to connect to database %s, host %s\n"%(DB_NAME, DB_HOST))
	  return pd.DataFrame()

	dict_cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
	dict_cur.execute(q)
	recs = dict_cur.fetchall()
	df = pd.DataFrame()
	if len(recs)>0:
	    df = pd.DataFrame(recs, columns=recs[0].keys())

	conn.close()

	return df

def get_metrics(start_date, end_date, all=False):
   db_config = config.DATABASE['metrics']

   if all:
   	   query =  "select hour as h, sum(media_cost) as cost, sum(views) as views from metrics \
        		where report_date >='%s' and report_date <= '%s'  \
        		group by hour "%(start_date, end_date)
   else:
   		query =  "select hour as h, advertiser_id, sum(media_cost) as cost, sum(views) as views from metrics \
        		 where report_date >='%s' and report_date <= '%s'  \
        		 group by hour, advertiser_id "%(start_date, end_date)

   df = get_data(query, db_config)

   return df

def get_kevin_adv_verticals():
	db_config = config.DATABASE['kevin']

	query = "select advertiser.advertiser_id, advertiser.account_id, \
	account.brand_vertical_id, account.name as account_name, brand_vertical.name as vertical\
	from advertiser \
	inner join account on advertiser.account_id = account.id \
	inner join brand_vertical on account.brand_vertical_id = brand_vertical.id"

	df = get_data(query, db_config)

	return df

def fill_gaps(df_in):
    #fill gaps in hourly data
    df_new = pd.DataFrame({'h':range(24),'ccf':0,'cvf':0})
    for ix,row in df_new.iterrows():
        if row['h'] not in df_in.h.values:
            df_in = df_in.append(row)
    df_in = df_in.sort_values('h')        
    return df_in

def get_ccf(df_in, grp_factor):
	d_ccf = {}
	for grp_id, tmp in df_in.groupby(grp_factor):
		tot_cost = np.nansum(tmp.cost)
		if tot_cost < 10.:
			continue
		if len(tmp) < 24:
			tmp = fill_gaps(tmp)
		tmp = tmp.sort_values('h')
		tmp = tmp.fillna(0)
		tmp['cum_cost'] = np.cumsum(tmp.cost)
		tmp['ccf'] = (tmp.cum_cost/(tot_cost + 0.001)).round(3)
		d = tmp.set_index('h')['ccf'].to_dict()
		d_ccf[grp_id] = d

	return d_ccf


def get_cvf(df_in, grp_factor):
	d_cvf = {}
	for grp_id, tmp in df_in.groupby(grp_factor):
		tot_views = np.nansum(tmp.views)
		if tot_views < 100:
			continue
		if len(tmp) < 24:
			tmp = fill_gaps(tmp)
		tmp = tmp.sort_values('h')
		tmp = tmp.fillna(0)
		tmp['cum_views'] = np.cumsum(tmp.views)
		tmp['cvf'] = (tmp.cum_views/(tot_views + 0.001)).round(3)
		d = tmp.set_index('h')['cvf'].to_dict()
		d_cvf[grp_id] = d

	return d_cvf

#Get Kevin data on advertisers and vertical
print('Extracting Kevin flight data ...')
df_kev = get_kevin_adv_verticals()
d_adv_vert = df_kev.set_index('advertiser_id')['vertical'].to_dict()
d_adv_acc  = df_kev.set_index('advertiser_id')['account_id'].to_dict()

#Get Metrics data for last 2 years
print('Extracting metrics advertisers data ...')
start_date = '2017-04-01'
end_date   = '2019-05-31'
dfm = get_metrics(start_date, end_date)
dfm['vertical'] = dfm['advertiser_id'].apply(lambda x: d_adv_vert.get(x,'NA'))
dfm['account_id']  = dfm['advertiser_id'].apply(lambda x: d_adv_acc.get(x,'NA'))
dfm = dfm.query('vertical != "NA" and account_id != "NA"')

# CCF curves per Advertiser
print('Calculating CCF and CVF curves ...')
d_ccf_account = get_ccf(dfm, 'account_id')
d_cvf_account = get_cvf(dfm, 'account_id')

# CCF curves per Brand Vertical
d_ccf_vertical = get_ccf(dfm, 'vertical')
d_cvf_vertical = get_cvf(dfm, 'vertical')

#CCF averaged curve for all verticals
dfm_all = get_metrics(start_date, end_date, all=True)
dfm_all['All_verticals'] = 'All'

d_ccf_vertical['All'] = get_ccf(dfm_all, 'All_verticals')['All']
d_cvf_vertical['All'] = get_cvf(dfm_all, 'All_verticals')['All']

MyDicts = {'d_ccf_account':d_ccf_account, 'd_ccf_vertical':d_ccf_vertical,
		   'd_cvf_account':d_cvf_account, 'd_cvf_vertical':d_cvf_vertical}

#Store results
file_output = 'data_ccf_cvf_vs_hour.p'
print('Stiring results as 4 dictionaries to %s'%file_output)
with open(file_output, 'wb') as fp:
	pickle.dump(MyDicts, fp)

#myDicts = pickle.load( open ("data_ccf_vs_hour.p", "rb") )

