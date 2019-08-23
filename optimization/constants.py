
class Const(object):
	#Small value
	EPS = 1.e-6
	#Size of window in #days for Score aggregation and CPV fitting
	DAYS_WINDOW = 15
	#Min #day parts to make Score fitting
	MIN_NDAY_PARTS = 5
	#Min fraction of new campaigns to switch to Static optimization
	CAMP_FRAC_MIN = 0.50
	#Default VR (VCR, CTR) increase factor in the opt. run
	VR_INCREASE_FACTOR = 0.95
	#Default CPV decrease factor in the opt. run
	CPV_DECREASE_FACTOR = 1.005
	#Number of evaluation days used in Score definition
	N_EVALUATION_DAYS = 3
	#Initial #sigmas used to increase Score during the evaluation days 
	N_SIGMA_MAX = 3.0
	#Number of hours required to pass before a new change of per-campaign CPV_max
	HOURS_CPV_MIN = 4
	#Min number of views per-campaign required to accumulate before a new change of per-campaign CPV_max
	VIEWS_CPV_MIN = 10
	#Defines a critical value of CPV_project (=K_CPV_MARGIN*CPV_max) above which we start pausing high CPV campaigns 
	K_CPV_MARGIN = 0.95
	#Defines max possible increase of CPV campaigns after which it will be paused at the critical CPV_project
	K_CPV_MAX = 1.3
	#Suppression factor used in the metrics aggregation during last DAYS_WINDOW
	GAMMA = 0.7
	#Max fraction of daily budget allowed to distribute to a campaign
	MAX_B_FRAC = 0.5  
	#No Dynamic optimization happens if delivered/expected budget ratio is below this value
	BUDGET_RATIO_CRITICAL = 0.85 
	#Min #views to keep campaign running (active)
	MIN_VIEWS = 2
	#Power factor used in the CPV fitting 
	COST_POWER_MAX = 0.25
	#Flag to switch to other secondary KPIs (VCR, CTR, VR) in addition to default VR
	ADDITIONAL_SEC_KPI = -1
    #Print datails (0 means no printing)
	PRINT_DETAILS = 0
	#Version of static budget optimization
	STATIC_OPT_VERSION = 2
	#Allow total LI budget sum to be > Budget_flight_daily
	ALLOW_HIGH_LI_BMAX = False

	def __setattr__(self, *_):
	    pass
