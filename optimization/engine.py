from datetime import datetime
import logging
import os

from model import optimization_model
from s3 import S3_handler

_logger = logging.getLogger(__name__)


def run(strike_id, optimization_params, ccf_data, metrics, project_information,
        summary_metrics, project_camps_miss_active):

    try:
        opt_model = optimization_model.OptimizationModel(
            ccf_data, optimization_params, metrics, project_information,
            project_camps_miss_active, summary_metrics
        )
        df_out, df_fact, df_exp_b, df_pause, df_enable, df_opt_res = opt_model.run()

        return {
            'strike_id': strike_id,
            'df_res': df_out,
            'df_fact': df_fact,
            'df_exp_b': df_exp_b,
            'df_pause': df_pause,
            'df_enable': df_enable,
            'df_opt_res': df_opt_res,
            'df_sum': opt_model.get_summary_df(),
            'cpm_optimization': 'AV' in project_information['strike_id'],
            'vr_target': opt_model.get_vr_target(),
            'cpv_bid_time': opt_model.get_cpv_bid_time(),
            'budget_link': opt_model.get_budget_link(),
            'opt_type': 'Static' if opt_model.Dynamic_scores else 'Dynamic'
        }

    except Exception as e:
        _logger.exception("An unexpected exception occurred in the model: %s", e)

        is_trueview_project = strike_id.startswith('TV')

        exception_envelope = {
            'exception': type(e),
            'exception_message': e.message,
            'input_data': {
                'strike_id': strike_id,
                'optimization_params': optimization_params,
                'ccf_data': ccf_data,
                'metrics': metrics,
                'project_information': project_information,
                'summary_metrics': summary_metrics,
                'project_camps_miss_active': project_camps_miss_active
            }
        }

        if not is_trueview_project or (is_trueview_project and os.environ.get('SAVE_UTOPSA_ERROR_FLAG', 'false').lower() == 'true'):
            key = '{}_{}.failure'.format(strike_id, datetime.utcnow().isoformat())
            S3_handler.put_object(os.environ.get('OPTIMIZATION_FAILURE_BUCKET'), key, exception_envelope)
        raise e
