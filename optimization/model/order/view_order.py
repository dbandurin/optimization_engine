"""
View Order.

Functions related to calculate budgets from a view order campaign.
"""

import numpy as np

from optimization.model import CPV_bids, utils
from optimization.model.order.base_order import BaseOrder


class ViewOrder(BaseOrder):
    """
    View Order.

    Logic to handle order calculations for customers who submitted a view order.
    """

    def get_expected_order(self, campaign_df, data_cost_vs_hour, account_id, vertical, prev_hour, days_passed):
        """
        Get Expected Order.

        Get an estimate for the number of views the campaign will get for the given hour based on
        historical data either from a historical estimate by vertical or by historical data of
        the campaign.
        """
        if days_passed < self.cost_function_sample_days:
            if account_id in data_cost_vs_hour['d_cvf_account']:
                cost_function = data_cost_vs_hour['d_cvf_account'][account_id]
            else:
                cost_function = self.get_cost_function_for_vertical(
                    data_cost_vs_hour['d_cvf_vertical'], vertical, days_passed)
            return cost_function[prev_hour]
        else:
            views_by_hour = campaign_df[['hour_of_day','views']].groupby(['hour_of_day']).sum()
            views_by_hour.reset_index(inplace=True)
            total_views = sum(views_by_hour.views.values) + 0.01
            views_by_hour['cum_view_frac'] = 1.0 * np.cumsum(views_by_hour.views.values) / total_views
            prev_views = float(views_by_hour.query('hour_of_day == %d' % prev_hour)['cum_view_frac'])
            return min(1.0, prev_views + 0.01)

    def get_total_daily_budget(self, budget_ratio, campaign_df, order_to_cost, order_today):
        n_sigma = CPV_bids.get_nsigma(budget_ratio)
        cpv_std = self._get_cpv_std(campaign_df)
        estimated_cpv = order_to_cost + n_sigma * cpv_std
        return max(1.0, estimated_cpv * order_today)

    def get_budget_ratio(self, days_passed, days_left, order_today, order_delivered, order_total):
        total_delivered = days_passed * order_total
        views_expected_by_now = total_delivered / (days_passed + days_left) + order_today + 1
        return 1.0 * order_delivered / views_expected_by_now

    def _get_cpv_std(self, campaign_df):
        campaign_df['day_hour_group'] = np.array(
            ['{}_{}'.format(d, utils.hour_group(d, h, 6, -1, 0))
             for d, h in zip(campaign_df.D.values, campaign_df.hour_of_day.values)]
        )
        df_dh_groups = campaign_df.groupby(['day_hour_group']).sum()[['impressions', 'views', 'cost']]
        df_dh_groups = df_dh_groups.reset_index(inplace=False).query('cost>0')
        df_dh_groups['CPV'] = df_dh_groups['cost'] / (df_dh_groups['views']+1)
        return max(0.001, np.std(df_dh_groups.CPV.values))
