"""
budget_order.py

Functions related to calculate budgets from a budget order
"""

import numpy as np

from optimization.model.order.base_order import BaseOrder


class BudgetOrder(BaseOrder):
    """
    Budget Order

    Logic to handle order calculations for customers who submitted a budget order.
    """
    def get_expected_order(self, campaign_df, data_cost_vs_hour, account_id, vertical, prev_hour, days_passed):
        """
        Get Expected Order

        Get an estimate for the cost of the campaign for the given hour based on historical
        data either from a historical estimate by vertical or by historical data of the
        campaign
        """
        if days_passed < self.cost_function_sample_days:
            if account_id in data_cost_vs_hour['d_ccf_account']:
                cost_function = data_cost_vs_hour['d_ccf_account'][account_id]
            else:
                cost_function = self.get_cost_function_for_vertical(
                    data_cost_vs_hour['d_ccf_vertical'], vertical, days_passed)
            return cost_function[prev_hour]
        else:
            cost_by_hour = campaign_df[['hour_of_day', 'cost']].groupby(['hour_of_day']).sum()
            cost_by_hour.reset_index(inplace=True)
            total_cost = sum(cost_by_hour.cost.values) + 0.001
            cost_by_hour['cum_frac_cost'] = np.cumsum(cost_by_hour.cost.values) / total_cost
            prev_cost = float(cost_by_hour.query('hour_of_day == %d' % prev_hour)['cum_frac_cost'])
            return min(1., prev_cost + 0.01)

    def get_total_daily_budget(self, budget_ratio, campaign_df, order_to_cost, order_today):
        """Calculate the total daily budget for the given order"""
        return order_today  # limiting parameter is the max budget, so we're done

    def get_budget_ratio(self, days_passed, days_left, order_today, order_delivered, order_total):
        """Get the budget ratio for the order associated with the given project"""
        total_delivered = days_passed * order_total
        budget_expected_by_now = total_delivered / (days_passed + days_left) + order_today + 1
        return order_delivered / budget_expected_by_now
