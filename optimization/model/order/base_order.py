"""
base_order.py

The base abstraction around what defines a customer order type
(e.g. budget, view, etc.)
"""


class BaseOrder:
    """
    Base Order

    Abstract interface for all types of customer orders. Defines the functions
    needed in order to calculate values dependent on the type of customer order
    and convert orders into an estimated budget
    """
    cost_function_sample_days = 3  # How many days we rely on historical instead of project data

    def get_expected_order(self, campaign_df, data_cost_vs_hour, vertical, prev_hour, days_passed):
        """Get the expected order for the run project"""

    def get_total_daily_budget(self, limitation_param):
        """Calculate the total daily budget for the given order"""

    def get_budget_ratio(self, days_passed, days_left, order_today, order_delivered, order_total):
        """Get the budget ratio for the order associated with the given project"""

    def get_cost_function_for_vertical(self, cost_function_by_vertical, vertical, days_passed):
        """
        Get Cost Function for Vertical

        Given a specified vertical and a number of days a campaign has been running, determine
        which cumulative cost function we want to use for the campaign and vertical for budget
        optimization
        """
        if days_passed < 3 and vertical in cost_function_by_vertical:
            return cost_function_by_vertical[vertical]
        return cost_function_by_vertical['All']
