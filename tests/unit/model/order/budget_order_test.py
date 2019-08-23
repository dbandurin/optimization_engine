"""
Test Budget Order

Unit tests for the budget order logic
"""

from mock import patch
import pandas as pd

from optimization.model.order import budget_order


def test_get_expected_order_from_historical_data_for_vertical(monkeypatch):
    """Make sure we return the historical estimate of the hour for projects with limited data"""
    cost_function = {'vertical': {'prev_hour': [42]}, 'd_ccf_account': {1: {'prev_hour': 10}}}
    order = budget_order.BudgetOrder()
    actual = order.get_expected_order(None, cost_function, 1, 'vertical', 'prev_hour', 1)
    assert actual == 10


def test_get_expected_order_from_historical_data_from_all():
    """Make sure we return the historical estimate of the hour for projects with limited data"""
    cost_function = {'notVertical': {'prev_hour': [42]}, 'd_ccf_account': {1: {'prev_hour': 10}}, 'd_ccf_vertical': {'All': {'prev_hour': 420}}}
    order = budget_order.BudgetOrder()
    actual = order.get_expected_order(None, cost_function, 2, 'vertical', 'prev_hour', 1)
    assert actual == 420


def test_get_expected_order_long_running_project():
    """Make sure we determine the expected budget from the flights own data"""
    cost_function = {'All': {'prev_hour': [42]}}
    campaign_df = pd.DataFrame.from_dict({
        'hour_of_day': [5, 6],
        'cost': [1, 1]
    })
    order = budget_order.BudgetOrder()
    actual = order.get_expected_order(campaign_df, cost_function, 1, 'vertical', 5, 3)


def test_get_total_daily_budget():
    """Make sure we just return the limiting param"""
    order = budget_order.BudgetOrder()
    assert order.get_total_daily_budget(None, None, None, 42) == 42


def test_get_budget_ratio():
    """Make sure we return the expected formula output"""
    order = budget_order.BudgetOrder()
    actual = order.get_budget_ratio(2, 18, 10.0, 50.0, 300.2)
    assert round(actual, 2) == 1.22
