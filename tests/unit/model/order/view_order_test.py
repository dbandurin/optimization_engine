"""
Test View Order

Unit tests for the view order logic
"""

from mock import patch
import pandas as pd

from optimization.model.order import view_order


def test_get_expected_order_from_historical_data_for_vertical():
    """Make sure we return the historical estimate of the hour for projects with limited data"""
    cost_function = {'vertical': {'prev_hour': [42]}, 'd_cvf_account': {1: {'prev_hour': 10}}}
    order = view_order.ViewOrder()
    actual = order.get_expected_order(None, cost_function, 1, 'vertical', 'prev_hour', 1)
    assert actual == 10


def test_get_expected_order_from_historical_data_from_all():
    """Make sure we return the historical estimate of the hour for projects with limited data"""
    cost_function = {'notVertical': {'prev_hour': [42]}, 'd_cvf_account': {1: {'prev_hour': 10}}, 'd_cvf_vertical': {'All': {'prev_hour': 420}}}
    order = view_order.ViewOrder()
    actual = order.get_expected_order(None, cost_function, 2, 'vertical', 'prev_hour', 1)
    assert actual == 420


def test_get_expected_order_long_running_project():
    """Make sure we determine the expected budget from the flights own data"""
    cost_function = {'All': {'prev_hour': [42]}}
    campaign_df = pd.DataFrame.from_dict({
        'hour_of_day': [5, 6],
        'views': [1, 1]
    })
    order = view_order.ViewOrder()
    actual = order.get_expected_order(campaign_df, cost_function, 1, 'vertical', 5, 3)
    assert round(actual, 2) == 0.51


@patch('optimization.model.CPV_bids.get_nsigma', lambda _: 10)
@patch('optimization.model.utils.hour_group', lambda d, h, a, b, c: h)
def test_get_total_daily_budget():
    mock_campaign_df = pd.DataFrame.from_dict({
        'D': ['1997-08-04', '1997-08-04', '1997-08-06'],
        'hour_of_day': [1, 1, 3],
        'impressions': [0, 0, 0],
        'views': [2, 2, 1],
        'cost': [4, 6, 1]
    })  # cpv_std == 0.75
    order = view_order.ViewOrder()
    actual = order.get_total_daily_budget(None, mock_campaign_df, 1.0, 10.0)
    assert actual == 85.0


def test_get_budget_ratio():
    """Make sure we return the expected formula output"""
    order = view_order.ViewOrder()
    actual = order.get_budget_ratio(2, 18, 10.0, 50.0, 300.2)
    assert round(actual, 2) == 1.22
