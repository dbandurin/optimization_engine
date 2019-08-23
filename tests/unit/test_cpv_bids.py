import pytest

from optimization.model.CPV_bids import apply_network_handicap
from optimization.model.CPV_bids import get_fair_CPV
from optimization.model.CPV_bids import get_nsigma


def test_apply_network_handicap_no_network_match():
    result = 0
    expected = 0
    network = 'foo'
    result += apply_network_handicap(network)
    assert result == expected


def test_apply_network_handicap_using_InSearchYouTube():
    result = 1
    expected = 3
    network = 'InSearchYouTube'
    result += apply_network_handicap(network)
    assert result == expected


def test_apply_network_handicap_using_InDisplayYouTube():
    result = 1
    expected = 3
    network = 'InDisplayYouTube'
    result += apply_network_handicap(network)
    assert result == expected


def test_get_fair_CPV_nsignma_happy_path():
    expected = 0.02274036908603927
    result = get_fair_CPV([0.02, 0.05], 1, 10, 0.1, 0.1)
    assert result == expected


def test_get_nsigma_happy():
    expected = 4.8972465448410398
    result = get_nsigma(0.5)
    assert result == expected


def test_get_nsigma_with_0():
    expected = 5
    result = get_nsigma(0)
    assert result == expected


def test_get_nsigma_with_0_point_7():
    expected = 2.5719725084898193
    result = get_nsigma(0.7)
    assert result == expected


def test_get_nsigma_with_1():
    expected = 0.97889892738602047
    result = get_nsigma(1)
    assert result == expected


def test_get_nsigma_with_1_point_2():
    expected = 0.51410544818889437
    result = get_nsigma(1.2)
    assert result == expected

