from backtest_lab.costs import TaiwanCostModel


def test_broker_fee_uses_minimum_fee() -> None:
    model = TaiwanCostModel()
    assert model.broker_fee(1000) == 20


def test_broker_fee_rounds_standard_rate() -> None:
    model = TaiwanCostModel()
    assert model.broker_fee(1_000_000) == 1425


def test_stock_sell_cost_includes_stock_tax() -> None:
    model = TaiwanCostModel()
    assert model.sell_cost(1_000_000, "stock") == 4425


def test_etf_sell_cost_includes_etf_tax() -> None:
    model = TaiwanCostModel()
    assert model.sell_cost(1_000_000, "etf") == 2425

