import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ibkr_bridge import (
    _minimum_profitable_sell_limit,
    _quantity_for_signal,
    _submit_action_limit_order,
    _submit_limit_order,
)
from live_strategy import LiveDryRunStrategy


class FractionalExitTests(unittest.TestCase):
    def test_full_position_reaches_order_and_fill_accounting(self):
        for mode in ("fixed", "cash_reserve"):
            for position in (22.6507, 0.6507, 22.0):
                with self.subTest(mode=mode, position=position):
                    strategy = LiveDryRunStrategy(
                        symbol="GOOGL", initial_position=position,
                        initial_avg_cost=100, order_quantity=4,
                        sizing_mode=mode, fixed_buy_fee=1,
                        fixed_sell_fee=1.5, min_sell_profit=5,
                        profit_target_mode="fixed",
                    )
                    signal = SimpleNamespace(
                        signal="WOULD SELL", blocked=False, symbol="GOOGL",
                        close_bid=120, close_ask=120.01,
                    )
                    quantity = _quantity_for_signal(signal, strategy, 4, mode)
                    self.assertEqual(quantity, position)
                    self.assertEqual(strategy.active_quantity_for_cost(100), position)
                    floor = _minimum_profitable_sell_limit(
                        strategy, quantity, min_profit=5, profit_target_mode="fixed",
                    )
                    self.assertEqual(floor, strategy._minimum_profitable_sell_price())
                    self.assertGreaterEqual((floor - 100) * position - 2.5, 5 - 1e-10)
                    ib = Mock()
                    _submit_limit_order(ib, object(), signal, quantity, 1)
                    self.assertEqual(ib.placeOrder.call_args.args[1].totalQuantity, position)
                    _submit_action_limit_order(ib, object(), "GOOGL", "SELL", quantity, 120)
                    self.assertEqual(ib.placeOrder.call_args.args[1].totalQuantity, position)
                    strategy.apply_filled_order("SELL", quantity, 120)
                    self.assertEqual(strategy.initial_position, 0)

    def test_cash_has_no_sell_and_buy_stays_whole(self):
        strategy = LiveDryRunStrategy(
            symbol="GOOGL", initial_position=0, sizing_mode="cash_reserve",
            capital_budget=1000, cash_reserve=100, fixed_buy_fee=1,
        )
        self.assertEqual(strategy.active_sell_quantity(), 0)
        self.assertEqual(strategy.active_buy_quantity(100), 8)

    def test_per_share_target_uses_fractional_quantity(self):
        strategy = LiveDryRunStrategy(symbol="GOOGL", min_sell_profit_per_share=1)
        self.assertEqual(strategy._target_profit(0.6507), 0.6507)


if __name__ == "__main__":
    unittest.main()
