"""تست‌های واحد موتور زیستی/اقتصادی."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.assumptions import Assumptions    # noqa: E402
from core.biology import Biology            # noqa: E402

A = Assumptions()
B = Biology(A)


def test_growth_hits_observed_milestones():
    assert abs(B.weight_at_age(80) - 1.0) < 1e-9
    assert abs(B.weight_at_age(100) - 2.0) < 1e-9
    assert abs(B.weight_at_age(130) - 10.0) < 1e-9
    assert abs(B.weight_at_age(140) - 15.0) < 1e-9


def test_five_gram_interpolation_is_between_2g_and_10g():
    d5 = B.age_at_weight(5.0)
    assert 100 < d5 < 130
    assert abs(d5 - 117.1) < 1.5          # درون‌یابی log-linear شفاف


def test_growth_monotone_and_invertible():
    for w in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 25.0):
        assert abs(B.weight_at_age(B.age_at_weight(w)) - w) < 1e-6
    prev = -1
    for day in range(0, 200):
        w = B.weight_at_age(day)
        assert w > prev
        prev = w


def test_mortality_matches_observed_and_is_front_loaded():
    assert abs(B.cum_mortality(80) - 0.28) < 1e-12
    assert abs(B.cum_mortality(100) - 0.32) < 1e-12
    assert abs(B.cum_mortality(140) - 0.40) < 1e-12
    assert B.cum_mortality(0) == 0.0
    # front-loaded: در نیمه اول مرحله تخم→۱g بیش از نصف تلفات آن مرحله رخ دهد
    assert B.cum_mortality(40) > 0.5 * 0.28


def test_survival_ratio_composes():
    r1 = B.survival_ratio(0, 50)
    r2 = B.survival_ratio(50, 120)
    assert abs(r1 * r2 - B.survival_ratio(0, 120)) < 1e-12


def test_capacity_anchors_and_monotonicity():
    assert abs(B.fish_per_pond(1.0) - 40000) < 1
    assert abs(B.fish_per_pond(2.0) - 27500) < 1
    assert abs(B.fish_per_pond(15.0) - 13500) < 1
    c5, c10 = B.fish_per_pond(5.0), B.fish_per_pond(10.0)
    assert 13500 < c10 < c5 < 27500
    assert not B.counts_toward_pond_capacity(0.9)
    assert B.counts_toward_pond_capacity(1.0)


def test_heterogeneity_quantiles():
    q10, q50, q90 = (B.weight_quantile(5.0, q) for q in (0.1, 0.5, 0.9))
    assert q10 < q50 < q90
    # میانگین توزیع لاگ‌نرمال باید همان mean_weight باشد
    import math
    cv = B.cv_at_weight(5.0)
    mu, sig = B._lognorm_params(5.0, cv)
    assert abs(math.exp(mu + sig * sig / 2) - 5.0) < 1e-9
    # CV با وزن زیاد می‌شود
    assert B.cv_at_weight(1.0) < B.cv_at_weight(15.0)
    # سهم بالای میانگین برای لاگ‌نرمال کمی زیر ۵۰٪ است
    f = B.fraction_above(5.0, 5.0)
    assert 0.4 < f < 0.5


def test_fraction_above_is_monotone():
    prev = 1.1
    for t in (1, 2, 5, 10, 15, 20):
        f = B.fraction_above(10.0, t)
        assert f <= prev + 1e-12
        prev = f


def test_feed_price_is_piecewise_not_flat():
    assert B.feed_price(0.8) == 310000
    assert B.feed_price(4.0) == 305000
    assert B.feed_price(12.0) == 199000
    assert B.feed_price(12.0) < B.feed_price(0.8)


def test_feed_cost_per_gram_uses_bands():
    c_small = B.feed_cost_per_gram_gain(0.5, 1.5)
    c_big = B.feed_cost_per_gram_gain(10.0, 15.0)
    assert abs(c_small - 310.0) < 1e-6      # 310,000/kg → 310 تومان/گرم با FCR=1
    assert abs(c_big - 199.0) < 1e-6


def test_feed_kg_charges_dead_fish_growth():
    a = B.feed_kg_for_growth(1000, 1.0, 2.0, n_died=0)
    b = B.feed_kg_for_growth(1000, 1.0, 2.0, n_died=200)
    assert abs(a - 1.0) < 1e-9              # 1000 fish × 1 g = 1 kg با FCR=1
    assert b > a


def test_sale_price_curve():
    assert B.sale_price(1.0) == 11000
    assert B.sale_price(2.0) == 11800
    assert B.sale_price(5.0) == 14200
    assert B.sale_price(10.0) == 18200
    assert B.sale_price(15.0) == 22200


def test_oxygen_is_diagnostic_only():
    o = B.oxygen_headroom(200.0)
    assert o["max_biomass_kg"] > 0
    assert 0 <= o["load_ratio"]


def test_speed_multiplier_shifts_curve():
    A2 = Assumptions()
    A2.defs["growth.speed_multiplier"]["value"] = 1.2
    B2 = Biology(A2)
    assert B2.weight_at_age(80) > B.weight_at_age(80)
    assert B2.age_at_weight(1.0) < 80
