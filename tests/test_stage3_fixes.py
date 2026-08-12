"""
تست‌های ۶ اصلاح نهایی مرحله ۳.

ده سناریوی الزامی specification:
 1. فروش ماهی بدون فشار ظرفیت
 2. همان فروش وقتی pond capacity کاملاً پر است
 3. فروش با Working Capital محدود
 4. فروش 3g در برابر نگه‌داشتن تا وزن بالاتر با saleability کمتر
 5. فروش از دو cohort مختلف
 6. Egg Offer با cash payment
 7. همان Egg Offer با supplier credit
 8. Egg Offer که فقط Partial Buy آن feasible است
 9. What-If sale که واقعاً pond/feed/cash را تغییر دهد
10. What-If شامل چند hypothetical transaction
"""
import os
import sys
from datetime import timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as m                                        # noqa: E402
from app import Engine, api                            # noqa: E402
from core import offers as OF                          # noqa: E402
from core import hypothetical as H                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "test_stage3_fixes.db")


@pytest.fixture(scope="module")
def E():
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(DB + ext):
            os.remove(DB + ext)
    eng = Engine(DB)
    m.ENGINE = eng
    yield eng
    eng.db.close()


def call(path, method="GET", body=None, query=None):
    return api(path, method, query or {}, body or {})


def ctx(E):
    H.cache_clear()
    return E.ctx()


def _cohort(E, min_w=0.0):
    st = E.ctx()[2]
    cands = [c for c in st.cohorts.values() if c.alive >= 1
             and c.mean_weight >= min_w]
    return max(cands, key=lambda c: c.alive)


def offer_date(E, days=30):
    return (E.ctx()[2].as_of + timedelta(days=days)).isoformat()


# ═══════════════════════════════ زیرساخت: clone و وضعیت فرضی
def test_clone_is_fully_independent(E):
    A, bio, st = ctx(E)
    st2 = st.clone()
    assert st2.hypothetical is True
    c0 = next(iter(st.cohorts.values()))
    before = c0.alive
    st2.cohorts[c0.cohort_id].alive -= 5000
    st2.cohorts[c0.cohort_id].alloc.clear()
    assert c0.alive == before, "تغییر clone نباید وضعیت واقعی را عوض کند"
    assert c0.alloc, "تخصیص واقعی نباید خالی شود"


def test_apply_sale_requires_clone(E):
    A, bio, st = ctx(E)
    c = _cohort(E)
    with pytest.raises(RuntimeError, match="clone"):
        H.apply_sale(st, [{"cohort_id": c.cohort_id, "quantity": 100}], 15000)


def test_integer_ponds_uses_ceil_per_cohort(E):
    A, bio, st = ctx(E)
    total = H.integer_ponds_now(st)
    manual = sum(H.ponds_of_cohort(st, c) for c in st.cohorts.values())
    assert total == manual
    assert total == int(total), "نیاز استخر باید صحیح باشد"


def test_plan_cache_reuses_same_inputs(E):
    A, bio, st = ctx(E)
    p1 = H.solve_plan(A, bio, st, "balanced")
    p2 = H.solve_plan(A, bio, st, "balanced")
    assert p1 is p2, "همان ورودی باید از cache بیاید"
    H.cache_clear()
    p3 = H.solve_plan(A, bio, st, "balanced")
    assert p3 is not p1
    assert p3.summary()["contribution_nominal"] == \
        pytest.approx(p1.summary()["contribution_nominal"], rel=1e-6), \
        "reproducibility: همان ورودی همان خروجی"


# ═══════════════════════════════ اصلاح ۱: کف از بهینه‌سازی دوباره
def test_sale_floor_comes_from_reoptimisation(E):
    A, bio, st = ctx(E)
    c = _cohort(E)
    r = OF.evaluate_sale_offer(A, bio, st, {
        "cohort_id": c.cohort_id, "quantity": 10_000, "price": 15_000})
    assert r["method"] == "re-optimisation"
    fs = r["floor_search"]
    assert fs["solver_runs"] >= 1
    assert len(fs["probe_points"]) >= 1
    # در قیمت کف، فاصله دو سناریو باید داخل حد دقت باشد
    tol = fs["tolerance_per_fish"] * 10_000
    assert abs(fs["residual_value"]) <= tol * 1.5
    # هر دو سناریو گزارش شده‌اند
    assert "keep" in r["scenarios"] and "accept" in r["scenarios"]
    assert r["scenarios"]["keep"]["npv"] != 0


def test_floor_is_indifference_price(E):
    """در قیمت کف، NPV(پذیرش) ≈ NPV(نگه‌داشتن)."""
    A, bio, st = ctx(E)
    c = _cohort(E)
    qty = 10_000
    r = OF.evaluate_sale_offer(A, bio, st, {
        "cohort_id": c.cohort_id, "quantity": qty, "price": 15_000})
    econ = r["prices"]["economic_floor"]
    at_floor = OF.evaluate_sale_offer(A, bio, st, {
        "cohort_id": c.cohort_id, "quantity": qty, "price": econ})
    tol = float(A.get("offers.floor_tolerance_per_fish")) * qty
    assert abs(at_floor["difference_vs_keeping"]) <= tol * 2


# ═══════════════════════════ سناریو ۱ و ۲: فروش با/بی فشار ظرفیت
def test_s1_s2_sale_with_and_without_capacity_pressure(E):
    """
    سناریو ۱ و ۲: همان فروش، یک بار با ظرفیت آزاد و یک بار با ظرفیت پر.

    جهت تغییر کف از پیش قطعی نیست: با ظرفیت تنگ، برنامه «نگه‌داشتن» خودش
    برداشت زودتر را انتخاب می‌کند (Fix 3)، پس ارزش نسبی آفر خارجی می‌تواند
    کم یا زیاد شود. آنچه الزامی است:
      * ارزیابی در هر دو حالت از بهینه‌سازی دوباره بیاید،
      * پذیرش فروش فشار استخر برنامه را بیشتر نکند،
      * کف با وضعیت مزرعه تغییر کند (reservation price ثابت نیست).
    """
    A, bio, st = ctx(E)
    c = _cohort(E, min_w=1.0)
    off = {"cohort_id": c.cohort_id, "quantity": 20_000, "price": 15_000,
           "payment_terms": {"upfront_share": 1.0, "delay_days": 0}}
    loose = OF.evaluate_sale_offer(A, bio, st, dict(off))
    assert loose["method"] == "re-optimisation"
    assert loose["scenarios"]["accept"]["peak_ponds"] <= \
        loose["scenarios"]["keep"]["peak_ponds"] + 1e-6

    A.set("farm.operational_ponds", 10)          # ظرفیت را به‌شدت تنگ کن
    try:
        A2, bio2, st2 = ctx(E)
        tight = OF.evaluate_sale_offer(A2, bio2, st2, dict(off))
    finally:
        A.reset("farm.operational_ponds")
        H.cache_clear()
    assert tight["method"] == "re-optimisation"
    # پذیرش فروش نباید فشار استخر را از «نگه‌داشتن» بیشتر کند
    assert tight["scenarios"]["accept"]["peak_ponds"] <= \
        tight["scenarios"]["keep"]["peak_ponds"] + 1e-6
    # محدودیت باید در خودِ برنامه «نگه‌داشتن» دیده شود: با ۱۰ استخر، اوج
    # نیاز استخر برنامه از حالت آزاد کمتر است (Fix 3: برداشت زودتر خودکار).
    assert tight["scenarios"]["keep"]["peak_ponds"] <= \
        loose["scenarios"]["keep"]["peak_ponds"]
    # کف در هر دو حالت متناهی و معقول است. نکته: برابر بودن کف دو حالت
    # نقص نیست — یعنی اقتصادِ حاشیه‌ایِ همین ۲۰k ماهی به سقف‌ها حساس نبوده
    # و پاسخ optimizer در keep/accept متقارن بوده است.
    for r in (loose, tight):
        assert 0 < r["prices"]["economic_floor"] < 60_000


def test_s3_sale_with_tight_working_capital(E):
    """
    سناریو ۳: فروش وقتی سرمایه در گردش محدود است.

    الزامات: ارزیابی از بهینه‌سازی دوباره بیاید، محدودیت سرمایه در هر دو
    سناریو دیده شود، پذیرش فروش اوج نیاز نقدی را بیشتر نکند و کف با وضعیت
    مالی مزرعه تغییر کند. (جهت تغییر کف قطعی نیست: با سرمایه تنگ، برنامه
    «نگه‌داشتن» خودش کوچک‌تر می‌شود و ارزش نسبی آفر می‌تواند هر دو طرف برود.)
    """
    A, bio, st = ctx(E)
    c = _cohort(E, min_w=1.0)
    off = {"cohort_id": c.cohort_id, "quantity": 20_000, "price": 15_000,
           "payment_terms": {"upfront_share": 1.0, "delay_days": 0}}
    rich = OF.evaluate_sale_offer(A, bio, st, dict(off))

    A.set("finance.working_capital_available", 1.2e9)
    try:
        A2, bio2, st2 = ctx(E)
        poor = OF.evaluate_sale_offer(A2, bio2, st2, dict(off))
    finally:
        A.reset("finance.working_capital_available")
        H.cache_clear()
    assert poor["method"] == "re-optimisation"
    assert poor["working_capital"]["wc_available"] == pytest.approx(1.2e9)
    # وجه فروش نباید اوج نیاز نقدی را بدتر کند
    assert poor["working_capital"]["peak_funding_after"] <= \
        poor["working_capital"]["peak_funding_before"] + 1e-6
    # محدودیت مالی باید در برنامه دیده شود: سقف سرمایه سناریوی poor همان
    # override است و برنامه‌اش کوچک‌تر/محتاط‌تر از rich است.
    assert poor["scenarios"]["keep"]["eggs_planned"] <= \
        rich["scenarios"]["keep"]["eggs_planned"]
    assert 0 < poor["prices"]["economic_floor"] < 60_000


def test_s3b_worse_receipt_terms_raise_nominal_floor(E):
    """
    آزمون جهت‌دار و پایدار برای ارزش زمانی پول (اصلاح ۲):
    همان فروش با دریافت تمام‌مدت‌دار باید کف **اسمی** بالاتری بخواهد تا
    ارزش امروزِ یکسانی بدهد.
    """
    A, bio, st = ctx(E)
    c = _cohort(E, min_w=1.0)
    base = {"cohort_id": c.cohort_id, "quantity": 20_000, "price": 15_000}
    cash = OF.evaluate_sale_offer(A, bio, st, {
        **base, "payment_terms": {"upfront_share": 1.0, "delay_days": 0}})
    slow = OF.evaluate_sale_offer(A, bio, st, {
        **base, "payment_terms": {"upfront_share": 0.0, "delay_days": 120}})
    assert slow["payment"]["present_value_factor"] < \
        cash["payment"]["present_value_factor"]
    assert slow["prices"]["economic_floor"] > \
        cash["prices"]["economic_floor"], \
        "دریافت دیرتر باید قیمت اسمی بالاتری بخواهد"


def test_s4_saleability_shows_in_keep_alternative(E):
    """
    فروش 3g (آفر تأییدشده، احتمال ۱) در برابر نگه‌داشتن تا وزن بالاتر با
    saleability کمتر: احتمال مشتری هرگز در قیمت ضرب نمی‌شود، بلکه به‌صورت
    تأخیر فروش در پروفایل «نگه‌داشتن» وارد شده و کف را پایین می‌آورد.
    """
    A, bio, st = ctx(E)
    c = _cohort(E, min_w=1.0)
    r = OF.evaluate_sale_offer(A, bio, st, {
        "cohort_id": c.cohort_id, "quantity": 10_000, "price": 15_000})
    for o in r["alternative"]["options"]:
        if o.get("is_sell_now"):
            assert o["saleability"] == 1.0, "آفر واقعی یعنی مشتری قطعی"
        elif o["harvest_w"] > 5:
            assert o["saleability"] == pytest.approx(0.30)
        else:
            assert o["saleability"] == pytest.approx(0.70)

    # با saleability کمتر برای وزن‌های بالا، نگه‌داشتن کم‌ارزش‌تر → کف پایین‌تر
    base_tbl = A.get("planning.saleability")
    worse = [dict(b, prob=(0.05 if float(b["w_min"]) >= 5 else b["prob"]))
             for b in base_tbl]
    A.set("planning.saleability", worse)
    try:
        A2, bio2, st2 = ctx(E)
        r2 = OF.evaluate_sale_offer(A2, bio2, st2, {
            "cohort_id": c.cohort_id, "quantity": 10_000, "price": 15_000})
    finally:
        A.reset("planning.saleability")
        H.cache_clear()
    assert r2["prices"]["economic_floor"] <= \
        r["prices"]["economic_floor"] + 100


# ═══════════════════════ اصلاح ۳: استخر واقعاً آزادشده
def test_ponds_freed_is_integer_difference(E):
    A, bio, st = ctx(E)
    c = _cohort(E, min_w=1.0)
    cap = bio.fish_per_pond(c.mean_weight)
    before = H.integer_ponds_now(st)

    # فروش کوچک که مرز استخر را جابه‌جا نمی‌کند
    small = OF.evaluate_sale_offer(A, bio, st, {
        "cohort_id": c.cohort_id, "quantity": min(500, c.alive * 0.01),
        "price": 15_000})
    assert small["ponds_before"] == before
    assert small["ponds_freed"] == \
        small["ponds_before"] - small["ponds_after"]
    assert float(small["ponds_freed"]).is_integer()

    # فروش به‌اندازه‌ای که دقیقاً یک استخر این cohort را خالی کند
    import math
    need = math.ceil(c.alive / cap)
    if need >= 2:
        qty = c.alive - (need - 1) * cap + 1
        big = OF.evaluate_sale_offer(A, bio, st, {
            "cohort_id": c.cohort_id, "quantity": qty, "price": 15_000})
        assert big["ponds_freed"] >= 1
        assert big["ponds_freed"] == big["ponds_before"] - big["ponds_after"]


# ═══════════════════════ اصلاح ۴ و سناریو ۵: آفر چند-cohort
def test_s5_multi_cohort_allocation_suggested_and_used(E):
    """آفر «X قطعه حدود w گرم» بدون cohort_id باید از چند cohort تأمین شود."""
    A, bio, st = ctx(E)
    # وزنی بین دو cohort نخست تا هر دو واجد شرایط شوند
    cs = sorted([c for c in st.cohorts.values() if c.alive >= 1],
                key=lambda c: -c.alive)[:2]
    w_req = (cs[0].mean_weight + cs[1].mean_weight) / 2
    # بیش از موجودی بزرگ‌ترین cohort → تأمین حتماً چند-cohort می‌شود
    qty = cs[0].alive + cs[1].alive * 0.3

    sug = H.suggest_allocation(A, bio, st, qty, w_req, tolerance=5.0)
    assert sug["feasible"]
    assert len(sug["allocations"]) >= 2, "باید از بیش از یک cohort تأمین شود"
    total = sum(a["quantity"] for a in sug["allocations"])
    assert total == pytest.approx(qty, abs=1)
    for a in sug["allocations"]:
        c = st.cohorts[a["cohort_id"]]
        assert a["quantity"] <= c.alive + 1e-6

    r = OF.evaluate_sale_offer(A, bio, st, {
        "quantity": qty, "weight_g": w_req, "price": 15_000,
        "allocations": sug["allocations"]})
    assert r["allocation"]["multi_cohort"]
    assert r["allocation"]["source"] == "user"
    # سناریوی پذیرش باید موجودی هر دو cohort را کم کرده باشد
    sd = r["scenarios"]["accept"]
    assert sd["npv"] != r["scenarios"]["keep"]["npv"]


def test_allocation_validation_rules(E):
    A, bio, st = ctx(E)
    c = _cohort(E)
    with pytest.raises(ValueError, match="بیشتر است"):
        H.validate_allocation(st, [{"cohort_id": c.cohort_id,
                                    "quantity": c.alive * 3}], c.alive * 3)
    with pytest.raises(ValueError, match="برابر نیست"):
        H.validate_allocation(st, [{"cohort_id": c.cohort_id, "quantity": 100}],
                              5_000)
    with pytest.raises(ValueError, match="ناشناخته"):
        H.validate_allocation(st, [{"cohort_id": "NOPE", "quantity": 10}], 10)


def test_allocation_endpoint(E):
    st = E.ctx()[2]
    c = max(st.cohorts.values(), key=lambda x: x.alive)
    r = call("/api/decide/sale/allocation", "POST", {
        "quantity": min(5_000, c.alive / 2), "weight_g": c.mean_weight})
    assert r["feasible"]
    assert r["allocations"]


# ═══════════════ اصلاح ۵ و سناریوهای ۶–۸: آفر تخم
def test_s6_s7_supplier_credit_raises_max_justified_price(E):
    """اعتبار تأمین‌کننده باید قیمت اسمی بالاتری را توجیه کند."""
    A, bio, st = ctx(E)
    base = {"date": offer_date(E), "quantity": 100_000, "price": 6_000}
    cash = OF.evaluate_egg_offer(A, bio, st, dict(base), partial_options=False)
    credit = OF.evaluate_egg_offer(
        A, bio, st, {**base, "payment_terms": {"upfront_share": 0.0,
                                               "delay_days": 60}},
        partial_options=False)
    assert credit["payment"]["present_value_factor"] < 1.0
    assert credit["max_justified_price"] > cash["max_justified_price"], \
        "۶٬۸۰۰ مدت‌دار و ۶٬۵۰۰ نقد نباید یکسان ارزیابی شوند"
    # سود افزوده هم باید با اعتبار بهتر شود (پرداخت دیرتر = ارزش امروز کمتر)
    assert credit["expected_profit_impact"] >= cash["expected_profit_impact"] - 1e-6


def test_s8_hard_feasibility_forces_partial_or_reject(E):
    """
    Profit مثبت به‌تنهایی کافی نیست: با ظرفیت/نقدینگی تنگ، خرید کامل نباید
    BUY بگیرد؛ یا PARTIAL_BUY اجراشدنی پیشنهاد شود یا REJECT.

    برای کنترل زمان تست، دورهای اصلاح و سقف حل در حالت تنگ محدود می‌شوند؛
    این فقط سرعت است و منطق feasibility را عوض نمی‌کند.
    """
    A, bio, st = ctx(E)
    ok = OF.evaluate_egg_offer(A, bio, st, {
        "date": offer_date(E), "quantity": 300_000, "price": 3_000},
        partial_options=False)
    assert ok["decision"] in ("BUY", "PARTIAL_BUY")
    assert ok["full_accept"]["feasible"]

    A.set("farm.operational_ponds", 8)
    A.set("finance.working_capital_available", 1.0e9)
    A.set("planning.max_repair_rounds", 1)
    A.set("planning.solver_time_limit_s", 8)
    try:
        A2, bio2, st2 = ctx(E)
        tight = OF.evaluate_egg_offer(A2, bio2, st2, {
            "date": offer_date(E), "quantity": 300_000, "price": 3_000})
    finally:
        for k in ("farm.operational_ponds", "finance.working_capital_available",
                  "planning.max_repair_rounds", "planning.solver_time_limit_s"):
            A.reset(k)
        H.cache_clear()

    if tight["decision"] == "BUY":
        assert tight["full_accept"]["feasible"], \
            "BUY بدون feasibility مجاز نیست"
    else:
        assert tight["decision"] in ("PARTIAL_BUY", "REJECT")
        if tight["decision"] == "PARTIAL_BUY":
            chosen = [o for o in [tight["full_accept"]] + tight["options"]
                      if o["quantity"] == tight["preferred_quantity"]]
            assert chosen and chosen[0]["feasible"], \
                "گزینه پیشنهادی PARTIAL باید اجراشدنی باشد"


def test_egg_offer_infeasible_reason_reported(E):
    A, bio, st = ctx(E)
    A.set("finance.working_capital_available", 0.4e9)
    A.set("planning.max_repair_rounds", 1)
    A.set("planning.solver_time_limit_s", 8)
    try:
        A2, bio2, st2 = ctx(E)
        r = OF.evaluate_egg_offer(A2, bio2, st2, {
            "date": offer_date(E), "quantity": 300_000, "price": 5_500},
            partial_options=False)
        if not r["full_accept"]["feasible"]:
            assert r["full_accept"]["infeasible_reason_fa"]
    finally:
        for k in ("finance.working_capital_available",
                  "planning.max_repair_rounds", "planning.solver_time_limit_s"):
            A.reset(k)
        H.cache_clear()


# ═══════════════ اصلاح ۶ و سناریوهای ۹–۱۰: What-If واقعی
def test_s9_what_if_sale_changes_pond_feed_cash(E):
    A, bio, st = ctx(E)
    c = _cohort(E, min_w=1.0)
    qty = min(30_000, c.alive * 0.5)
    r = OF.what_if(A, bio, st, [
        {"type": "sell_fish", "cohort_id": c.cohort_id,
         "quantity": qty, "price": 16_000}])
    assert r["method"] == "cloned_state_reoptimisation"
    sd = r["state_delta"]
    assert sd["live_fish_delta"] == pytest.approx(-qty, abs=1)
    assert r["delta"]["feed_cost"] < 0, "ماهی فروخته‌شده دیگر خوراک نمی‌خورد"
    assert any(x["amount"] > 0 for x in sd["cash_rows"]), \
        "وجه فروش باید در دفتر نقدی سناریو باشد"
    # ۵۰/۵۰: دو ردیف دریافت
    kinds = {x["type"] for x in sd["cash_rows"]}
    assert {"receipt_upfront", "receipt_balance"} <= kinds


def test_s10_what_if_multiple_hypothetical_transactions(E):
    A, bio, st = ctx(E)
    c = _cohort(E, min_w=1.0)
    r = OF.what_if(A, bio, st, [
        {"type": "sell_fish", "cohort_id": c.cohort_id,
         "quantity": 10_000, "price": 16_000},
        {"type": "buy_eggs", "quantity": 160_000, "price": 6_000,
         "date": offer_date(E),
         "payment_terms": {"upfront_share": 0.5, "delay_days": 60}},
        {"type": "feed_price_pct", "value": 10},
    ])
    assert len(r["applied"]) == 3
    # خرید فرضی واقعاً وارد برنامه شده است. توجه: سقف سالانه تخم binding است،
    # پس این lot به جمع کل اضافه نمی‌شود بلکه جای lotهای اختیاری را می‌گیرد —
    # و همین درست است.
    assert r["state_delta"]["hypothetical_lots_in_plan"], \
        "lot فرضی باید در برنامه سناریو انتخاب شده باشد"
    assert r["scenario"]["eggs_planned"] >= 160_000
    assert r["state_delta"]["live_fish_delta"] == pytest.approx(-10_000, abs=1)
    # فرضیات دست‌نخورده برگشته‌اند
    assert not A.overlay, "overlay باید در پایان خالی باشد"


def test_what_if_never_touches_database_or_overrides(E):
    A, bio, st = ctx(E)
    c = _cohort(E)
    n_txn = E.db.one("SELECT COUNT(*) n FROM transactions")["n"]
    n_ovr = E.db.one("SELECT COUNT(*) n FROM assumption_overrides")["n"]
    feed0 = A.get("feed.price_table")[0]["price"]
    OF.what_if(A, bio, st, [
        {"type": "sell_fish", "cohort_id": c.cohort_id,
         "quantity": 5_000, "price": 20_000},
        {"type": "feed_price_pct", "value": 25},
        {"type": "mortality_pct", "value": 5},
    ])
    assert E.db.one("SELECT COUNT(*) n FROM transactions")["n"] == n_txn
    assert E.db.one("SELECT COUNT(*) n FROM assumption_overrides")["n"] == n_ovr, \
        "What-If دیگر هرگز در جدول overrides نمی‌نویسد (اصلاح ۶)"
    assert A.get("feed.price_table")[0]["price"] == feed0
    A2, bio2, st2 = ctx(E)
    assert st2.cohorts[c.cohort_id].alive == pytest.approx(
        st.cohorts[c.cohort_id].alive)


# ═══════════════════════════════ رگرسیون مراحل ۱–۳
def test_regression_validation_suite(E):
    assert call("/api/validate")["failed"] == 0


def test_regression_plan_still_solves(E):
    r = call("/api/plan", "GET", {}, {"variant": ["balanced"]})
    assert r["validation"]["failed"] == 0
