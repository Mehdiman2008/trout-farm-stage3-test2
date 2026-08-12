"""
plan_model.py — ساخت پروفایل‌های هفتگی برای برنامه‌ریزی (مرحله ۲)
====================================================================
هر «تصمیم ممکن» (یک lot تخم با اندازه و وزن برداشت مشخص، یا یک cohort
موجود که در وزن مشخصی برداشت شود) به یک مجموعه ضریب هفتگی تبدیل می‌شود:

    ponds[t] · feed_kg[t] · feed_cost[t] · revenue[t] · capital[t]

چون این ضرایب از پیش محاسبه می‌شوند، مدل بهینه‌سازی خطی می‌ماند و در عین
حال «استخر صحیح» (ceil) و «برداشت چندموجی» را حفظ می‌کند.

### برداشت چندموجی بر اساس پراکندگی رشد
یک cohort در یک روز به یک وزن نمی‌رسد. اگر وزن تک‌ماهی لاگ‌نرمال با میانگین
`mean(t)` باشد، ماهی در چندک q وزنی برابر `mean(t) × k_q` دارد. پس ماهیِ
تندرشد وقتی به وزن هدف wH می‌رسد که میانگین cohort هنوز `wH / k_90` باشد:

    سن رسیدن گروه تندرشد  = age_at_weight(wH / k_90)
    سن رسیدن گروه معمول   = age_at_weight(wH)
    سن رسیدن گروه کندرشد  = age_at_weight(wH / k_10)

به این ترتیب برداشت به‌صورت سه موج (grading / partial harvest) انجام می‌شود،
فشار استخر پله‌پله آزاد می‌شود و forecast واقع‌بینانه‌تر است.

### تقویم غیرچرخه‌ای (Fix 2)
همه شاخص‌های زمانی هفته‌های واقعی تقویمی از تاریخ مبنا هستند؛ هیچ modulo-52.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

from .state import _age


class Grid:
    """شبکه زمانی هفتگی غیرچرخه‌ای."""

    def __init__(self, start: date, weeks: int):
        self.start = start
        self.weeks = weeks
        self.dates = [start + timedelta(days=7 * t) for t in range(weeks + 1)]

    def index_of(self, day: date) -> int:
        return max(0, min(self.weeks, (day - self.start).days // 7))

    def month_of(self, t: int) -> str:
        return self.dates[t].strftime("%Y-%m")

    def quarter_of(self, t: int) -> str:
        d0 = self.dates[t]
        return f"{d0.year}-Q{(d0.month - 1) // 3 + 1}"


class Scenario:
    """
    ضرایب سناریو. سناریوی نامساعد از پارامترهای stochastic در config ساخته
    می‌شود (Fix 5) — هیچ عددی در پایتون hard-code نیست.
    """

    def __init__(self, name: str, mortality_mult=1.0, fcr_mult=1.0,
                 price_mult=1.0, feed_price_mult=1.0, growth_mult=1.0):
        self.name = name
        self.mortality_mult = mortality_mult
        self.fcr_mult = fcr_mult
        self.price_mult = price_mult
        self.feed_price_mult = feed_price_mult
        self.growth_mult = growth_mult

    @staticmethod
    def base():
        return Scenario("base")

    @staticmethod
    def adverse(A, k: float = 1.0):
        """یک انحراف معیار در جهت نامساعد روی متغیرهای کلیدی."""
        return Scenario(
            "adverse",
            mortality_mult=1.0 + k * float(A.get("stochastic.mortality_cv")),
            fcr_mult=1.0 + k * float(A.get("stochastic.fcr_cv")),
            price_mult=1.0 - k * float(A.get("stochastic.sale_price_cv")),
            feed_price_mult=1.0 + k * float(A.get("stochastic.feed_price_cv")),
            growth_mult=1.0 - k * float(A.get("stochastic.growth_duration_cv")),
        )


class Profile:
    """ضرایب هفتگی یک تصمیم ممکن، برای یک سناریو."""

    __slots__ = ("key", "kind", "label", "month", "quantity", "harvest_w",
                 "purchase_date", "ponds", "feed_kg", "feed_cost", "revenue",
                 "egg_cost", "capital", "harvest_fish", "harvest_weeks",
                 "sold_fish", "survival_at_harvest", "first_week", "last_week",
                 "fish", "weight", "group", "mortality")

    def __init__(self, key, kind, label, weeks):
        self.key = key
        self.kind = kind                      # "new_lot" | "existing"
        self.label = label
        self.month = None
        self.quantity = 0.0
        self.harvest_w = 0.0
        self.purchase_date = None
        z = [0.0] * (weeks + 1)
        self.ponds = list(z)
        self.feed_kg = list(z)
        self.feed_cost = list(z)
        self.revenue = list(z)
        self.egg_cost = list(z)
        self.capital = list(z)
        self.harvest_fish = list(z)
        self.fish = list(z)              # تعداد زنده در پایان هفته
        self.mortality = list(z)         # تلفات هفته (جدا از برداشت)
        self.weight = list(z)            # وزن متوسط در هر هفته
        self.group = key                 # شناسه گروه فیزیکی (cohort یا lot)
        self.harvest_weeks = []
        self.sold_fish = 0.0
        self.survival_at_harvest = 0.0
        self.first_week = 0
        self.last_week = 0

    # ---------------------------------------------------------- aggregates
    def contribution(self) -> float:
        return sum(self.revenue) - sum(self.feed_cost) - sum(self.egg_cost)

    def peak_capital(self) -> float:
        return max(self.capital) if self.capital else 0.0


class PlanModel:
    """سازنده پروفایل‌ها از روی وضعیت واقعی مزرعه."""

    def __init__(self, A, bio, state, weeks: int | None = None):
        self.A, self.bio, self.state = A, bio, state
        self.weeks = int(weeks or A.get("planning.eval_horizon_weeks"))
        self.grid = Grid(state.as_of, self.weeks)
        self.wave_fracs = [float(x) for x in A.get("planning.harvest_wave_fractions")]
        self.slow_q = float(A.get("heterogeneity.slow_quantile"))
        self.fast_q = float(A.get("heterogeneity.fast_quantile"))

    # ------------------------------------------------------ heterogeneity
    def wave_ages(self, harvest_w: float, cv: float, growth_mult: float = 1.0) -> list:
        """
        سن رسیدن سه گروه (تندرشد، معمول، کندرشد) به وزن هدف.
        بازگشت: [(سن روز, سهم), ...] مرتب از زودترین به دیرترین.
        """
        bio = self.bio
        k_fast = bio.weight_quantile(1.0, self.fast_q, cv)     # نسبت چندک به میانگین
        k_slow = bio.weight_quantile(1.0, self.slow_q, cv)
        f_fast, f_typ, f_slow = self.wave_fracs
        targets = [(harvest_w / max(k_fast, 1e-6), f_fast),
                   (harvest_w, f_typ),
                   (harvest_w / max(k_slow, 1e-6), f_slow)]
        out = []
        for mean_w_needed, frac in targets:
            age = bio.age_at_weight(max(mean_w_needed, bio.knots[0][1] * 1.0001))
            if growth_mult != 1.0:
                age = age / max(growth_mult, 1e-6)    # رشد کندتر = دیرتر
            out.append((age, frac))
        return out

    # ------------------------------------------------- saleability (۵)
    def saleability(self, w: float) -> float:
        """احتمال یافتن مشتری در این وزن — Modelling Assumption."""
        for b in self.A.get("planning.saleability"):
            if float(b["w_min"]) <= w < float(b["w_max"]):
                return max(0.0, min(1.0, float(b["prob"])))
        return 1.0

    def sale_waves(self, harvest_w: float, cv: float, growth_mult: float = 1.0) -> list:
        """
        موج‌های فروش = پراکندگی رشد × احتمال یافتن مشتری.

        احتمال فروش هرگز در قیمت ضرب نمی‌شود. نبود مشتری یعنی فروش به دوره
        بعد موکول می‌شود: ماهی در استخر می‌ماند، خوراک می‌خورد، در معرض
        تلفات است و سرمایه بیشتر درگیر می‌ماند.

        اگر احتمال فروش p باشد، سهم فروش در دوره k برابر (1-p)^k · p است و
        باقیمانده پس از آخرین دوره در همان دوره فروخته می‌شود.
        """
        p = self.saleability(harvest_w)
        delay = int(self.A.get("planning.sale_delay_days"))
        kmax = int(self.A.get("planning.max_sale_delay_periods"))
        out = []
        for age, frac in self.wave_ages(harvest_w, cv, growth_mult):
            if p >= 0.999:
                out.append((age, frac, 0))
                continue
            rem = 1.0
            for k in range(kmax):
                share = rem * p
                if share > 1e-9:
                    out.append((age + k * delay, frac * share, k))
                rem -= share
            if rem > 1e-9:
                out.append((age + kmax * delay, frac * rem, kmax))
        out.sort(key=lambda x: x[0])
        return out

    # ------------------------------------------------ شبیه‌سازی یک cohort
    def _simulate(self, p, t0: int, anchor, base_count: float, sc: Scenario,
                  harvest_w: float, waves: list, weight_fn, survive_fn,
                  unit_cost: float):
        """
        هسته مشترک شبیه‌سازی، با گذار حالت صریح:

            موجودی ابتدای هفته
            − تلفات
            − برداشت/فروش
            = موجودی پایان هفته

        خوراک فقط برای ماهی زنده همان هفته محاسبه می‌شود؛ ماهی فروخته‌شده
        در هفته‌های بعد نه خوراک می‌خورد و نه دوباره به‌عنوان تلفات شمرده
        می‌شود. سرمایه هم به نسبت ماهی فروخته‌شده آزاد می‌گردد.
        """
        bio, g = self.bio, self.grid
        wave_at: dict[int, float] = {}
        for age, frac, k in waves:
            wk = g.index_of(anchor(age))
            wave_at[wk] = wave_at.get(wk, 0.0) + frac
        p.harvest_weeks = sorted(wave_at)
        p.last_week = min(self.weeks, max(wave_at) if wave_at else t0)

        alive = base_count                 # موجودی ابتدای دوره
        remaining_frac = 1.0               # سهم برداشت‌نشده از cohort اولیه
        capital_pool = base_count * unit_cost
        prev_w = weight_fn(t0)

        for t in range(t0, min(self.weeks, p.last_week) + 1):
            w = weight_fn(t)

            # ۱) تلفات هفته — فقط روی ماهی زنده
            if t > t0 and alive > 0:
                sr = survive_fn(t - 1, t)
                sr = max(0.0, 1.0 - (1.0 - sr) * sc.mortality_mult)
                died = alive * (1.0 - sr)
                alive -= died
                p.mortality[t] = died

            # ۲) خوراک — برای ماهی‌ای که این هفته زنده بوده و رشد کرده
            if t > t0 and alive > 0:
                kg = bio.feed_kg_for_growth(alive, prev_w, w,
                                            n_died=p.mortality[t]) * sc.fcr_mult
                p.feed_kg[t] = kg
                cost = kg * bio.feed_price(prev_w) * sc.feed_price_mult
                p.feed_cost[t] = cost
                capital_pool += cost

            # ۳) برداشت/فروش — از موجودی همین لحظه، نه از عدد اولیه
            frac = wave_at.get(t, 0.0)
            if frac > 1e-9 and alive > 0 and remaining_frac > 1e-9:
                take_share = min(1.0, frac / remaining_frac)
                n = alive * take_share
                sale_w = max(harvest_w, w)          # اگر فروش عقب افتاده، وزن بیشتر است
                price = bio.sale_price(sale_w, on_date=g.dates[t].isoformat())
                p.revenue[t] += n * price * sc.price_mult
                p.harvest_fish[t] += n
                p.sold_fish += n
                alive -= n
                remaining_frac = max(0.0, remaining_frac - frac)
                capital_pool *= (1.0 - take_share)   # آزادسازی سرمایه (۱۰)

            # ۴) ثبت وضعیت پایان هفته
            p.fish[t] = alive
            p.weight[t] = w
            if alive >= 1 and bio.counts_toward_pond_capacity(w):
                p.ponds[t] = math.ceil(alive / bio.fish_per_pond(w))
            p.capital[t] = capital_pool if alive >= 1 else 0.0
            prev_w = w

        p.survival_at_harvest = p.sold_fish / base_count if base_count else 0.0
        return p

    # -------------------------------------------------------- new-lot plan
    def new_lot_profile(self, purchase_date: date, qty: float, harvest_w: float,
                        sc: Scenario) -> Profile:
        A, bio, g = self.A, self.bio, self.grid
        key = f"L|{purchase_date.isoformat()}|{int(qty)}|{harvest_w:g}"
        p = Profile(key, "new_lot",
                    f"{int(qty):,} تخم در {purchase_date.isoformat()} → برداشت {harvest_w:g}g",
                    self.weeks)
        p.month = purchase_date.strftime("%Y-%m")
        p.group = f"L|{purchase_date.isoformat()}|{int(qty)}"
        p.quantity = qty
        p.harvest_w = harvest_w
        p.purchase_date = purchase_date

        egg_price = float(A.get_at("egg.base_price", purchase_date.isoformat()))
        t0 = g.index_of(purchase_date)
        p.first_week = t0
        p.egg_cost[t0] = qty * egg_price

        cv = bio.cv_at_weight(harvest_w)
        waves = self.sale_waves(harvest_w, cv, sc.growth_mult)

        def age_of(t):
            a = (g.dates[t] - purchase_date).days
            return a * sc.growth_mult if sc.growth_mult != 1.0 else a

        return self._simulate(
            p, t0,
            anchor=lambda age: purchase_date + timedelta(days=int(round(age))),
            base_count=qty, sc=sc, harvest_w=harvest_w, waves=waves,
            weight_fn=lambda t: bio.weight_at_age(max(0.0, age_of(t))),
            survive_fn=lambda a, b: bio.survival_ratio(
                max(0.0, (g.dates[a] - purchase_date).days),
                max(0.0, (g.dates[b] - purchase_date).days)),
            unit_cost=egg_price)

    # ------------------------------------------------- existing cohort plan
    def existing_profile(self, c, harvest_w: float, sc: Scenario) -> Profile:
        """
        پروفایل یک cohort موجود اگر «کل» آن در وزن harvest_w برداشت شود.
        بهینه‌ساز می‌تواند کسری از cohort را به هر وزن اختصاص دهد
        (partial harvest / grading) چون این پروفایل‌ها خطی ترکیب می‌شوند.
        """
        A, bio, st, g = self.A, self.bio, self.state, self.grid
        key = f"C|{c.cohort_id}|{harvest_w:g}"
        p = Profile(key, "existing",
                    f"{c.cohort_id} → برداشت {harvest_w:g}g", self.weeks)
        p.quantity = c.alive
        p.group = f"C|{c.cohort_id}"
        p.harvest_w = harvest_w
        p.purchase_date = c.purchase_date
        p.first_week = 0

        cv = st.cv_of(c, max(harvest_w, c.mean_weight))
        waves = self.sale_waves(harvest_w, cv, sc.growth_mult)

        def anchor(age):
            day = c.purchase_date + timedelta(
                days=int(round(age - c.growth_offset_days)))
            return max(day, st.as_of)

        def weight_fn(t):
            if sc.growth_mult != 1.0:
                eff = _age(c.purchase_date, g.dates[t]) + c.growth_offset_days
                return bio.weight_at_age(eff * sc.growth_mult)
            return st.weight_of(c, g.dates[t])

        def survive_fn(a, b):
            a0 = _age(c.purchase_date, g.dates[a])
            a1 = _age(c.purchase_date, g.dates[b])
            return bio.survival_ratio(a0, a1)

        unit_cost = ((c.egg_count * c.egg_price / c.alive) if c.alive else 0.0)
        return self._simulate(p, 0, anchor, c.alive, sc, harvest_w, waves,
                              weight_fn, survive_fn, unit_cost)

    # ---------------------------------------------------- candidate builder
    def decision_months(self, n: int | None = None) -> list:
        n = int(n or self.A.get("planning.decision_months"))
        day = int(self.A.get("planning.purchase_day_of_month"))
        out = []
        y, m = self.state.as_of.year, self.state.as_of.month
        for k in range(n):
            mm = m + k
            yy = y + (mm - 1) // 12
            mm = (mm - 1) % 12 + 1
            dt = date(yy, mm, day)
            if dt < self.state.as_of:
                dt = self.state.as_of + timedelta(days=1)
            out.append((f"{yy}-{mm:02d}", dt))
        return out

    def build_candidates(self, sc: Scenario) -> dict:
        """همه پروفایل‌های ممکن: lotهای جدید + cohortهای موجود."""
        lots = [float(x) for x in self.A.get("planning.lot_candidates")]
        hws = [float(x) for x in self.A.get("planning.harvest_weights")]
        new_lots, existing = {}, {}
        for (mkey, pdate) in self.decision_months():
            for q in lots:
                for w in hws:
                    pr = self.new_lot_profile(pdate, q, w, sc)
                    pr.month = mkey
                    new_lots[pr.key] = pr
        for c in self.state.cohorts.values():
            if c.alive < 1:
                continue
            for w in hws:
                if w < c.mean_weight - 1e-9:
                    continue          # از این وزن گذشته است
                existing[f"C|{c.cohort_id}|{w:g}"] = self.existing_profile(c, w, sc)
            # همیشه گزینه «فروش فوری در وزن فعلی» موجود باشد
            wnow = round(max(c.mean_weight, 1.0), 1)
            k = f"C|{c.cohort_id}|{wnow:g}"
            if k not in existing:
                existing[k] = self.existing_profile(c, wnow, sc)
        return {"new_lots": new_lots, "existing": existing}
