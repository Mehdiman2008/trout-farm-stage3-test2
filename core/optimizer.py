"""
optimizer.py — بهینه‌سازی برنامه خرید و برداشت (مرحله ۲)
=========================================================
یک MILP روی PuLP/CBC. متغیرها:

  x[m, q, w] ∈ {0,1}   خرید یک lot به اندازه q در ماه m با هدف برداشت وزن w
  f[c, w]    ∈ [0,1]   سهمی از cohort موجود c که در وزن w برداشت می‌شود

`f` پیوسته است، پس بهینه‌ساز می‌تواند **partial harvest / grading** انتخاب
کند: بخشی از یک cohort در ۵ گرم و بخشی در ۱۵ گرم. این همان
«Optimizer-Based Forced Sale» است (Fix 3) — وقتی ظرفیت یا نقدینگی تنگ
می‌شود، خودِ مدل تصمیم می‌گیرد کدام cohort، چه تعداد و در چه وزنی زودتر
فروخته شود تا زیان اقتصادی کمینه گردد؛ نه یک قاعده سرانگشتی مثل
«قدیمی‌ترین cohort اول».

محدودیت‌ها:
  * ظرفیت استخر در هر هفته (استخر صحیح، از قبل گرد شده در پروفایل)
  * سرمایه در گردش در هر هفته ≤ سرمایه موجود
  * راهنمای ماهانه تخم (نرم، با جریمه) و سقف سخت (اختیاری)
  * عرضه ماهانه و سناریوی حجم سالانه
  * حداکثر تعداد lot در هر ماه
  * هر cohort موجود دقیقاً یک بار تخصیص داده شود (Σ f = 1)

تابع هدف:
  (1−λ)·contribution_base + λ·contribution_adverse − penalties
"""
from __future__ import annotations

import math

import sys

# دلیل دقیق در دسترس نبودن solver نگه داشته می‌شود تا عیب‌یابی ممکن باشد.
# «نصب نیست» و «نصب است ولی خطا می‌دهد» دو مسئله کاملاً متفاوت‌اند.
PULP_ERROR = None
PULP_VERSION = None
CBC_OK = None
CBC_ERROR = None

try:
    import pulp
    HAVE_PULP = True
    PULP_VERSION = getattr(pulp, "__version__", "?")
except Exception as _e:                              # pragma: no cover
    HAVE_PULP = False
    PULP_ERROR = f"{type(_e).__name__}: {_e}"


def check_solver() -> dict:
    """
    بررسی واقعی اینکه آیا می‌توان یک مدل کوچک را حل کرد.

    نصب بودن بسته PuLP کافی نیست؛ حل‌کننده CBC هم باید اجرا شود. این تابع
    یک مسئله یک‌متغیره را حل می‌کند تا هر دو لایه آزمایش شوند.
    """
    global CBC_OK, CBC_ERROR
    if not HAVE_PULP:
        CBC_OK = False
        CBC_ERROR = PULP_ERROR
        return solver_status()
    try:
        prob = pulp.LpProblem("healthcheck", pulp.LpMaximize)
        x = pulp.LpVariable("x", 0, 3, cat="Integer")
        prob += x
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        CBC_OK = (pulp.LpStatus[prob.status] == "Optimal"
                  and abs((x.value() or 0) - 3) < 1e-6)
        CBC_ERROR = None if CBC_OK else "حل‌کننده CBC نتیجه درست نداد"
    except Exception as e:                            # pragma: no cover
        CBC_OK = False
        CBC_ERROR = f"{type(e).__name__}: {e}"
    return solver_status()


def solver_status() -> dict:
    return {
        "available": bool(HAVE_PULP and CBC_OK is not False),
        "pulp_installed": HAVE_PULP,
        "pulp_version": PULP_VERSION,
        "cbc_ok": CBC_OK,
        "import_error": PULP_ERROR,
        "cbc_error": CBC_ERROR,
        "python": sys.executable,
        "install_command": f"{sys.executable} -m pip install PuLP",
    }


def _missing_message_fa() -> str:
    st = solver_status()
    if not st["pulp_installed"]:
        return (
            "موتور بهینه‌سازی در دسترس نیست — بسته PuLP در همین پایتون نصب نشده است.\n"
            f"پایتونی که برنامه با آن اجرا می‌شود: {st['python']}\n"
            f"دستور نصب: {st['install_command']}\n"
            f"جزئیات خطای import: {st['import_error']}\n"
            "داشبورد، ثبت داده و Live Farm State بدون آن هم کار می‌کنند؛ فقط تب "
            "«برنامه و اهداف» غیرفعال است."
        )
    return (
        "بسته PuLP نصب است ولی حل‌کننده CBC اجرا نشد.\n"
        f"نسخه PuLP: {st['pulp_version']} · پایتون: {st['python']}\n"
        f"خطا: {st['cbc_error']}\n"
        f"معمولاً با نصب دوباره حل می‌شود: {st['install_command']} --force-reinstall"
    )


PULP_MISSING_EN = "Optimisation engine unavailable — PuLP dependency is not installed."


class OptimizerUnavailable(RuntimeError):
    """PuLP الزامی است؛ نتیجه heuristic ضعیف به‌جای خروجی optimizer نمایش داده نمی‌شود."""

    def __init__(self):
        msg = _missing_message_fa()
        super().__init__(msg)
        self.message_fa = msg
        self.message_en = PULP_MISSING_EN
        self.status = solver_status()


class SolverTimeout(RuntimeError):
    """
    CBC از سقف زمانی خودش عبور کرد و با ناظر بیرونی متوقف شد.

    پارامتر timeLimit فقط در گره‌های branch-and-bound بررسی می‌شود؛ اگر LP
    ریشه به‌دلیل مقیاس بد ضرایب (جریمه‌های بزرگ + ضرایب سرمایه میلیاردی)
    گیر کند، CBC عملاً هرگز برنمی‌گردد. این ناظر همان تضمین سختِ زمان است.
    """

    def __init__(self, tl: int, wall: float):
        super().__init__(
            f"حل‌کننده CBC پس از {wall:.0f} ثانیه (سقف تنظیم‌شده {tl} ثانیه) "
            f"پاسخی نداد و متوقف شد. معمولاً با کوچک‌کردن "
            f"planning.solver_time_limit_s یا شل‌کردن محدودیت‌ها حل می‌شود.")
        self.tl = tl
        self.wall = wall


def _solve_with_watchdog(prob, tl: int):
    """اجرای CBC با تضمین سخت زمان: اگر ۳×سقف + ۱۵ ثانیه گذشت، فرزند cbc کشته می‌شود."""
    import os
    import signal
    import subprocess
    import threading

    wall = max(15.0, tl * 3.0 + 10.0)
    tripped = {"v": False}
    me = os.getpid()

    def _kill():
        tripped["v"] = True
        try:
            out = subprocess.run(["pgrep", "-P", str(me), "cbc"],
                                 capture_output=True, text=True)
            for pid in out.stdout.split():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except Exception:
                    pass
        except Exception:
            pass

    timer = threading.Timer(wall, _kill)
    timer.daemon = True
    timer.start()
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=tl))
    except Exception as e:
        if tripped["v"]:
            raise SolverTimeout(tl, wall) from e
        raise
    finally:
        timer.cancel()
    if tripped["v"]:
        raise SolverTimeout(tl, wall)


class Variant:
    """سه حالت مدیریتی برنامه."""

    ALL = {
        "max_profit": {"label_fa": "بیشینه سود", "risk_aversion": 0.0,
                       "utilisation": 1.00, "keep_reserve_free": False,
                       "wc_buffer": 1.00},
        "balanced": {"label_fa": "متعادل", "risk_aversion": 0.30,
                     "utilisation": 0.95, "keep_reserve_free": True,
                     "wc_buffer": 0.95},
        "conservative": {"label_fa": "محافظه‌کارانه", "risk_aversion": 0.60,
                         "utilisation": 0.85, "keep_reserve_free": True,
                         "wc_buffer": 0.80},
    }

    def __init__(self, name: str, A):
        if name not in self.ALL:
            raise ValueError(f"حالت برنامه نامعتبر: {name}")
        self.name = name
        cfg = dict(self.ALL[name])
        self.label_fa = cfg["label_fa"]
        self.risk_aversion = cfg["risk_aversion"]
        self.utilisation = min(cfg["utilisation"],
                               float(A.get("planning.pond_utilisation_target"))
                               if name != "max_profit" else cfg["utilisation"])
        self.keep_reserve_free = cfg["keep_reserve_free"] and \
            bool(A.get("planning.keep_reserve_free"))
        self.wc_buffer = cfg["wc_buffer"]


class PlanSolution:
    def __init__(self):
        self.status = "not_solved"
        self.solver = None
        self.objective = 0.0
        self.chosen_lots = []        # [{month, quantity, harvest_w, key}]
        self.cohort_split = {}       # cohort_id -> {harvest_w: fraction}
        self.selected = {}           # profile key -> weight (1.0 یا کسر)
        self.pond_shortfall = []     # کمبود استخر در هر هفته (متغیر کشسان)
        self.wc_shortfall = []       # کمبود سرمایه در گردش در هر هفته
        self.notes = []


def solve(A, model, cand_base, cand_adv, variant: Variant,
          wc_available: float, existing_pond_floor=None,
          pond_cuts: dict | None = None,
          force_group: str | None = None) -> PlanSolution:
    """
    cand_base / cand_adv: خروجی PlanModel.build_candidates برای دو سناریو.
    existing_pond_floor: نیاز استخر cohortهایی که در تصمیم نیستند (اگر باشد).
    """
    sol = PlanSolution()
    if not HAVE_PULP:                                # pragma: no cover
        raise OptimizerUnavailable()

    weeks = model.weeks
    new_base, ex_base = cand_base["new_lots"], cand_base["existing"]
    new_adv, ex_adv = cand_adv["new_lots"], cand_adv["existing"]

    op_ponds = int(A.get("farm.operational_ponds"))
    reserve = int(A.get("farm.reserve_ponds"))
    pond_cap = op_ponds * variant.utilisation
    if not variant.keep_reserve_free:
        pond_cap += reserve
    pond_cap = math.floor(pond_cap + 1e-9)

    guideline = float(A.get("egg.monthly_guideline"))
    hard_on = bool(A.get("egg.enforce_hard_monthly_max"))
    hard_max = float(A.get("egg.hard_monthly_max"))
    avail = float(A.get("planning.monthly_availability"))
    annual = float(A.get("planning.annual_scenario"))
    max_lots = int(A.get("planning.max_lots_per_month"))
    pen = float(A.get("planning.guideline_penalty_per_egg"))
    pond_pen = float(A.get("planning.pond_breach_penalty"))
    wc_pen = float(A.get("planning.wc_breach_penalty"))
    lam = variant.risk_aversion
    wc_limit = wc_available * variant.wc_buffer

    # مقیاس پولی داخل مدل: تومان → میلیون تومان (فقط داخل solver).
    # بدون این، ماتریس محدودیت‌ها ضرایبی از ~۱ (استخر) تا ~۱e9 (سرمایه) دارد
    # و LP ریشه CBC می‌تواند به‌دلیل conditioning بد عملاً قفل کند. اقتصاد
    # مسئله ذره‌ای عوض نمی‌شود؛ خروجی در پایان به تومان برگردانده می‌شود.
    MSCALE = 1e6

    prob = pulp.LpProblem("trout_plan", pulp.LpMaximize)

    x = {k: pulp.LpVariable(f"x_{i}", cat="Binary")
         for i, k in enumerate(new_base)}
    f = {k: pulp.LpVariable(f"f_{i}", lowBound=0, upBound=1)
         for i, k in enumerate(ex_base)}
    # عبور از راهنمای ماهانه: متغیر نرم
    months = sorted({p.month for p in new_base.values()})
    over = {m: pulp.LpVariable(f"ov_{i}", lowBound=0) for i, m in enumerate(months)}
    # محدودیت‌های کشسان: مدل هرگز Infeasible نمی‌شود، بلکه کمبود را صریح
    # گزارش و کمینه می‌کند. وضعیت واقعی امروز ممکن است خودش از ظرفیت عبور
    # کرده باشد و یک برنامه «غیرممکن» به مدیر کمکی نمی‌کند.
    pond_over = [pulp.LpVariable(f"po_{t}", lowBound=0) for t in range(weeks + 1)]
    wc_over = [pulp.LpVariable(f"wo_{t}", lowBound=0) for t in range(weeks + 1)]

    def blend(kbase, kadv, pool_b, pool_a, attr, t=None):
        pb = pool_b[kbase]
        pa = pool_a.get(kadv, pb)
        vb = getattr(pb, attr)[t] if t is not None else sum(getattr(pb, attr))
        va = getattr(pa, attr)[t] if t is not None else sum(getattr(pa, attr))
        return (1 - lam) * vb + lam * va

    # ---------------------------------------------------------- objective
    # همه جمله‌های پولی بر MSCALE تقسیم می‌شوند (میلیون تومان).
    obj = []
    for k in new_base:
        c = (blend(k, k, new_base, new_adv, "revenue")
             - blend(k, k, new_base, new_adv, "feed_cost")
             - sum(new_base[k].egg_cost))
        obj.append((c / MSCALE) * x[k])
    for k in ex_base:
        c = (blend(k, k, ex_base, ex_adv, "revenue")
             - blend(k, k, ex_base, ex_adv, "feed_cost"))
        obj.append((c / MSCALE) * f[k])
    obj.append(-(pen / MSCALE) * pulp.lpSum(over.values()))
    obj.append(-(pond_pen / MSCALE) * pulp.lpSum(pond_over))
    # wc_over خودش به «میلیون تومان» تعریف می‌شود (پایین)؛ جریمه هر میلیون:
    obj.append(-(wc_pen * MSCALE / MSCALE) * pulp.lpSum(wc_over))
    prob += pulp.lpSum(obj)

    # -------------------------------------------------------- constraints
    # ۱) هر cohort موجود دقیقاً یک بار (اجازه تقسیم بین وزن‌ها)
    by_cohort = {}
    for k in ex_base:
        cid = k.split("|")[1]
        by_cohort.setdefault(cid, []).append(k)
    for cid, keys in by_cohort.items():
        prob += pulp.lpSum(f[k] for k in keys) == 1, f"cohort_{cid}"

    # ۲) ظرفیت استخر در هر هفته
    floor = existing_pond_floor or [0.0] * (weeks + 1)
    for t in range(weeks + 1):
        terms = [new_base[k].ponds[t] * x[k] for k in new_base if new_base[k].ponds[t]]
        terms += [ex_base[k].ponds[t] * f[k] for k in ex_base if ex_base[k].ponds[t]]
        if terms:
            # برش ظرفیت: اگر اعتبارسنجی صحیح نشان دهد این هفته پس از گرد
            # کردن از ظرفیت عبور می‌کند، در دور بعد سقف همان هفته سفت‌تر
            # می‌شود تا برنامه نهایی واقعاً feasible باشد.
            cut = float((pond_cuts or {}).get(t, 0.0))
            prob += (pulp.lpSum(terms) <= pond_cap - floor[t] - cut + pond_over[t],
                     f"pond_w{t}")

    # ۳) سرمایه در گردش در هر هفته (برحسب میلیون تومان؛ wc_over هم میلیون است)
    for t in range(weeks + 1):
        terms = [(new_base[k].capital[t] / MSCALE) * x[k]
                 for k in new_base if new_base[k].capital[t]]
        terms += [(ex_base[k].capital[t] / MSCALE) * f[k]
                  for k in ex_base if ex_base[k].capital[t]]
        if terms:
            prob += pulp.lpSum(terms) <= (wc_limit / MSCALE) + wc_over[t], f"wc_w{t}"

    # ۴) راهنما / سقف سخت / عرضه ماهانه / تعداد lot
    for m in months:
        keys = [k for k in new_base if new_base[k].month == m]
        qty = pulp.lpSum(new_base[k].quantity * x[k] for k in keys)
        prob += qty <= avail, f"avail_{m}"
        prob += qty - over[m] <= guideline, f"guide_{m}"
        if hard_on:
            prob += qty <= hard_max, f"hard_{m}"
        prob += pulp.lpSum(x[k] for k in keys) <= max_lots, f"lots_{m}"

    # ۵) سناریوی حجم سالانه (۱۲ ماه اول تصمیم)
    prob += pulp.lpSum(new_base[k].quantity * x[k] for k in new_base) <= annual, "annual"

    # ۶) آفر اجباری: دقیقاً یکی از وزن‌های برداشت این lot انتخاب شود.
    #    برای ارزیابی آفر واقعی لازم است: «اگر این lot را بخریم، بهترین
    #    برنامه ممکن چه می‌شود؟»
    #    force_group می‌تواند یک گروه یا فهرستی از گروه‌ها باشد (سناریوی
    #    What-If با چند خرید فرضی).
    if force_group:
        groups = [force_group] if isinstance(force_group, str) else list(force_group)
        for i, grp in enumerate(groups):
            keys = [k for k in new_base if new_base[k].group == grp]
            if keys:
                prob += pulp.lpSum(x[k] for k in keys) == 1, f"forced_offer_{i}"

    # ------------------------------------------------------------- solve
    tl = int(A.get("planning.solver_time_limit_s"))
    try:
        _solve_with_watchdog(prob, tl)
    except SolverTimeout:
        raise
    except Exception as e:                            # pragma: no cover
        raise RuntimeError(f"خطای اجرای solver: {e}") from e

    sol.status = pulp.LpStatus[prob.status]
    sol.solver = "CBC (PuLP)"
    sol.objective = (pulp.value(prob.objective) or 0.0) * MSCALE   # ← تومان
    for k, v in x.items():
        val = v.value() or 0
        if val > 0.5:
            p = new_base[k]
            sol.selected[k] = 1.0
            sol.chosen_lots.append({"key": k, "month": p.month,
                                    "purchase_date": p.purchase_date.isoformat(),
                                    "quantity": p.quantity,
                                    "harvest_w": p.harvest_w})
    for k, v in f.items():
        val = v.value() or 0
        if val > 1e-6:
            cid = k.split("|")[1]
            w = float(k.split("|")[2])
            sol.selected[k] = val
            sol.cohort_split.setdefault(cid, {})[w] = val
    sol.pond_shortfall = [(v.value() or 0.0) for v in pond_over]
    sol.wc_shortfall = [(v.value() or 0.0) * MSCALE for v in wc_over]  # ← تومان
    peak_ps = max(sol.pond_shortfall) if sol.pond_shortfall else 0
    if peak_ps > 0.01:
        wk = sol.pond_shortfall.index(peak_ps)
        sol.notes.append(
            f"حتی با بهترین برنامه، در هفته {wk} به {peak_ps:.0f} استخر بیش از "
            f"ظرفیت مجاز نیاز است. مدل این کمبود را کمینه کرده است.")
    peak_ws = max(sol.wc_shortfall) if sol.wc_shortfall else 0
    if peak_ws > 1:
        sol.notes.append(
            f"اوج نیاز سرمایه در گردش {peak_ws:,.0f} تومان بیش از سرمایه موجود است.")
    ov = sum((v.value() or 0) for v in over.values())
    if ov > 1:
        sol.notes.append(f"برنامه {ov:,.0f} عدد بیش از راهنمای ماهانه خرید می‌کند "
                         f"(مجاز است، ولی جریمه‌دار).")
    sol.chosen_lots.sort(key=lambda r: r["purchase_date"])
    return sol
