"""
Модель прогноза расходов Социального фонда России на пенсии, 2015–2050.

Структура (см. data/README_data.md):
  1. Демография: когортно-компонентный пересчёт (год × пол × возраст 0..100+)
     на основе UN WPP 2024; миграция восстанавливается как остаток WPP,
     что позволяет менять смертность/миграцию в сценариях.
  2. Численность пенсионеров: население × коэффициент охвата, зависящий от
     пенсионного возраста когорты (350-ФЗ) и профиля досрочных выходов.
     Параметры охвата калибруются на факте 2018–2025.
  3. Размер страховой пенсии: ИПК × СПК + ФВ × (1 + доля 80+).
     Средний ИПК — модель «запас-поток» (выбытие по смертности, приток
     новых пенсионеров с ИПК, растущим во времени). Калибровка на 2018–2025.
  4. Индексация: факт до 2026, план 2027–2029 (февраль по ИПЦ + апрель),
     после 2029 — сценарное правило.
  5. Расходы на страховые пенсии = 12 × Σ N × p̄ × λ (λ — калибровочный
     множитель: доставка, надбавки, перерасчёты, структура).
  6. Доходы ОПС = ФОТ × τ_эфф, трансферт = разрыв.
  7. Сценарии + Монте-Карло.

Запуск:
    python src/pension_model.py            # данные из data/, результаты в output/
    python pension_model.py --mc 500          # с Монте-Карло
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

AGES = np.arange(0, 101)          # 100 = 100+
SEXES = ("M", "F")
BASE_YEAR = 2025                  # последний год факта по расходам
START_YEAR = 2015
END_YEAR = 2050


# ----------------------------------------------------------------------------
# Конфигурация сценария
# ----------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str = "base"
    # макро после горизонта прогноза МЭР (2029)
    cpi_long: float = 4.0                 # ИПЦ, %
    real_wage_long: float = 1.5           # рост реальной зарплаты, %
    real_gdp_long: float = 1.2            # рост реального ВВП, %
    gdp_deflator_extra: float = 0.0       # дефлятор ВВП = ИПЦ + extra
    # индексация страховых пенсий после 2029
    #   'two_stage'   : февраль = ИПЦ(t-1) дек/дек, апрель = alpha × реальная з/п(t-1)
    #   'cpi_only'    : только ИПЦ
    #   'wage'        : по росту номинальной зарплаты (сохранение замещения)
    indexation_rule: str = "two_stage"
    april_alpha: float = 0.8
    # социальные пенсии: индексация по ПМП ≈ ИПЦ + soc_real_extra
    soc_real_extra: float = 0.5
    # демография
    mortality_mult_2050: float = 1.0      # множитель qx (возраст 50+) к 2050, линейно с 2026
    migration_mult: float = 1.0           # множитель остаточной миграции WPP
    fertility_mult: float = 1.0           # множитель численности возраста 0
    # пенсионная система
    working_share_long: float | None = None   # доля работающих пенсионеров (None = факт 2025)
    raise_retirement_age_2035: float = 0.0     # доп. повышение возраста (лет) для когорт, достигающих его с 2035
    contrib_rate_mult: float = 1.0             # множитель эффективного тарифа взносов
    employment_mult_older: float = 1.0         # занятость в возрасте 55–RA
    # стохастика (используется в Монте-Карло)
    cpi_shocks: dict = field(default_factory=dict)
    rw_shocks: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Загрузка данных
# ----------------------------------------------------------------------------
class Data:
    def __init__(self, path: str | Path):
        p = Path(path)
        self.pop = pd.read_csv(p / "population_by_age_sex_2015_2050.csv")
        self.lt = pd.read_csv(p / "life_tables_2015_2050.csv")
        self.ra = pd.read_csv(p / "retirement_age_schedule_350fz.csv")
        self.spk = pd.read_csv(p / "pension_point_value_fixed_payment.csv", parse_dates=["effective_date"])
        self.idx_plan = pd.read_csv(p / "indexation_plan_2027_2029.csv")
        self.pens = pd.read_csv(p / "pensioners_and_avg_pensions.csv")
        self.budget = pd.read_csv(p / "sfr_budget_2015_2029.csv")
        self.wages = pd.read_csv(p / "wages.csv")
        self.macro = pd.read_csv(p / "macro.csv")
        self.pmp = pd.read_csv(p / "pensioner_subsistence_minimum.csv")
        self._prepare()

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _series(df, ind, val="value", key="year", status=None):
        d = df[df.indicator == ind]
        if status is not None:
            d = d[d.status.isin(status if isinstance(status, (list, tuple)) else [status])]
        s = d.set_index(key)[val].dropna()
        s.index = s.index.astype(int)
        return s.sort_index()

    def pens_jan1(self, ind):
        d = self.pens[(self.pens.indicator == ind) & self.pens.date.astype(str).str.endswith("-01-01")]
        s = pd.Series(d.value.values, index=d.date.str[:4].astype(int)).dropna()
        return s.sort_index()

    def budget_fact(self, ind):
        return self._series(self.budget, ind, "value_bn_rub", status="fact")

    def budget_plan(self, ind):
        return self._series(self.budget, ind, "value_bn_rub", status="plan")

    def _prepare(self):
        # население: массивы [год][пол] -> вектор 101
        self.P = {}
        for y, g in self.pop.groupby("year"):
            g = g.sort_values("age")
            self.P[int(y)] = {"M": g.male_thous.values.astype(float), "F": g.female_thous.values.astype(float)}
        # дожитие S_x (из возраста x в x+1 за год) по lx
        self.S = {}
        for (y, sex), g in self.lt.groupby(["year", "sex"]):
            g = g.sort_values("age")
            lx = g.lx.values.astype(float)
            ex = g.ex.values.astype(float)
            Lx = np.empty_like(lx)
            Lx[:-1] = 0.5 * (lx[:-1] + lx[1:])
            Lx[-1] = lx[-1] * ex[-1]
            S = np.empty_like(lx)
            S[:-2] = Lx[1:-1] / Lx[:-2]
            S[-2] = Lx[-1] / (Lx[-2] + Lx[-1])   # 99 -> 100+
            S[-1] = ex[-1] / (1.0 + ex[-1])      # 100+ остаётся
            self.S[(int(y), "M" if sex == "Male" else "F")] = np.clip(S, 0, 1)
        # остаточная миграция WPP: M_{t+1,x} = P_{t+1,x} - сдвиг(P_t × S_t)
        self.Mig = {}
        years = sorted(self.P)
        for y in years[:-1]:
            for s in SEXES:
                surv = self._shift(self.P[y][s], self.S[(y, s)])
                self.Mig[(y + 1, s)] = self.P[y + 1][s] - surv  # возраст 0 = рождения (включены)
        # пенсионный возраст по когортам
        self.ra = self.ra.set_index("birth_year")

    @staticmethod
    def _shift(P, S):
        out = np.zeros_like(P)
        out[1:] = P[:-1] * S[:-1]
        out[-1] += P[-1] * S[-1]
        return out

    def ret_age(self, birth_year: np.ndarray, sex: str, kind="general") -> np.ndarray:
        col = {("general", "M"): "men_general_age", ("general", "F"): "women_general_age",
               ("social", "M"): "social_pension_age_men", ("social", "F"): "social_pension_age_women"}[(kind, sex)]
        s = self.ra[col]
        b = np.clip(birth_year, s.index.min(), s.index.max())
        return s.reindex(b).values.astype(float)


# ----------------------------------------------------------------------------
# Траектории СПК / ФВ и индексации
# ----------------------------------------------------------------------------
def build_macro(data: Data, sc: Scenario) -> pd.DataFrame:
    yrs = range(START_YEAR, END_YEAR + 1)
    m = pd.DataFrame(index=pd.Index(yrs, name="year"))
    cpi_dd = data._series(data.macro, "cpi_dec_dec")
    cpi_dd_f = data._series(data.macro, "cpi_dec_dec_mer")
    cpi_avg_f = data._series(data.macro, "cpi_annual_avg_mer")
    cpi_avg_h = data._series(data.macro, "cpi_annual_avg")
    gdp = data._series(data.macro, "gdp_nominal")
    rw = data._series(data.wages, "real_wage_growth_forecast")
    w = data._series(data.wages, "avg_nominal_wage")
    nw26 = data._series(data.wages, "nominal_wage_growth_forecast")

    m["cpi_dd"] = [cpi_dd.get(y, cpi_dd_f.get(y, 100 + sc.cpi_long)) for y in yrs]
    m["cpi_avg"] = [cpi_avg_h.get(y, cpi_avg_f.get(y, np.nan)) for y in yrs]
    m["cpi_avg"] = m.cpi_avg.fillna(m.cpi_dd)  # для истории без среднегодового — дек/дек
    for y, v in sc.cpi_shocks.items():          # стохастика
        if y in m.index and y > 2029:
            m.loc[y, "cpi_dd"] += v
            m.loc[y, "cpi_avg"] += v
    # реальная и номинальная зарплата
    m["real_wage"] = [rw.get(y, sc.real_wage_long) for y in yrs]
    for y, v in sc.rw_shocks.items():
        if y in m.index and y > 2029:
            m.loc[y, "real_wage"] += v
    wage = []
    for y in yrs:
        if y in w.index:
            wage.append(float(w[y]))
        elif y == 2026 and 2026 in nw26.index:
            wage.append(wage[-1] * (1 + nw26[2026] / 100))
        else:
            wage.append(wage[-1] * (m.loc[y, "cpi_avg"] / 100) * (1 + m.loc[y, "real_wage"] / 100))
    m["wage"] = wage
    # ВВП (факт 2015–2024 отсутствует в пакете → NaN; с 2025 — МЭР, далее экстраполяция)
    g = []
    for y in yrs:
        if y in gdp.index:
            g.append(float(gdp[y]))
        elif y > gdp.index.max():
            g.append(g[-1] * (1 + sc.real_gdp_long / 100) * (m.loc[y, "cpi_avg"] / 100 + sc.gdp_deflator_extra / 100))
        else:
            g.append(np.nan)
    m["gdp"] = g
    # уровень цен (2025 = 1)
    lvl = np.cumprod(m.cpi_avg.values / 100)
    m["price_level"] = lvl / lvl[list(yrs).index(BASE_YEAR)]
    return m


def monthly_index_path(data: Data, macro: pd.DataFrame, sc: Scenario):
    """Возвращает dict year -> (СПК_ср, ФВ_ср, СПК_дек, ФВ_дек, СПК_янв, ФВ_янв)
    с учётом месяцев индексации."""
    t = data.spk.sort_values("effective_date")
    events = [(d.year, d.month, spk, fv) for d, spk, fv in zip(t.effective_date, t.point_value_rub, t.fixed_payment_rub)]
    last_hist_year = max(e[0] for e in events)
    spk, fv = events[-1][2], events[-1][3]
    plan = data.idx_plan.set_index("year")
    for y in range(last_hist_year + 1, END_YEAR + 1):
        if y in plan.index:
            stages = [(2, 1 + plan.loc[y, "stage1_pct_cpi"] / 100), (4, 1 + plan.loc[y, "stage2_pct"] / 100)]
        else:
            cpi_prev = macro.loc[y - 1, "cpi_dd"] / 100
            rw_prev = macro.loc[y - 1, "real_wage"] / 100
            if sc.indexation_rule == "cpi_only":
                stages = [(2, cpi_prev)]
            elif sc.indexation_rule == "wage":
                nominal_w = macro.loc[y - 1, "wage"] / macro.loc[y - 2, "wage"]
                stages = [(2, cpi_prev), (4, nominal_w / cpi_prev)]
            else:  # two_stage
                stages = [(2, cpi_prev), (4, 1 + sc.april_alpha * max(rw_prev, 0.0))]
        for mth, k in stages:
            spk, fv = spk * k, fv * k
            events.append((y, mth, spk, fv))
    # помесячная развёртка
    out = {}
    ev = sorted(events)
    cur_spk, cur_fv = ev[0][2], ev[0][3]
    i = 0
    for y in range(START_YEAR, END_YEAR + 1):
        ms, mf = [], []
        for mth in range(1, 13):
            while i < len(ev) and (ev[i][0], ev[i][1]) <= (y, mth):
                cur_spk, cur_fv = ev[i][2], ev[i][3]
                i += 1
            ms.append(cur_spk)
            mf.append(cur_fv)
        out[y] = (np.mean(ms), np.mean(mf), ms[-1], mf[-1], ms[0], mf[0])
    return out


# ----------------------------------------------------------------------------
# Демографическая проекция
# ----------------------------------------------------------------------------
def project_population(data: Data, sc: Scenario):
    """Население по годам; до 2025 — WPP как есть, дальше пересчёт с
    модифицированными смертностью/миграцией/рождаемостью."""
    P = {y: {s: data.P[y][s].copy() for s in SEXES} for y in range(START_YEAR, BASE_YEAR + 1)}
    S_used = {}
    for y in range(START_YEAR, END_YEAR + 1):
        k = 1.0
        if y > BASE_YEAR:
            k = 1 + (sc.mortality_mult_2050 - 1) * (y - BASE_YEAR) / (END_YEAR - BASE_YEAR)
        for s in SEXES:
            S = data.S[(y, s)].copy()
            q = 1 - S
            q[50:] *= k
            S_used[(y, s)] = np.clip(1 - q, 0, 1)
    for y in range(BASE_YEAR + 1, END_YEAR + 1):
        P[y] = {}
        for s in SEXES:
            surv = Data._shift(P[y - 1][s], S_used[(y - 1, s)])
            mig = data.Mig[(y, s)].copy()
            births = mig[0] * sc.fertility_mult   # возраст 0 из WPP (рождения×дожитие)
            mig[0] = 0.0
            P[y][s] = surv + mig * sc.migration_mult
            P[y][s][0] = births
            P[y][s] = np.maximum(P[y][s], 0)
    return P, S_used


# ----------------------------------------------------------------------------
# Охват пенсиями по старости
# ----------------------------------------------------------------------------
def coverage(year, sex, data, params, sc: Scenario):
    """Доля населения возраста x, получающая страховую пенсию по старости.
    params = (c_post_2021, тренд c_post в год (до 2025, далее заморожен), early_scale, длина плато досрочных, лет)."""
    c0, c_tr, e_scale, plateau = params
    tau = 2.0
    c_post = c0 + c_tr * (min(year, BASE_YEAR) - 2021)
    birth = year - AGES
    RA = data.ret_age(birth, sex)
    if sc.raise_retirement_age_2035:
        # дополнительное повышение для достигающих возраста с 2035 (линейно 0.5 года/год)
        reach = birth + RA
        extra = np.clip((reach - 2035) * 0.5, 0, sc.raise_retirement_age_2035)
        RA = RA + extra
    post = np.clip(AGES + 1 - RA, 0, 1)
    d = RA - (AGES + 0.5)                     # лет до общего пенсионного возраста
    sex_k = 1.0 if sex == "M" else 1.15        # у женщин больше досрочных (педагоги, медики, многодетные)
    # плато 5 лет (Север, списки №1/2, педстаж, длинный стаж), затем затухание
    early = np.where(d > 0, e_scale * sex_k * np.where(d <= plateau, 1.0, np.exp(-(d - plateau) / tau)), 0.0)
    cov = c_post * post + (1 - post) * np.minimum(early, c_post)
    early_part = (1 - post) * np.minimum(early, c_post)
    return cov, early_part


def oldage_counts(P, year, data, params, sc):
    tot, early, by_age = 0.0, 0.0, {}
    for s in SEXES:
        cov, e = coverage(year, s, data, params, sc)
        by_age[s] = P[year][s] * cov
        tot += by_age[s].sum()
        early += (P[year][s] * e).sum()
    return tot, early, by_age


def calibrate_coverage(data: Data, P, sc: Scenario):
    obs_oa = data.pens_jan1("pensioners_oldage_insurance")
    obs_early = data.pens_jan1("pensioners_early_retirement")
    yrs = [y for y in range(2018, 2026) if y in obs_oa.index]

    def resid(x):
        params = tuple(x)
        r = []
        for y in yrs:
            # 1 января y ≈ среднее середин y-1 и y
            t0, e0, _ = oldage_counts(P, y - 1, data, params, sc)
            t1, e1, _ = oldage_counts(P, y, data, params, sc)
            r.append(((t0 + t1) / 2 - obs_oa[y]) / obs_oa[y])
            if y in obs_early.index:
                r.append(0.5 * ((e0 + e1) / 2 - obs_early[y]) / obs_early[y])
        return np.array(r)

    sol = least_squares(resid, x0=[0.9, -0.005, 0.15, 4.0], bounds=([0.6, -0.03, 0.0, 0.5], [1.05, 0.03, 0.8, 8.0]))
    return tuple(sol.x), resid(sol.x)


# ----------------------------------------------------------------------------
# Основная модель
# ----------------------------------------------------------------------------
class PensionModel:
    # соотношения средних пенсий к пенсии по старости (СФР, 01.01.2023)
    RATIO_DIS = 12477.57 / 20678.42
    RATIO_SURV = 13695.73 / 20678.42
    WORK_RECALC_POINTS = 2.0   # среднее увеличение ИПК работающего за год (≈265 ₽ / СПК 133 в 2024)
    IPK_NEW_GROWTH = 0.0       # рост ИПК новых пенсионеров в год (баллы привязаны к з/п → ≈0)
    LAMBDA_YEARS = (2023, 2025)  # окно калибровки λ (структура пенсий после 2022, надбавка 80+ с 2025)
    FSD_ELASTICITY = 3.0       # чувствительность ФСД к отношению ПМП/пенсия

    def __init__(self, data: Data, sc: Scenario | None = None):
        self.d = data
        self.sc = sc or Scenario()
        self.calib = {}

    # ---------------------------------------------------------------------
    def run(self, calibrate: bool = True, calib_from: dict | None = None) -> pd.DataFrame:
        d, sc = self.d, self.sc
        macro = build_macro(d, sc)
        idx = monthly_index_path(d, macro, sc)
        P, S = project_population(d, sc)

        # 1) охват
        if calib_from is not None:
            self.calib = copy.deepcopy(calib_from)
        if calibrate and calib_from is None:
            params, res = calibrate_coverage(d, P, Scenario())
            self.calib["coverage"] = params
            self.calib["coverage_resid"] = res.tolist()
        params = self.calib["coverage"]

        years = list(range(START_YEAR, END_YEAR + 1))
        rows = []
        cnt = {}
        for y in years:
            tot, early, by_age = oldage_counts(P, y, d, params, sc)
            n80 = sum(by_age[s][80:].sum() for s in SEXES)
            # выбывание по смертности (из середины y в середину y+1)
            surv = sum((by_age[s] * S[(y, s)]).sum() for s in SEXES)
            ra_m = d.ret_age(np.array([y - 60]), "M")[0]
            ra_f = d.ret_age(np.array([y - 55]), "F")[0]
            pop = {s: P[y][s] for s in SEXES}
            # «трудоспособные» (20 .. общий пенсионный возраст когорты)
            wa = 0.0
            for s in SEXES:
                RA = d.ret_age(y - AGES, s)
                mask = (AGES >= 20) & (AGES + 0.5 < RA)
                wa += pop[s][mask].sum()
                mask_old = (AGES >= 55) & (AGES + 0.5 < RA)
                wa += (sc.employment_mult_older - 1) * pop[s][mask_old].sum() * 0.6
            cnt[y] = dict(oa=tot, early=early, s80=n80 / tot, surv=surv,
                          pop_total=sum(pop[s].sum() for s in SEXES),
                          pop_0_22=sum(pop[s][:23].sum() for s in SEXES),
                          pop_wa=wa,
                          pop_65p=sum(pop[s][65:].sum() for s in SEXES))

        # 2) прочие виды пенсий: доля от референтной группы
        obs = {k: d.pens_jan1(k) for k in ["pensioners_disability_insurance", "pensioners_survivor_insurance",
                                           "pensioners_social", "pensioners_working_oldage",
                                           "pensioners_oldage_insurance", "avg_pension_oldage",
                                           "avg_pension_social", "pensioners_total"]}
        def rate(key, ref):
            r = {y: obs[key][y] / cnt[y][ref] for y in obs[key].index if y in cnt}
            last = r[max(r)]
            return {y: r.get(y, last if y > max(r) else r[min(r)]) for y in years}
        r_dis = rate("pensioners_disability_insurance", "pop_wa")
        r_surv = rate("pensioners_survivor_insurance", "pop_0_22")
        r_soc = rate("pensioners_social", "pop_total")
        w_share_obs = (obs["pensioners_working_oldage"] / obs["pensioners_oldage_insurance"]).dropna()
        w_last = w_share_obs.iloc[-1]
        w_long = sc.working_share_long if sc.working_share_long is not None else w_last

        # 3) ИПК: наблюдаемый эффективный ИПК на 1 января
        obs_p = obs["avg_pension_oldage"]
        ipk_obs = {}
        for y in obs_p.index:
            if y - 1 < START_YEAR:
                continue
            # Росстат «на 1 января» включает январскую индексацию этого года
            spk_jan, fv_jan = idx[y][4], idx[y][5]
            s80 = 0.5 * (cnt[y - 1]["s80"] + cnt[y]["s80"])
            ipk_obs[y] = (obs_p[y] - fv_jan * (1 + s80)) / spk_jan

        g_new = self.IPK_NEW_GROWTH

        def simulate_ipk(theta, until=END_YEAR):
            ipk_new0 = theta[0]
            g = g_new
            first = min(k for k in ipk_obs if k >= 2018)
            ipk = {first: ipk_obs[first]}
            for y in range(first, until):
                ws = w_share_obs.get(y, w_last if y <= BASE_YEAR else w_long)
                surv, n_next = cnt[y]["surv"], cnt[y + 1]["oa"]
                entr = max(n_next - surv, 0.0)
                ipk_new = ipk_new0 * (1 + g) ** (y - 2018)
                share_s = surv / (surv + entr)
                nxt = (ipk[y] * surv + ipk_new * entr) / (surv + entr) + ws * self.WORK_RECALC_POINTS * share_s
                # заморозка индексации работающим 2016–2024: их «эффективный» ИПК
                # (в единицах полного СПК) снижается на темп индексации
                if 2016 <= y + 1 <= 2024:
                    k = idx[y + 1][4] / idx[y][4]
                    nxt -= ws * share_s * ipk[y] * (1 - 1 / k)
                ipk[y + 1] = nxt
            return ipk

        if calib_from is None:
            fit_years = [y for y in ipk_obs if 2019 <= y <= 2025]
            def res_ipk(theta):
                sim = simulate_ipk(theta, until=2025)
                return np.array([(sim[y] - ipk_obs[y]) / ipk_obs[y] for y in fit_years])
            sol = least_squares(res_ipk, x0=[100.0], bounds=([30], [250]))
            self.calib["ipk_theta"] = sol.x.tolist()
            self.calib["ipk_obs"] = ipk_obs
        ipk = simulate_ipk(self.calib["ipk_theta"])
        first_ipk_year = min(ipk)

        # 4) годовые величины
        soc_avg = {}
        obs_soc = obs["avg_pension_social"]
        for y in years:
            if y in obs_soc.index:
                soc_avg[y] = obs_soc[y]
            elif y <= 2027:
                # среднегодовой рост: 2026 — хвост апр.2025 (+14.75%) и апр.2026 (+6.8%); 2027 — +6.8% с апреля
                soc_avg[y] = soc_avg[y - 1] * {2026: 1.1475 ** 0.25 * 1.068 ** 0.75, 2027: 1.068}.get(y, 1.068)
            else:
                soc_avg[y] = soc_avg[y - 1] * (macro.loc[y, "cpi_avg"] / 100) * (1 + sc.soc_real_extra / 100)

        for y in years:
            c = cnt[y]
            spk_avg, fv_avg = idx[y][0], idx[y][1]
            if y < first_ipk_year:
                ipk_mid = np.nan
            else:
                ipk_mid = 0.5 * (ipk.get(y, np.nan) + ipk.get(y + 1, ipk.get(y, np.nan)))
            p_oa = spk_avg * ipk_mid + fv_avg * (1 + c["s80"])
            n_dis = r_dis[y] * c["pop_wa"]
            n_surv = r_surv[y] * c["pop_0_22"]
            n_soc = r_soc[y] * c["pop_total"]
            raw = 12 * (c["oa"] * p_oa + n_dis * p_oa * self.RATIO_DIS + n_surv * p_oa * self.RATIO_SURV) / 1e6  # млрд ₽ (тыс. чел × ₽ = тыс. ₽)
            rows.append(dict(year=y, pop_total=c["pop_total"], pop_65p=c["pop_65p"], pop_working_age=c["pop_wa"],
                             n_oldage=c["oa"], n_early=c["early"], n_disability=n_dis, n_survivor=n_surv,
                             n_social=n_soc, n_total=c["oa"] + n_dis + n_surv + n_soc,
                             share80=c["s80"], ipk=ipk_mid, spk_avg=spk_avg, fv_avg=fv_avg,
                             avg_oldage_pension=p_oa, avg_social_pension=soc_avg[y],
                             ins_spend_raw=raw))
        out = pd.DataFrame(rows).set_index("year").join(macro)

        # 5) калибровка λ на факте расходов на страховые пенсии
        fact = d.budget_fact("spend_insurance_pensions")
        if calib_from is None:
            lam = (fact / out.ins_spend_raw).dropna()
            lam = lam[lam.index >= 2019]
            self.calib["lambda_by_year"] = lam.to_dict()
            self.calib["lambda"] = float(lam.loc[self.LAMBDA_YEARS[0]:self.LAMBDA_YEARS[1]].mean())
        lam = self.calib["lambda"]
        out["ins_spend"] = out.ins_spend_raw * lam
        out["ins_spend_fact"] = fact.reindex(out.index)
        out["ins_spend_plan"] = d.budget_plan("spend_insurance_pensions").reindex(out.index)

        # 6) государственные (социальные) пенсии и ФСД
        gov = d.budget_fact("spend_state_pensions_166fz")
        fsd = d.budget_fact("spend_federal_social_topup_FSD")
        g0, f0 = gov[BASE_YEAR], fsd[BASE_YEAR]
        n0, s0 = out.loc[BASE_YEAR, "n_social"], out.loc[BASE_YEAR, "avg_social_pension"]
        out["gov_pens_spend"] = g0 * out.n_social / n0 * out.avg_social_pension / s0
        out.loc[out.index <= BASE_YEAR, "gov_pens_spend"] = gov.reindex(out.index[out.index <= BASE_YEAR])
        # ФСД: ∝ числу пенсионеров × ПМП-разрыв; ПМП растёт как соц. пенсия, пенсии — как p_oa
        pmp_idx = out.avg_social_pension / s0
        rel = pmp_idx / (out.avg_oldage_pension / out.loc[BASE_YEAR, "avg_oldage_pension"])
        gap_idx = pmp_idx * np.maximum(0.2, 1 + self.FSD_ELASTICITY * (rel - 1))
        out["fsd_spend"] = f0 * out.n_total / out.loc[BASE_YEAR, "n_total"] * gap_idx
        out.loc[out.index <= BASE_YEAR, "fsd_spend"] = fsd.reindex(out.index[out.index <= BASE_YEAR])
        out.loc[out.index > BASE_YEAR, "ins_spend_fact"] = np.nan

        # 7) доходы ОПС
        contrib = d.budget_fact("contrib_ops_for_insurance_pension")
        fot = d.budget_fact("wage_fund_contrib_base")
        tau = float((contrib[fot.index] / fot).mean()) * sc.contrib_rate_mult
        self.calib["tau_eff"] = tau / sc.contrib_rate_mult
        emp_idx = out.pop_working_age / out.loc[BASE_YEAR, "pop_working_age"]
        wage_idx = out.wage / out.loc[BASE_YEAR, "wage"]
        out["wage_fund"] = fot[BASE_YEAR] * emp_idx * wage_idx
        out.loc[fot.index, "wage_fund"] = fot
        out["contributions"] = out.wage_fund * tau
        out.loc[out.index <= BASE_YEAR, "contributions"] = contrib.reindex(out.index[out.index <= BASE_YEAR])
        # трансферты ФБ на страховые пенсии: валоризация + нестраховые периоды (убывают с «советскими» когортами)
        val = d.budget_fact("fed_valorization").loc[2023:2024].mean()
        nonins = d.budget_fact("fed_noninsurance_periods").loc[2023:2024].mean()
        soviet = np.array([self._soviet_share(y) for y in out.index])
        soviet = soviet / soviet[list(out.index).index(BASE_YEAR)]
        price = out.spk_avg / out.loc[BASE_YEAR, "spk_avg"]
        out["fed_valorization"] = val * soviet * price
        out["fed_noninsurance"] = nonins * price * out.n_oldage / out.loc[BASE_YEAR, "n_oldage"]
        out["ops_gap"] = out.ins_spend - out.contributions - out.fed_valorization - out.fed_noninsurance
        # итог: все пенсионные расходы СФР
        out["pension_spend_total"] = out.ins_spend + out.gov_pens_spend + out.fsd_spend
        # производные
        out["pension_spend_real2025"] = out.pension_spend_total / out.price_level
        out["pension_spend_pct_gdp"] = 100 * out.pension_spend_total / out.gdp
        out["ins_spend_pct_gdp"] = 100 * out.ins_spend / out.gdp
        out["replacement_rate"] = out.avg_oldage_pension / out.wage
        out["support_ratio"] = out.pop_working_age / out.n_total
        out["scenario"] = sc.name
        return out

    def _soviet_share(self, y):
        """Число пенсионеров по старости, родившихся до 1975 (стаж до 1991 → валоризация)."""
        P, _ = self._pop_cache if hasattr(self, "_pop_cache") else (None, None)
        if P is None:
            self._pop_cache = project_population(self.d, self.sc)
            P = self._pop_cache[0]
        _, _, by_age = oldage_counts(P, y, self.d, self.calib["coverage"], self.sc)
        birth = y - AGES
        w = np.clip((1991 - (birth + 20)) / 30, 0, 1)   # доля «советского» стажа в 30-летней карьере
        return sum((by_age[s] * w).sum() for s in SEXES)


# ----------------------------------------------------------------------------
# Сценарии и Монте-Карло
# ----------------------------------------------------------------------------
SCENARIOS = {
    "base": Scenario(),
    "cpi_only": Scenario(name="cpi_only", indexation_rule="cpi_only"),
    "wage_indexation": Scenario(name="wage_indexation", indexation_rule="wage"),
    "low_mortality": Scenario(name="low_mortality", mortality_mult_2050=0.75),
    "high_migration": Scenario(name="high_migration", migration_mult=1.5),
    "stagnation": Scenario(name="stagnation", real_wage_long=0.5, real_gdp_long=0.5, cpi_long=5.0),
    "ra_plus2_from2035": Scenario(name="ra_plus2_from2035", raise_retirement_age_2035=2.0),
}


def monte_carlo(data, calib, n=300, seed=42, base: Scenario | None = None):
    rng = np.random.default_rng(seed)
    base = base or Scenario()
    res = []
    for i in range(n):
        sc = copy.deepcopy(base)
        sc.name = f"mc_{i}"
        # AR(1)-шоки ИПЦ и реальной зарплаты после 2029
        e_c, e_w = 0.0, 0.0
        for y in range(2030, END_YEAR + 1):
            e_c = 0.6 * e_c + rng.normal(0, 1.5)
            e_w = 0.5 * e_w + rng.normal(0, 1.5)
            sc.cpi_shocks[y], sc.rw_shocks[y] = e_c, e_w
        sc.mortality_mult_2050 = float(np.exp(rng.normal(np.log(0.9), 0.12)))
        sc.migration_mult = float(max(0.0, rng.normal(1.0, 0.4)))
        sc.fertility_mult = float(rng.normal(1.0, 0.08))
        m = PensionModel(data, sc)
        o = m.run(calibrate=False, calib_from=calib)
        res.append(o[["pension_spend_total", "pension_spend_real2025", "pension_spend_pct_gdp",
                      "ins_spend", "ops_gap", "n_total", "replacement_rate"]].assign(draw=i))
    df = pd.concat(res).reset_index()
    q = df.groupby("year").quantile([0.05, 0.5, 0.95]).drop(columns="draw")
    return df, q


# ----------------------------------------------------------------------------
# Отчёт
# ----------------------------------------------------------------------------
def plot(results: dict, mcq, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    ax = ax.ravel()
    base = results["base"]
    for name, o in results.items():
        lw = 2.5 if name == "base" else 1.2
        ax[0].plot(o.index, o.ins_spend, lw=lw, label=name)
        ax[1].plot(o.index, o.pension_spend_real2025, lw=lw, label=name)
        ax[2].plot(o.index, o.pension_spend_pct_gdp, lw=lw, label=name)
        ax[3].plot(o.index, o.n_total / 1e3, lw=lw, label=name)
        ax[4].plot(o.index, 100 * o.replacement_rate, lw=lw, label=name)
        ax[5].plot(o.index, o.ops_gap, lw=lw, label=name)
    ax[0].scatter(base.index, base.ins_spend_fact, c="k", s=18, zorder=5, label="факт")
    ax[0].scatter(base.index, base.ins_spend_plan, c="r", marker="x", s=30, zorder=5, label="план СФР")
    if mcq is not None:
        lo, hi = mcq.xs(0.05, level=1), mcq.xs(0.95, level=1)
        ax[1].fill_between(lo.index, lo.pension_spend_real2025, hi.pension_spend_real2025, color="grey", alpha=0.25, label="МК 90%")
        ax[2].fill_between(lo.index, lo.pension_spend_pct_gdp, hi.pension_spend_pct_gdp, color="grey", alpha=0.25, label="МК 90%")
    titles = ["Расходы на страховые пенсии, млрд ₽", "Все пенсионные расходы СФР, млрд ₽ 2025 г.",
              "Пенсионные расходы, % ВВП", "Пенсионеры (СФР), млн", "Коэф. замещения: пенсия по старости / средняя з/п, %",
              "Разрыв ОПС, млрд ₽\n(страх. пенсии − взносы − валоризация)"]
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=10)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "forecast_scenarios.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "output"))
    ap.add_argument("--mc", type=int, default=0, help="число прогонов Монте-Карло")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = Data(args.data)
    base = PensionModel(data, SCENARIOS["base"])
    res = {"base": base.run()}
    calib = base.calib
    for k, sc in SCENARIOS.items():
        if k != "base":
            res[k] = PensionModel(data, sc).run(calibrate=False, calib_from=calib)

    allres = pd.concat(res.values()).reset_index()
    allres.to_csv(out_dir / "forecast_all_scenarios.csv", index=False)
    b = res["base"]
    cols = ["n_total", "n_oldage", "avg_oldage_pension", "ins_spend", "ins_spend_fact", "ins_spend_plan",
            "gov_pens_spend", "fsd_spend", "pension_spend_total", "pension_spend_real2025",
            "pension_spend_pct_gdp", "contributions", "ops_gap", "replacement_rate"]
    b[cols].round(3).to_csv(out_dir / "forecast_base.csv")

    mcq = None
    if args.mc:
        draws, mcq = monte_carlo(data, calib, n=args.mc)
        mcq.to_csv(out_dir / "monte_carlo_quantiles.csv")

    # калибровочный отчёт
    rep = {k: (v if not isinstance(v, dict) else {int(a): float(c) for a, c in v.items()})
           for k, v in calib.items()}
    rep["coverage"] = list(calib["coverage"])
    with open(out_dir / "calibration.json", "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=float)
    plot(res, mcq, out_dir)

    # проверка подгонки на истории
    oa = data.pens_jan1("pensioners_oldage_insurance")
    fit = pd.DataFrame({
        "oldage_jan1_fact": oa,
        "oldage_jan1_model": (b.n_oldage.shift(1) + b.n_oldage) / 2,
        "early_jan1_fact": data.pens_jan1("pensioners_early_retirement"),
        "early_jan1_model": (b.n_early.shift(1) + b.n_early) / 2,
        "ipk_jan1_fact": pd.Series(calib["ipk_obs"]),
        "ins_spend_fact": b.ins_spend_fact,
        "ins_spend_model": b.ins_spend,
        "ins_spend_plan": b.ins_spend_plan,
    }).loc[2016:2029]
    fit["ins_spend_err_%"] = 100 * (fit.ins_spend_model / fit.ins_spend_fact.fillna(fit.ins_spend_plan) - 1)
    fit.round(1).to_csv(out_dir / "calibration_fit.csv")
    print("\nПодгонка на истории и сравнение с планом СФР:")
    print(fit.round(1).to_string())

    print("Калибровка:", json.dumps({k: rep[k] for k in ["coverage", "ipk_theta", "lambda", "tau_eff", "lambda_by_year"]}, default=float))
    print("ИПК факт:", {k: round(v, 1) for k, v in calib["ipk_obs"].items()})
    print("\nБазовый сценарий (млрд ₽):")
    print(b.loc[[2015, 2020, 2024, 2025, 2027, 2028, 2029, 2030, 2035, 2040, 2045, 2050], cols].round(1).to_string())
    print("\n2050 по сценариям:")
    print(allres[allres.year == 2050].set_index("scenario")[["n_total", "avg_oldage_pension", "pension_spend_total",
          "pension_spend_real2025", "pension_spend_pct_gdp", "replacement_rate"]].round(3).to_string())
    if mcq is not None:
        print("\nМонте-Карло, 2050 (5% / 50% / 95%):")
        print(mcq.xs(2050, level=0)[["pension_spend_real2025", "pension_spend_pct_gdp", "ops_gap"]].round(2).to_string())


if __name__ == "__main__":
    main()
