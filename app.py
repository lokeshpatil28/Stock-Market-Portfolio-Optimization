"""Streamlit dashboard for the portfolio_opt package.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (requirements.txt is the runtime set;
               cvxpy is NOT required at runtime -- only the validation tests use it).

The app is self-contained given the committed price CSVs in data/prices/: it
recomputes mu/Sigma from those at runtime, so no network is needed except the
optional live Nifty-50 benchmark download in the Backtest tab (handled
gracefully if unavailable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from portfolio_opt import backtest as B
from portfolio_opt import data as D
from portfolio_opt import models as M

st.set_page_config(page_title="Portfolio QP Optimizer", layout="wide",
                   initial_sidebar_state="expanded")

PCT = "{:.2f}%"           # consistent 2-decimal percentage formatting
RA_MIN, RA_MAX = 0.1, 100.0


def fmt_pct(x):
    return PCT.format(x) if pd.notna(x) else "-"


# ---------------------------------------------------------------------------
# Cached data layer
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_config_maps():
    cfg = D.load_config()
    tickers = D.get_tickers(cfg)
    return {
        "tickers": tickers,
        "sector_map": D.get_sector_map(cfg),
        "cap_tier_map": D.get_cap_tier_map(cfg),
        "risk_free_rate": float(cfg.get("risk_free_rate", 0.06)),
        "min_observations": int(cfg.get("min_observations", 0)),
        "rebalance_frequency": cfg.get("rebalance_frequency", "monthly"),
    }


@st.cache_data(show_spinner=False)
def load_all_prices(min_observations):
    cfg_tickers = load_config_maps()["tickers"]
    return D.load_combined_prices(cfg_tickers, min_observations=min_observations)


@st.cache_data(show_spinner=False)
def compute_moments(tickers, method, min_observations):
    """mu, Sigma, shrinkage intensity and the returns frame for a ticker subset."""
    prices = load_all_prices(min_observations)[list(tickers)]
    returns = D.compute_returns(prices)
    mu, Sigma, intensity = D.estimate_mu_sigma(returns, method=method)
    return mu, Sigma, intensity, returns


@st.cache_data(show_spinner=True)
def compute_frontier(tickers, method, max_weight, rf, sector_caps_items,
                     min_observations, n_points=40):
    """Sweep risk aversion to trace the efficient frontier (vol, ret, sharpe)."""
    mu, Sigma, _, _ = compute_moments(tickers, method, min_observations)
    caps = dict(sector_caps_items) or None
    smap = [load_config_maps()["sector_map"][t] for t in tickers]
    rows = []
    for lam in np.logspace(-1, 2.7, n_points):
        r = M.mean_variance_utility(mu, Sigma, risk_aversion=lam,
                                    max_weight=max_weight, sector_caps=caps,
                                    sector_map=smap, tickers=list(tickers),
                                    risk_free_rate=rf)
        if r["converged"]:
            rows.append((r["volatility"], r["expected_return"], r["sharpe_ratio"]))
    return pd.DataFrame(rows, columns=["vol", "ret", "sharpe"])


@st.cache_data(show_spinner=True)
def run_backtest_cached(tickers, model_name, model_kwargs_items, lookback_years,
                        cost_bps, method, rebalance_frequency, rf, min_observations):
    """Walk-forward backtest for one strategy + benchmarks (cache-friendly)."""
    prices = load_all_prices(min_observations)[list(tickers)]
    model_fn = MODEL_FNS[model_name]
    model_kwargs = dict(model_kwargs_items)
    model_kwargs["tickers"] = list(tickers)
    # risk_parity takes neither sector_map nor sector_caps; the others do.
    if model_name != "Risk Parity":
        # sector_map must align to the selected universe order.
        model_kwargs["sector_map"] = [load_config_maps()["sector_map"][t]
                                      for t in tickers]
        if model_kwargs.get("sector_caps_items"):
            model_kwargs["sector_caps"] = dict(model_kwargs["sector_caps_items"])
    model_kwargs.pop("sector_caps_items", None)

    bt = B.run_backtest(prices, model_fn, model_kwargs,
                        rebalance_frequency=rebalance_frequency,
                        lookback_years=lookback_years,
                        transaction_cost_bps=cost_bps, estimation_method=method)
    if bt.get("status") == "insufficient_history":
        # Nothing to score / benchmark — surface the flag to the tab.
        return {"status": "insufficient_history", "bt": bt}
    perf = B.compute_performance_metrics(bt["equity_curve"], rf,
                                         turnover=bt["turnover"])
    benches = B.compute_benchmarks(prices, bt["first_rebalance"],
                                   end_date=prices.index.max(),
                                   transaction_cost_bps=cost_bps)
    bench_perf = {name: B.compute_performance_metrics(curve, rf)
                  for name, curve in benches.items()}
    return {"status": "ok", "bt": bt, "perf": perf, "benches": benches,
            "bench_perf": bench_perf}


@st.cache_data(show_spinner=True)
def compare_all_cached(tickers, method, max_weight, target_return, risk_aversion,
                       rf, sector_caps_items, min_observations):
    mu, Sigma, _, _ = compute_moments(tickers, method, min_observations)
    caps = dict(sector_caps_items) or None
    smap = [load_config_maps()["sector_map"][t] for t in tickers]
    out = M.compare_all_models(mu, Sigma, list(tickers), target_return=target_return,
                               risk_aversion=risk_aversion, risk_free_rate=rf,
                               max_weight=max_weight, sector_map=smap,
                               sector_caps=caps)
    # Return only serializable pieces.
    return {"summary": out["summary"], "weights": out["weights"],
            "results": {k: {"weights_pct": v["weights_pct"],
                            "risk_contribution_pct": v["risk_contribution_pct"]}
                        for k, v in out["results"].items()}}


# Model-name -> callable (module-level so the cache function can reach it).
MODEL_FNS = {
    "Mean-Variance (utility)": M.mean_variance_utility,
    "Max Sharpe": M.max_sharpe,
    "Min Variance (target return)": M.min_variance_target_return,
    "Risk Parity": M.risk_parity,
}

# Validation results from tests/test_solver_validation.py (cvxpy/OSQP vs ours).
VALIDATION_ROWS = pd.DataFrame([
    {"n assets": 2,  "constraints": "budget+return+box",       "max |Δweight|": 6.7e-16, "obj rel.diff": 2.2e-15},
    {"n assets": 5,  "constraints": "budget+return+box (1 bind)", "max |Δweight|": 1.6e-15, "obj rel.diff": 4.3e-15},
    {"n assets": 10, "constraints": "budget+return+box (cap binds)", "max |Δweight|": 9.7e-17, "obj rel.diff": 1.6e-15},
    {"n assets": 25, "constraints": "budget+return+box (7 bind)", "max |Δweight|": 1.2e-16, "obj rel.diff": 1.5e-15},
])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
maps = load_config_maps()
sector_map = maps["sector_map"]
cap_tier_map = maps["cap_tier_map"]
universe = maps["tickers"]
min_obs = maps["min_observations"]

st.sidebar.title("⚙️ Configuration")

# --- Universe (labeled by sector) ---
label_for = {t: f"{t.replace('.NS','')} ({sector_map[t]})" for t in universe}
default_sel = universe[:12]
selected_labels = st.sidebar.multiselect(
    "Assets (universe of 25, labeled by sector)",
    options=[label_for[t] for t in universe],
    default=[label_for[t] for t in default_sel],
    help="Pick the investable universe. Each option shows TICKER (Sector).",
)
label_to_ticker = {v: k for k, v in label_for.items()}
selected = [label_to_ticker[l] for l in selected_labels]

if len(selected) < 2:
    st.sidebar.error("Select at least 2 assets.")
    st.title("📈 Portfolio QP Optimizer")
    st.info("Choose at least 2 assets in the sidebar to begin.")
    st.stop()

# --- Covariance method (live shrinkage intensity) ---
cov_method = st.sidebar.radio("Covariance estimator", ["shrinkage", "sample"],
                              help="Ledoit-Wolf shrinkage vs plain sample covariance.")
mu, Sigma, intensity, returns_df = compute_moments(tuple(selected), cov_method, min_obs)
if cov_method == "shrinkage":
    st.sidebar.metric("Ledoit-Wolf shrinkage intensity", f"{intensity:.4f}",
                      help="0 = sample covariance, 1 = fully shrunk to the "
                           "structured target.")

# --- Model ---
model_name = st.sidebar.radio("Model", list(MODEL_FNS.keys()))
is_utility = model_name == "Mean-Variance (utility)"
is_sharpe = model_name == "Max Sharpe"
is_target = model_name == "Min Variance (target return)"

# --- Sliders ---
risk_aversion = st.sidebar.slider(
    "Risk aversion λ", RA_MIN, RA_MAX, 10.0, 0.1,
    disabled=not (is_utility or is_sharpe),
    help="Trade-off in max(μᵀx − λ·xᵀΣx). Active for utility; Max-Sharpe sweeps "
         "λ internally.")

mu_lo, mu_hi = float(np.min(mu)), float(np.max(mu))
default_tr = float(np.clip(np.mean(mu), mu_lo, mu_hi))
target_return = st.sidebar.slider(
    "Target return (annualized)", round(mu_lo, 3), round(mu_hi, 3),
    round(default_tr, 3), 0.005, disabled=not is_target,
    help="Required μᵀx for the min-variance model. Bounds are the min/max "
         "asset returns in the selection.")

max_weight = st.sidebar.slider("Max weight per asset", 0.05, 1.0, 0.25, 0.05,
                               help="Upper bound on each x_i.")

turnover_limit = st.sidebar.slider(
    "Turnover limit (L1 vs equal-weight)", 0.0, 2.0, 2.0, 0.1,
    disabled=not is_utility,
    help="Bound Σ|x_i − prev_i| (prev = equal weight). 2.0 = effectively off. "
         "Only applied to the utility model.")

rf = st.sidebar.number_input("Risk-free rate (annualized)", 0.0, 0.20,
                             value=maps["risk_free_rate"], step=0.005, format="%.3f")

# --- Advanced: sector caps ---
sector_caps = None
with st.sidebar.expander("Advanced constraints (sector caps)"):
    use_caps = st.checkbox("Enable per-sector caps")
    if use_caps:
        sectors_present = sorted({sector_map[t] for t in selected})
        caps = {}
        for s in sectors_present:
            caps[s] = st.slider(f"{s} cap", 0.0, 1.0, 1.0, 0.05, key=f"cap_{s}")
        # Keep only binding (<1) caps.
        sector_caps = {s: c for s, c in caps.items() if c < 1.0} or None

sector_caps_items = tuple(sorted(sector_caps.items())) if sector_caps else tuple()
smap_sel = [sector_map[t] for t in selected]


# ---------------------------------------------------------------------------
# Solve the currently selected portfolio
# ---------------------------------------------------------------------------
def solve_current():
    common = dict(max_weight=max_weight, sector_caps=sector_caps,
                  sector_map=smap_sel, tickers=selected, risk_free_rate=rf)
    if is_utility:
        tl = turnover_limit if turnover_limit < 2.0 else None
        prev = np.full(len(selected), 1.0 / len(selected)) if tl is not None else None
        return M.mean_variance_utility(mu, Sigma, risk_aversion=risk_aversion,
                                       turnover_limit=tl, prev_weights=prev, **common)
    if is_sharpe:
        return M.max_sharpe(mu, Sigma, **common)
    if is_target:
        return M.min_variance_target_return(mu, Sigma, target_return=target_return,
                                            **common)
    return M.risk_parity(mu, Sigma, max_weight=max_weight, tickers=selected,
                         risk_free_rate=rf)


result = solve_current()

st.title("📈 Portfolio QP Optimizer")
st.caption(f"{model_name}  ·  {len(selected)} assets  ·  {cov_method} covariance  "
           f"·  max weight {fmt_pct(max_weight*100)}")

if not result["converged"]:
    st.error(f"Solver did not converge / problem infeasible for these settings "
             f"({result.get('solver_status', 'see constraints')}). Try relaxing "
             f"the target return, max weight, or sector caps.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Efficient Frontier", "Allocation", "Backtest", "Solver Validation",
     "Model Comparison"])


# ---------------------------------------------------------------------------
# TAB 1: Efficient Frontier
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Efficient frontier (risk-aversion sweep)")
    overlay = st.checkbox("Overlay sample vs shrinkage frontiers", value=False)

    fig = go.Figure()

    def add_frontier(method, color, name):
        fr = compute_frontier(tuple(selected), method, max_weight, rf,
                              sector_caps_items, min_obs)
        if fr.empty:
            return None
        fig.add_trace(go.Scatter(x=fr["vol"] * 100, y=fr["ret"] * 100,
                                 mode="lines+markers", name=name,
                                 line=dict(color=color, width=2),
                                 marker=dict(size=4)))
        return fr

    if overlay:
        add_frontier("sample", "#888", "Frontier (sample)")
        fr_main = add_frontier("shrinkage", "#1f77b4", "Frontier (shrinkage)")
    else:
        fr_main = add_frontier(cov_method, "#1f77b4", f"Frontier ({cov_method})")

    # Explicit min-variance and max-Sharpe points (current cov method).
    ms = M.max_sharpe(mu, Sigma, risk_free_rate=rf, max_weight=max_weight,
                      sector_caps=sector_caps, sector_map=smap_sel, tickers=selected)
    gmv = M.mean_variance_utility(mu, Sigma, risk_aversion=RA_MAX,
                                  max_weight=max_weight, sector_caps=sector_caps,
                                  sector_map=smap_sel, tickers=selected,
                                  risk_free_rate=rf)
    if gmv["converged"]:
        fig.add_trace(go.Scatter(x=[gmv["volatility"] * 100], y=[gmv["expected_return"] * 100],
                                 mode="markers+text", name="Min variance",
                                 marker=dict(color="green", size=12, symbol="diamond"),
                                 text=["Min-Var"], textposition="bottom center"))
    if ms["converged"]:
        fig.add_trace(go.Scatter(x=[ms["volatility"] * 100], y=[ms["expected_return"] * 100],
                                 mode="markers+text", name="Max Sharpe",
                                 marker=dict(color="orange", size=12, symbol="circle"),
                                 text=["Max-Sharpe"], textposition="top center"))
        # Capital Market Line: from (0, rf) through the tangency portfolio.
        slope = (ms["expected_return"] - rf) / ms["volatility"]
        xmax = (fr_main["vol"].max() * 100 * 1.1) if fr_main is not None else ms["volatility"] * 110
        xs = np.linspace(0, xmax / 100, 50)
        fig.add_trace(go.Scatter(x=xs * 100, y=(rf + slope * xs) * 100,
                                 mode="lines", name="Capital Market Line",
                                 line=dict(color="orange", dash="dash", width=1.5)))

    # The currently selected portfolio as a star.
    if result["converged"]:
        fig.add_trace(go.Scatter(x=[result["volatility"] * 100], y=[result["expected_return"] * 100],
                                 mode="markers", name="Selected portfolio",
                                 marker=dict(color="red", size=18, symbol="star")))

    fig.update_layout(xaxis_title="Volatility (annualized %)",
                      yaxis_title="Expected return (annualized %)",
                      height=520, legend=dict(orientation="h", y=-0.2),
                      margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Star = your current portfolio. The CML is the tangent from the "
               "risk-free rate through the max-Sharpe (tangency) portfolio.")


# ---------------------------------------------------------------------------
# TAB 2: Allocation
# ---------------------------------------------------------------------------
with tab2:
    if not result["converged"]:
        st.warning("No allocation to show — solver did not converge.")
    else:
        w = result["weights"]
        rc = result["risk_contribution"]
        alloc = pd.DataFrame({
            "Ticker": [t.replace(".NS", "") for t in selected],
            "Sector": [sector_map[t] for t in selected],
            "Cap tier": [cap_tier_map[t] for t in selected],
            "Weight %": w * 100,
            "Risk contribution %": rc,
        }).sort_values("Weight %", ascending=False).reset_index(drop=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected return", fmt_pct(result["expected_return"] * 100))
        c2.metric("Volatility", fmt_pct(result["volatility"] * 100))
        c3.metric("Sharpe ratio", f"{result['sharpe_ratio']:.2f}")
        c4.metric("# holdings (>0.5%)", int((w > 0.005).sum()))

        st.markdown("#### Target allocation")
        st.dataframe(
            alloc.style.format({"Weight %": "{:.2f}", "Risk contribution %": "{:.2f}"})
                 .background_gradient(subset=["Weight %"], cmap="Blues")
                 .background_gradient(subset=["Risk contribution %"], cmap="Oranges"),
            use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            st.markdown("#### Weight % by asset")
            bar = alloc.sort_values("Weight %")
            fig_w = go.Figure(go.Bar(x=bar["Weight %"], y=bar["Ticker"],
                                     orientation="h", marker_color="#1f77b4"))
            fig_w.update_layout(height=420, xaxis_title="Weight %", margin=dict(t=10))
            st.plotly_chart(fig_w, use_container_width=True)
        with right:
            st.markdown("#### Weight vs risk contribution")
            st.caption("Where the orange bar exceeds the blue, that asset carries "
                       "**more risk than its weight** — the 'weight ≠ risk' insight.")
            top = alloc.head(15)
            fig_wr = go.Figure()
            fig_wr.add_bar(x=top["Ticker"], y=top["Weight %"], name="Weight %",
                           marker_color="#1f77b4")
            fig_wr.add_bar(x=top["Ticker"], y=top["Risk contribution %"],
                           name="Risk contribution %", marker_color="#ff7f0e")
            fig_wr.update_layout(barmode="group", height=420,
                                 yaxis_title="%", margin=dict(t=10),
                                 legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig_wr, use_container_width=True)

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("#### Sector exposure")
            sec = alloc.groupby("Sector")["Weight %"].sum().sort_values(ascending=False)
            fig_s = go.Figure(go.Pie(labels=sec.index, values=sec.values, hole=0.5))
            fig_s.update_traces(textinfo="label+percent")
            fig_s.update_layout(height=380, showlegend=False, margin=dict(t=10))
            st.plotly_chart(fig_s, use_container_width=True)
        with d2:
            st.markdown("#### Cap-tier exposure")
            cap = alloc.groupby("Cap tier")["Weight %"].sum().sort_values(ascending=False)
            fig_c = go.Figure(go.Pie(labels=cap.index, values=cap.values, hole=0.5))
            fig_c.update_traces(textinfo="label+percent")
            fig_c.update_layout(height=380, showlegend=False, margin=dict(t=10))
            st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("#### Correlation of selected assets")
        corr = returns_df.corr()
        corr.index = [t.replace(".NS", "") for t in corr.index]
        corr.columns = [t.replace(".NS", "") for t in corr.columns]
        fig_h = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                     zmin=-1, zmax=1, colorscale="RdBu_r",
                                     colorbar=dict(title="ρ")))
        fig_h.update_layout(height=520, margin=dict(t=10))
        st.plotly_chart(fig_h, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3: Backtest
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Walk-forward backtest")
    cc1, cc2, cc3 = st.columns(3)
    # Adaptive max: never let the slider request more trailing history than the
    # currently selected tickers actually provide (recomputed per selection).
    max_lb = max(1, len(returns_df) // 252)
    if max_lb >= 2:
        lookback = cc1.slider("Lookback (years)", 1, max_lb, 1)
    else:
        lookback = 1
        cc1.metric("Lookback (years)", 1)
        cc1.caption("Only ~1 year of aligned history for this selection.")
    cost_bps = cc2.slider("Transaction cost (bps)", 0, 50, 10, 5)
    st.caption("Strategy uses the current model + constraints, rebalanced "
               f"{maps['rebalance_frequency']}, vs equal-weight and Nifty-50 benchmarks.")

    # Build cache-safe model kwargs for the backtest.
    bt_kwargs = {"max_weight": max_weight,
                 "sector_caps_items": sector_caps_items or None}
    if is_utility:
        bt_kwargs["risk_aversion"] = risk_aversion
    elif is_sharpe:
        bt_kwargs["risk_free_rate"] = rf
    elif is_target:
        bt_kwargs["target_return"] = float(target_return)
    # risk_parity needs only max_weight.
    if model_name == "Risk Parity":
        bt_kwargs.pop("sector_caps_items", None)

    with st.spinner("Running walk-forward backtest…"):
        res = run_backtest_cached(tuple(selected), model_name,
                                  tuple(sorted(bt_kwargs.items())), lookback,
                                  cost_bps, cov_method, maps["rebalance_frequency"],
                                  rf, min_obs)

    if res["status"] == "insufficient_history":
        btx = res["bt"]
        st.warning(
            f"Not enough trailing history for a {lookback}-year lookback with the "
            f"current data/ticker selection — {btx['available_days']} trading days "
            f"available, need {btx['lookback_days']}. Try a shorter lookback or "
            f"select more history.")
    else:
        bt, perf, benches, bench_perf = (res["bt"], res["perf"], res["benches"],
                                         res["bench_perf"])

        strat_curve = bt["equity_curve"]
        # --- Equity curve ---
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=strat_curve.index, y=strat_curve.values,
                                    name=model_name, line=dict(color="#1f77b4", width=2)))
        bench_styles = {"equal_weight": ("Equal weight", "#2ca02c"),
                        "buy_and_hold_equal": ("Buy & hold EW", "#9467bd"),
                        "nifty50": ("Nifty 50", "#d62728")}
        for key, (lbl, col) in bench_styles.items():
            if key in benches:
                fig_eq.add_trace(go.Scatter(x=benches[key].index, y=benches[key].values,
                                            name=lbl, line=dict(color=col, width=1.5,
                                                                dash="dot")))
        fig_eq.update_layout(height=360, yaxis_title="Growth of 1.0",
                             margin=dict(t=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_eq, use_container_width=True)
        if "nifty50" not in benches:
            st.caption("⚠️ Nifty-50 (^NSEI) live download unavailable — overlay omitted "
                       "(not faked).")

        # --- Drawdown (same x-axis) ---
        dd = perf["drawdown_series"]
        fig_dd = go.Figure(go.Scatter(x=dd.index, y=dd.values * 100, fill="tozeroy",
                                      line=dict(color="#d62728"), name="Drawdown"))
        fig_dd.update_layout(height=240, yaxis_title="Drawdown %", margin=dict(t=10),
                             xaxis_range=[strat_curve.index.min(), strat_curve.index.max()])
        st.plotly_chart(fig_dd, use_container_width=True)

        # --- Rolling 6-month Sharpe ---
        rs = perf["rolling_sharpe"].dropna()
        fig_rs = go.Figure(go.Scatter(x=rs.index, y=rs.values, line=dict(color="#1f77b4"),
                                      name="Rolling 6m Sharpe"))
        fig_rs.add_hline(y=0, line=dict(color="#aaa", dash="dot"))
        fig_rs.update_layout(height=240, yaxis_title="Rolling 6m Sharpe",
                             margin=dict(t=10))
        st.plotly_chart(fig_rs, use_container_width=True)

        # --- Weight-over-time stacked area (rebalance history) ---
        st.markdown("#### Weights over time (rebalances)")
        rw = bt["rebalance_weights"] * 100
        rw.columns = [c.replace(".NS", "") for c in rw.columns]
        fig_area = go.Figure()
        for col in rw.columns:
            fig_area.add_trace(go.Scatter(x=rw.index, y=rw[col], name=col,
                                          stackgroup="one", mode="lines"))
        fig_area.update_layout(height=380, yaxis_title="Weight %",
                               margin=dict(t=10), legend=dict(font=dict(size=9)))
        st.plotly_chart(fig_area, use_container_width=True)

        # --- Metrics table ---
        st.markdown("#### Performance metrics")
        rows = {model_name: perf["metrics"]}
        for key, (lbl, _) in bench_styles.items():
            if key in bench_perf:
                rows[lbl] = bench_perf[key]["metrics"]
        mt = pd.DataFrame(rows).T
        show = pd.DataFrame({
            "Ann. return (CAGR) %": mt["annualized_return"] * 100,
            "Volatility %": mt["annualized_vol"] * 100,
            "Sharpe": mt["sharpe_ratio"],
            "Max drawdown %": mt["max_drawdown"] * 100,
            "Avg turnover": mt.get("avg_turnover"),
        })
        st.dataframe(show.style.format({"Ann. return (CAGR) %": "{:.2f}",
                                        "Volatility %": "{:.2f}", "Sharpe": "{:.2f}",
                                        "Max drawdown %": "{:.2f}", "Avg turnover": "{:.3f}"}),
                     use_container_width=True)
        st.caption("Sharpe uses annualized **arithmetic** mean excess return; the "
                   "Return column is geometric CAGR — both standard, they won't divide "
                   "exactly (volatility drag).")


# ---------------------------------------------------------------------------
# TAB 4: Solver Validation
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Solver validation: active-set QP vs cvxpy/OSQP")
    st.markdown(
        "The core solver is a **from-scratch active-set method** (pure NumPy, no "
        "cvxpy/scipy in the hot path). It is validated against the independent "
        "**cvxpy + OSQP** convex solver on seeded random portfolio problems, plus "
        "a closed-form check. Agreement is at **machine precision** "
        "(~1e-15), far inside the 1e-4 acceptance tolerance.")
    st.dataframe(
        VALIDATION_ROWS.style.format({"max |Δweight|": "{:.1e}",
                                      "obj rel.diff": "{:.1e}"}),
        use_container_width=True, hide_index=True)
    st.caption("From tests/test_solver_validation.py. n=2 also matches the "
               "closed-form GMV solution x* = Σ⁻¹1 / (1ᵀΣ⁻¹1) to 4.4e-16. "
               "(cvxpy is a test-only dependency and is not installed in the "
               "deployed app.)")

    st.markdown("#### How the active-set method works (plain language)")
    st.markdown(
        "- **Working set:** the solver guesses which inequality constraints are "
        "*binding* (held as equalities) — e.g. which assets sit exactly at 0 "
        "(long-only) or at the max-weight cap.\n"
        "- **Each iteration** solves one equality-constrained KKT linear system "
        "for a step direction, then either:\n"
        "    - **adds** a constraint when the step would violate a new bound "
        "(an asset hits 0 or its cap), or\n"
        "    - **drops** a constraint whose Lagrange multiplier turns negative "
        "(staying on that bound is no longer optimal), or\n"
        "    - **stops** when every multiplier is ≥ 0 — the KKT conditions hold "
        "and the point is the global optimum (the problem is convex).\n"
        "- **Why this beats clip-and-renormalize:** the old approach solved the "
        "*unconstrained* equality system, zeroed negative weights, and rescaled — "
        "which violates the KKT conditions and silently breaks the return target. "
        "The active-set method enforces every constraint exactly and re-optimizes "
        "the remaining assets each time one is pinned.")
    st.markdown(
        "**KKT conditions** at the optimum: stationarity "
        "`Px + q + Aᵀλ = 0`, primal feasibility, dual feasibility `λ ≥ 0`, and "
        "complementary slackness `λᵢ·(aᵢᵀx − bᵢ) = 0`.")


# ---------------------------------------------------------------------------
# TAB 5: Model Comparison
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("All five models on the current universe & constraints")
    with st.spinner("Solving all models…"):
        cmp = compare_all_cached(tuple(selected), cov_method, max_weight,
                                 float(target_return), risk_aversion, rf,
                                 sector_caps_items, min_obs)
    weights_wide = cmp["weights"].copy()
    weights_wide.index = [t.replace(".NS", "") for t in weights_wide.index]

    st.markdown("#### Weight % by ticker (rows) × model (columns)")
    st.dataframe(
        weights_wide.style.format("{:.2f}").background_gradient(cmap="Blues", axis=None),
        use_container_width=True)

    st.markdown("#### Weight % per ticker, grouped by model")
    fig_g = go.Figure()
    for model in weights_wide.columns:
        fig_g.add_bar(x=weights_wide.index, y=weights_wide[model], name=model)
    fig_g.update_layout(barmode="group", height=420, yaxis_title="Weight %",
                        margin=dict(t=10), legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("#### Risk/return summary (sorted by Sharpe)")
    summ = cmp["summary"].copy()
    summ_show = pd.DataFrame({
        "Expected return %": summ["expected_return"].astype(float) * 100,
        "Volatility %": summ["volatility"].astype(float) * 100,
        "Sharpe ratio": summ["sharpe_ratio"].astype(float),
        "Converged": summ["converged"],
    }).sort_values("Sharpe ratio", ascending=False)
    st.dataframe(
        summ_show.style.format({"Expected return %": "{:.2f}", "Volatility %": "{:.2f}",
                                "Sharpe ratio": "{:.2f}"}),
        use_container_width=True)
