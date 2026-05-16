#!/usr/bin/env python3
"""
Generate report figures from the Spark pipeline outputs.

Reads:
  - output/report/labelled_developers.parquet  (per-respondent)
  - output/report/<table>/part-*.csv           (summary tables)
  - output/run.log                             (elbow / silhouette)

Writes:
  - output/figures/*.png
"""

from __future__ import annotations
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "output" / "report"
FIG_DIR = ROOT / "output" / "figures"
LOG_FILE = ROOT / "output" / "run.log"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent ordering / colours for the three archetypes
ARCHETYPE_ORDER = ["Traditional Engineer", "AI Experimenter", "AI Power User"]
ARCHETYPE_COLOR = {
    "Traditional Engineer": "#4C72B0",
    "AI Experimenter":      "#DD8452",
    "AI Power User":        "#C44E52",
}
USAGE_ORDER = ["0_None", "1_Light_1to2", "2_Moderate_3to4", "3_Heavy_5plus"]
USAGE_LABEL = {
    "0_None":          "None (0)",
    "1_Light_1to2":    "Light (1–2)",
    "2_Moderate_3to4": "Moderate (3–4)",
    "3_Heavy_5plus":   "Heavy (5+)",
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 160
plt.rcParams["savefig.bbox"] = "tight"


def read_csv_dir(name: str) -> pd.DataFrame:
    """Read the single part-*.csv inside a Spark-style output directory."""
    files = glob.glob(str(REPORT_DIR / name / "part-*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV part file in {REPORT_DIR / name}")
    return pd.read_csv(files[0])


def save(fig, name: str):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────
# Figure 1: Elbow + Silhouette curves (parsed from run.log)
# ─────────────────────────────────────────────────────────────────────────
def fig_elbow():
    pat = re.compile(r"k=\s*(\d+)\s+WSSSE=\s*([\d.]+)\s+Silhouette=\s*([\d.]+)")
    rows = []
    for line in LOG_FILE.read_text().splitlines():
        m = pat.search(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    if not rows:
        print("  ! no elbow rows found in run.log")
        return
    df = pd.DataFrame(rows, columns=["k", "WSSSE", "Silhouette"]).drop_duplicates("k")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(df["k"], df["WSSSE"], marker="o", color="#4C72B0", lw=2)
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("WSSSE (within-cluster sum of squared errors)")
    ax1.set_title("Elbow method")
    ax1.grid(True, alpha=0.4)
    ax1.axvline(3, color="#C44E52", ls="--", alpha=0.7, label="chosen k = 3")
    ax1.legend()

    ax2.plot(df["k"], df["Silhouette"], marker="o", color="#DD8452", lw=2)
    ax2.set_xlabel("k (number of clusters)")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Cluster cohesion / separation")
    ax2.grid(True, alpha=0.4)
    ax2.axvline(3, color="#C44E52", ls="--", alpha=0.7, label="chosen k = 3")
    ax2.legend()

    fig.suptitle("Choosing k via Elbow + Silhouette (clean feature set)", y=1.02, fontsize=12)
    save(fig, "01_elbow_silhouette")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2: Cluster centroids heatmap (z-scores)
# ─────────────────────────────────────────────────────────────────────────
def fig_centroids():
    """Parse the centroid table from run.log into a heatmap."""
    text = LOG_FILE.read_text()
    block = re.search(
        r"Cluster centroids \(z-scores on key features\):.*?Final model Silhouette",
        text, flags=re.DOTALL,
    )
    if not block:
        print("  ! could not find centroid block in run.log")
        return
    lines = block.group().splitlines()
    rows, ids = [], []
    for ln in lines:
        m = re.match(r"\s*(\[info\])?\s*C\s*(\d+)\s+(.*)", ln)
        if m and "ai_sel" not in ln:
            cid = int(m.group(2))
            nums = re.findall(r"[-+]?\d+\.\d+", m.group(3))
            if len(nums) >= 9:
                ids.append(f"C{cid}")
                rows.append([float(x) for x in nums[:9]])
    if not rows:
        print("  ! no centroid rows parsed")
        return

    cols = ["ai_select", "ai_products", "ai_tasks",
            "ai_trust", "ai_threat", "ai_sentiment",
            "ai_benefits", "years_exp", "org_size"]
    df = pd.DataFrame(rows, index=ids, columns=cols)

    # Map cluster id → archetype label using the persisted summary
    sizes = read_csv_dir("cluster_sizes").set_index("prediction")
    df.index = [sizes.loc[int(c[1:]), "archetype"] for c in df.index]
    df = df.reindex([a for a in ARCHETYPE_ORDER if a in df.index])

    fig, ax = plt.subplots(figsize=(9, 3.2))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1.5, vmax=1.5, ax=ax,
                cbar_kws={"label": "z-score (relative to global mean)"})
    ax.set_title("Cluster centroids on the 9 clustering features")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    save(fig, "02_centroid_heatmap")


# ─────────────────────────────────────────────────────────────────────────
# Figure 3: Cluster sizes
# ─────────────────────────────────────────────────────────────────────────
def fig_cluster_sizes():
    df = read_csv_dir("cluster_sizes")
    df = df.set_index("archetype").reindex(ARCHETYPE_ORDER)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(df.index, df["count"],
                  color=[ARCHETYPE_COLOR[a] for a in df.index])
    ax.set_ylabel("Number of developers")
    ax.set_title("Cluster sizes (n = 13,552 full-time devs)")
    for b, v in zip(bars, df["count"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,}",
                ha="center", fontsize=10)
    save(fig, "03_cluster_sizes")


# ─────────────────────────────────────────────────────────────────────────
# Figure 4: Salary + JobSat per archetype (the headline numbers)
# ─────────────────────────────────────────────────────────────────────────
def fig_salary_jobsat():
    sal = read_csv_dir("salary_by_archetype").set_index("archetype").reindex(ARCHETYPE_ORDER)
    js  = read_csv_dir("jobsat_by_archetype").set_index("archetype").reindex(ARCHETYPE_ORDER)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    x = np.arange(len(ARCHETYPE_ORDER))
    w = 0.36
    ax1.bar(x - w/2, sal["mean_salary"],   w, label="Mean",
            color=[ARCHETYPE_COLOR[a] for a in ARCHETYPE_ORDER])
    ax1.bar(x + w/2, sal["median_salary"], w, label="Median", alpha=0.55,
            color=[ARCHETYPE_COLOR[a] for a in ARCHETYPE_ORDER])
    ax1.set_xticks(x); ax1.set_xticklabels(ARCHETYPE_ORDER, rotation=15, ha="right")
    ax1.set_ylabel("USD / year")
    ax1.set_title("Annual salary by archetype")
    ax1.legend()
    for i, (m, md) in enumerate(zip(sal["mean_salary"], sal["median_salary"])):
        ax1.text(i - w/2, m + 1500, f"${m/1000:.0f}k", ha="center", fontsize=9)
        ax1.text(i + w/2, md + 1500, f"${md/1000:.0f}k", ha="center", fontsize=9)

    ax2.bar(ARCHETYPE_ORDER, js["mean_job_sat"],
            color=[ARCHETYPE_COLOR[a] for a in ARCHETYPE_ORDER])
    ax2.set_ylim(6, 7.5)
    ax2.set_ylabel("Mean job satisfaction (0–10)")
    ax2.set_title("Job satisfaction by archetype")
    for i, v in enumerate(js["mean_job_sat"]):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("Headline outcomes per archetype", y=1.02, fontsize=12)
    save(fig, "04_salary_jobsat_per_archetype")


# ─────────────────────────────────────────────────────────────────────────
# Figure 5: Salary gradient by AI usage bin (with experience overlay)
# ─────────────────────────────────────────────────────────────────────────
def fig_salary_by_ai_usage():
    df = read_csv_dir("deep_salary_by_ai_usage")
    df = df.set_index("ai_usage_bin").reindex(USAGE_ORDER)

    fig, ax1 = plt.subplots(figsize=(8.5, 4))
    color1, color2 = "#C44E52", "#4C72B0"
    bars = ax1.bar([USAGE_LABEL[u] for u in df.index], df["mean_salary"],
                   color=color1, alpha=0.8, label="Mean salary")
    ax1.set_ylabel("Mean salary (USD)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    for b, v in zip(bars, df["mean_salary"]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 800,
                 f"${v/1000:.0f}k", ha="center", fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot([USAGE_LABEL[u] for u in df.index], df["mean_years_exp"],
             color=color2, marker="o", lw=2, label="Mean years of experience")
    ax2.set_ylabel("Mean years of professional experience", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Raw salary gradient by AI usage bin\n(years of experience overlay reveals the confound)")
    save(fig, "05_salary_by_ai_usage")


# ─────────────────────────────────────────────────────────────────────────
# Figure 6: The Simpson's-paradox plot (salary × experience × AI usage)
# ─────────────────────────────────────────────────────────────────────────
def fig_simpsons():
    df = read_csv_dir("deep_salary_by_exp_x_ai")
    df = df.set_index("experience_bin")
    df = df[USAGE_ORDER]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    markers = {"Junior_0to4": "o", "Mid_5to10": "s", "Senior_11plus": "D"}
    colors  = {"Junior_0to4": "#4C72B0",
               "Mid_5to10":   "#DD8452",
               "Senior_11plus": "#C44E52"}
    pretty = {"Junior_0to4": "Junior (0–4 yrs)",
              "Mid_5to10": "Mid (5–10 yrs)",
              "Senior_11plus": "Senior (11+ yrs)"}

    for exp in ["Junior_0to4", "Mid_5to10", "Senior_11plus"]:
        ax.plot([USAGE_LABEL[u] for u in USAGE_ORDER],
                df.loc[exp].values,
                marker=markers[exp], color=colors[exp],
                lw=2.2, ms=9, label=pretty[exp])
        for x, y in zip(range(4), df.loc[exp].values):
            ax.text(x, y + 1500, f"${y/1000:.0f}k", ha="center", fontsize=9,
                    color=colors[exp])

    ax.set_ylabel("Mean salary (USD)")
    ax.set_xlabel("AI usage bin")
    ax.set_title("Salary × experience × AI usage  —  Simpson's paradox\n"
                 "Within juniors, AI use lowers salary; within seniors, it raises it")
    ax.legend(title="Experience tier")
    ax.grid(True, alpha=0.4)
    save(fig, "06_simpsons_paradox")


# ─────────────────────────────────────────────────────────────────────────
# Figure 7: Salary × country × AI usage (top 5 countries)
# ─────────────────────────────────────────────────────────────────────────
def fig_country():
    df = read_csv_dir("deep_salary_by_country_x_ai")
    df = df.set_index("Country")[USAGE_ORDER]
    # Shorten country names for the chart
    rename = {
        "United States of America": "USA",
        "United Kingdom of Great Britain and Northern Ireland": "UK",
    }
    df.index = [rename.get(c, c) for c in df.index]
    df = df.reindex(["USA", "UK", "Germany", "Ukraine", "India"])

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    df.T.plot(kind="line", marker="o", ax=ax, lw=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([USAGE_LABEL[u] for u in USAGE_ORDER])
    ax.set_ylabel("Mean salary (USD)")
    ax.set_xlabel("AI usage bin")
    ax.set_title("Salary by country × AI usage\nThe penalty is concentrated in lower-cost markets (notably India)")
    ax.legend(title="Country")
    ax.grid(True, alpha=0.4)
    save(fig, "07_country_x_ai")


# ─────────────────────────────────────────────────────────────────────────
# Figure 8: Salary × dev role × AI usage (heatmap)
# ─────────────────────────────────────────────────────────────────────────
def fig_role_heatmap():
    df = read_csv_dir("deep_salary_by_role_x_ai")
    df = df.set_index("primary_role")[USAGE_ORDER]
    df.columns = [USAGE_LABEL[u] for u in df.columns]
    df.index = [r.replace("Developer, ", "").capitalize() for r in df.index]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sns.heatmap(df / 1000, annot=True, fmt=".0f",
                cmap="RdYlGn", center=85, ax=ax,
                cbar_kws={"label": "Mean salary (USD, thousands)"})
    ax.set_title("Salary (USD k) by primary role × AI usage")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, "08_role_x_ai_heatmap")


# ─────────────────────────────────────────────────────────────────────────
# Figure 9: Pearson correlation matrix
# ─────────────────────────────────────────────────────────────────────────
def fig_corr():
    df = read_csv_dir("deep_correlations")
    cols = sorted(set(df["var_a"]) | set(df["var_b"]))
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for _, r in df.iterrows():
        mat.loc[r["var_a"], r["var_b"]] = r["pearson_corr"]
        mat.loc[r["var_b"], r["var_a"]] = r["pearson_corr"]
    # Reorder for readability
    order = ["salary_usd", "years_code_pro", "org_size_int", "job_sat_int",
             "ai_product_count", "ai_task_count", "ai_ben_count",
             "ai_acc_int", "ai_threat_int", "ai_sent_int"]
    mat = mat.reindex(index=order, columns=order)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(mat, annot=True, fmt="+.2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, square=True,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Pearson correlations among numeric variables")
    save(fig, "09_correlation_matrix")


# ─────────────────────────────────────────────────────────────────────────
# Figure 10: Regression coefficients (with std-error bars)
# ─────────────────────────────────────────────────────────────────────────
def fig_regression():
    df = read_csv_dir("deep_regression_coefficients")
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    colors = ["#C44E52" if (c < 0 and p < 0.05) else
              "#4C72B0" if (c > 0 and p < 0.05) else
              "#888888"
              for c, p in zip(df["coefficient"], df["p_value"])]
    ax.barh(df["feature"], df["coefficient"], xerr=df["std_error"],
            color=colors, error_kw={"elinewidth": 1.4, "capsize": 5})
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Effect on salary (USD per +1 unit of feature)")
    ax.set_title("Linear regression  salary ~ AI use + experience + org size\n"
                 "Negative red bars (p<0.05) hurt salary;  positive blue bars help")
    xmax = max(abs(df["coefficient"].min()), abs(df["coefficient"].max()))
    pad  = xmax * 0.04
    for i, (c, p) in enumerate(zip(df["coefficient"], df["p_value"])):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        # Always place label OUTSIDE the bar so it never overlaps the rectangle
        x_text = (c + pad) if c >= 0 else (c - pad)
        ha     = "left"    if c >= 0 else "right"
        ax.text(x_text, i, f"{c:+,.0f}  ({sig})",
                va="center", ha=ha, fontsize=10)
    # Make sure the axes are wide enough for the labels
    ax.set_xlim(-xmax * 1.6, xmax * 1.35)
    save(fig, "10_regression_coefficients")


# ─────────────────────────────────────────────────────────────────────────
# Figure 11: Salary distributions per archetype (boxplot from parquet)
# ─────────────────────────────────────────────────────────────────────────
def fig_salary_dist():
    pq = read_csv_dir.__globals__["pd"].read_parquet(
        REPORT_DIR / "labelled_developers.parquet"
    )
    pq = pq[pq["salary_usd"].notna()].copy()
    # Cap the y-axis at the 99th percentile so outliers don't dominate
    cap = pq["salary_usd"].quantile(0.99)
    pq = pq[pq["salary_usd"] <= cap]

    pq["archetype"] = pd.Categorical(pq["archetype"],
                                     categories=ARCHETYPE_ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.boxplot(data=pq, x="archetype", y="salary_usd", ax=ax, hue="archetype",
                palette=ARCHETYPE_COLOR, showfliers=False, legend=False)
    ax.set_title("Salary distribution per archetype (outliers above p99 trimmed)")
    ax.set_ylabel("Annual salary (USD)")
    ax.set_xlabel("")
    save(fig, "11_salary_distribution_box")


# ─────────────────────────────────────────────────────────────────────────
# Figure 12: Primary role mix per archetype (stacked bar)
# ─────────────────────────────────────────────────────────────────────────
def fig_role_mix():
    df = read_csv_dir("primary_role_by_archetype")
    df = df.set_index("archetype").reindex(ARCHETYPE_ORDER)
    pct_cols = [c for c in df.columns if c.endswith("_pct")]
    short = {c: c.replace("dev_", "").replace("_pct", "")
                  .replace("_", " ").title() for c in pct_cols}

    plot_df = df[pct_cols].rename(columns=short)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    plot_df.plot(kind="barh", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("% of cluster")
    ax.set_title("Primary developer role per archetype")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=9, frameon=False)
    ax.set_ylabel("")
    save(fig, "12_role_mix_per_archetype")


# ─────────────────────────────────────────────────────────────────────────
# Figure 13: Remote-work mix per archetype
# ─────────────────────────────────────────────────────────────────────────
def fig_remote_mix():
    df = read_csv_dir("remote_work_by_archetype").set_index("archetype").reindex(ARCHETYPE_ORDER)
    pct_cols = [c for c in df.columns if c.endswith("_pct")]
    label = {
        "Remote_pct": "Remote",
        "Hybrid (some remote, some in-person)_pct": "Hybrid",
        "In-person_pct": "In-person",
    }
    df = df[pct_cols].rename(columns=label)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    df.plot(kind="bar", ax=ax,
            color=["#4C72B0", "#DD8452", "#C44E52"])
    ax.set_ylabel("% of cluster")
    ax.set_title("Remote-work mix per archetype  (very similar — not a discriminator)")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(title="Work mode")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=9, padding=2)
    save(fig, "13_remote_work_per_archetype")


def main():
    print(f"Writing figures to {FIG_DIR.relative_to(ROOT)}/")
    fig_elbow()
    fig_centroids()
    fig_cluster_sizes()
    fig_salary_jobsat()
    fig_salary_by_ai_usage()
    fig_simpsons()
    fig_country()
    fig_role_heatmap()
    fig_corr()
    fig_regression()
    fig_salary_dist()
    fig_role_mix()
    fig_remote_mix()
    print("done.")


if __name__ == "__main__":
    main()
