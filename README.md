# CENG790 – The AI Divide

A Spark/Scala pipeline that clusters software developers into **AI-usage archetypes** from the Stack Overflow Developer Survey, then stress-tests the headline finding *"AI Power Users earn less"* against confounders (experience, country, role) using MLlib regression and Pearson correlations.

A companion Python script converts the Spark outputs into a set of figures consumed by a LaTeX report.

Data source: [Stack Overflow Developer Survey](https://survey.stackoverflow.co/).

## Pipeline

| Step | Module | Description |
|------|--------|-------------|
| 1 | `Cleaning` | Load `dataset.csv`, drop malformed rows, normalise nulls, derive ordinal scores (AI trust, threat, sentiment, org size, job satisfaction). |
| 2 | `Features` | Expand semicolon-separated columns (languages, AI tools, AI tasks, AI benefits) into binary indicators; assemble + standard-scale the feature vector. |
| 3 | `OptimalK` | Compute WSSSE and Silhouette for k = 2–8 to support Elbow-method selection. |
| 4 | `Clustering` | Train the final K-Means model with the chosen k and label each cluster with an archetype (`Traditional Engineer`, `AI Experimenter`, `AI Power User`). |
| 5 | `Analysis.impactReport` | Per-archetype summaries: cluster sizes, salary, job satisfaction, AI-tool adoption, tech stack, experience / org-size profile, remote-work mix, MainBranch mix, primary dev role, and RDD-based top-5 countries / roles. |
| 6 | `Analysis.deepDive` | Salary gradient by AI-usage bin, salary × experience × AI (Simpson's-paradox check), salary × country × AI, salary × role × AI, Pearson correlation matrix, and a Spark MLlib `LinearRegression` of `salary ~ ai_use + experience + org_size` with coefficients, std errors, t-stats and p-values. |
| 7 | `scripts/make_plots.py` | Reads `output/report/` + `output/run.log` and writes 13 publication-quality figures into `output/figures/`. |

All Spark sections write a single-part CSV under `output/report/<name>/`. The fully-labelled per-respondent dataset is also persisted as `output/report/labelled_developers.parquet`.

## Prerequisites

**Scala pipeline**
- Java 8
- Scala 2.12.17
- sbt 1.x
- Apache Spark 3.2.4 (managed via sbt)

**Plotting (optional)**
- Python 3.10+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `pyarrow`

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd CENG790
   ```

2. Place the survey CSV in the `data/` directory:
   ```
   data/dataset.csv
   ```
   Download from <https://survey.stackoverflow.co/>. (You can point to a different file with the `SURVEY_CSV` env var.)

3. Run the Spark pipeline:
   ```bash
   sbt run
   ```
   The full report (cluster sizes → deep-dive regression) prints to the console and persists to `output/report/`. Pipe it to a log file so the plotting script can parse the WSSSE/Silhouette/centroid lines:
   ```bash
   sbt run 2>&1 | tee output/run.log
   ```

4. (Optional) Generate report figures:
   ```bash
   python -m venv .venv-plot
   source .venv-plot/bin/activate
   pip install pandas numpy matplotlib seaborn pyarrow
   python scripts/make_plots.py
   ```
   Figures land in `output/figures/*.png` and are picked up by `report/main.tex`.

## Configuration

The pipeline reads three environment variables (all optional):

| Variable | Default | Purpose |
|----------|---------|---------|
| `SURVEY_CSV` | `data/dataset.csv` | Path to the input survey CSV. |
| `CLUSTER_K` | `3` | Number of K-Means clusters used in the final model. Pick after reviewing the elbow + silhouette curves from step 3. |
| `OUTPUT_DIR` | `output/report` | Where the CSV summaries and the labelled-developers parquet are written. |

Other knobs:
- **Shuffle partitions:** set to `8` in `Main.scala` for local runs; raise for cluster execution.
- **Hadoop home (Windows only):** auto-set to `C:\hadoop` when `os.name` contains "win".

## Project Structure

```
.
├── build.sbt
├── data/
│   └── dataset.csv                # input survey (gitignored)
├── src/main/scala/ecosystem/      # Scala sources (package `ecosystem`)
│   ├── Main.scala
│   ├── Cleaning.scala
│   ├── Features.scala
│   ├── OptimalK.scala
│   ├── Clustering.scala
│   └── Analysis.scala
├── scripts/
│   └── make_plots.py              # Spark output → PNG figures
├── report/
│   └── main.tex                   # LaTeX report consuming output/figures
└── output/                        # generated (gitignored)
    ├── report/
    │   ├── cluster_sizes/
    │   ├── salary_by_archetype/
    │   ├── jobsat_by_archetype/
    │   ├── ai_tool_adoption_by_archetype/
    │   ├── tech_stack_by_archetype/
    │   ├── profile_by_archetype/
    │   ├── remote_work_by_archetype/
    │   ├── main_branch_by_archetype/
    │   ├── primary_role_by_archetype/
    │   ├── summary_by_archetype/
    │   ├── top_countries_by_archetype/
    │   ├── top_devroles_by_archetype/
    │   ├── deep_salary_by_ai_usage/
    │   ├── deep_salary_by_exp_x_ai/
    │   ├── deep_counts_by_exp_x_ai/
    │   ├── deep_salary_by_country_x_ai/
    │   ├── deep_salary_by_role_x_ai/
    │   ├── deep_correlations/
    │   ├── deep_regression_coefficients/
    │   └── labelled_developers.parquet
    ├── figures/                   # produced by scripts/make_plots.py
    └── run.log                    # tee'd console output (parsed for elbow / centroids)
```

## Course

CENG790 – Big Data Analytics
