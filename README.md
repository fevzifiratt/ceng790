# CENG790 – AI Ecosystem Clustering

A Spark/Scala pipeline that clusters software developers into archetypes based on their AI tool usage, programming languages, and job characteristics. Data source is the [Stack Overflow Developer Survey](https://survey.stackoverflow.co/).

## Overview

| Step | Module | Description |
|------|--------|-------------|
| 1 | `Cleaning` | Load `survey_results_public.csv`, drop malformed rows, handle nulls |
| 2 | `Features` | Expand semicolon-separated columns into binary indicators, assemble & scale feature vectors |
| 3 | `OptimalK` | Compute WSSSE for k = 2–8 to support Elbow Method selection |
| 4 | `Clustering` | Train final K-Means model with chosen k |
| 5 | `Analysis` | Generate per-cluster impact report |

## Prerequisites

- Java 8
- Scala 2.12.17
- sbt 1.x
- Apache Spark 3.2.4 (managed via sbt)

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd CENG790
   ```

2. Place the survey CSV in the `data/` directory:
   ```
   data/survey_results_public.csv
   ```
   Download from: https://survey.stackoverflow.co/

3. Run the pipeline:
   ```bash
   sbt run
   ```

## Configuration

- **Number of clusters (k):** Set in `src/main/scala/ecosystem/Main.scala` after reviewing the elbow plot output from Step 3.
- **Shuffle partitions:** Configured to `8` for local runs; increase for cluster execution.

## Project Structure

```
├── build.sbt
├── data/                  # CSV input files (gitignored)
├── output/                # Spark output and warehouse
└── src/main/scala/ecosystem/
    ├── Main.scala
    ├── Cleaning.scala
    ├── Features.scala
    ├── OptimalK.scala
    ├── Clustering.scala
    └── Analysis.scala
```

## Course

CENG790 – Big Data Analytics
