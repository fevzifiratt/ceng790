package divide

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object Analysis {

  /**
   * Produces a multi-section impact report using both DataFrame API and RDD operations.
   *
   * Sections:
   *   1. Cluster size distribution
   *   2. Salary impact by archetype          (mean, median, stddev)
   *   3. Job satisfaction by archetype       (mean score + label distribution)
   *   4. AI tool adoption breakdown          (most common tools per cluster)
   *   5. Tech stack profile                  (top languages per cluster)
   *   6. Experience & company size profile   (means per cluster)
   *   7. Country distribution (top 5 per cluster)  ← RDD-based
   *   8. DevType breakdown per cluster              ← RDD-based
   */
  def impactReport(df: DataFrame, spark: SparkSession): Unit = {

    import spark.implicits._

    println("\n" + "=" * 70)
    println("  THE AI DIVIDE — DEVELOPER ARCHETYPE IMPACT REPORT")
    println("=" * 70)

    // ── 1. Cluster size ────────────────────────────────────────────────────
    println("\n── 1. CLUSTER SIZE DISTRIBUTION ─────────────────────────────────────")
    df.groupBy("prediction", "archetype")
      .agg(count("*").as("count"))
      .orderBy("prediction")
      .show(truncate = false)

    // ── 2. Salary impact ───────────────────────────────────────────────────
    println("── 2. SALARY IMPACT BY ARCHETYPE (USD/year) ─────────────────────────")
    df.groupBy("archetype")
      .agg(
        count("salary_usd").as("n"),
        round(mean("salary_usd"),    2).as("mean_salary"),
        round(stddev("salary_usd"),  2).as("stddev_salary"),
        round(expr("percentile(salary_usd, 0.25)"), 2).as("p25_salary"),
        round(expr("percentile(salary_usd, 0.50)"), 2).as("median_salary"),
        round(expr("percentile(salary_usd, 0.75)"), 2).as("p75_salary")
      )
      .orderBy(desc("mean_salary"))
      .show(truncate = false)

    // ── 3. Job satisfaction impact ─────────────────────────────────────────
    println("── 3. JOB SATISFACTION BY ARCHETYPE ─────────────────────────────────")
    df.groupBy("archetype")
      .agg(
        count("job_sat_int").as("n"),
        round(mean("job_sat_int"), 3).as("mean_job_sat_score"),
        round(stddev("job_sat_int"), 3).as("stddev_job_sat")
      )
      .orderBy(desc("mean_job_sat_score"))
      .show(truncate = false)

    // JobSat label breakdown
    println("   JobSat label distribution per archetype:")
    df.groupBy("archetype", "JobSat")
      .agg(count("*").as("count"))
      .orderBy("archetype", desc("count"))
      .show(50, truncate = false)

    // ── 4. AI tool adoption breakdown ──────────────────────────────────────
    println("── 4. AI TOOL ADOPTION BY ARCHETYPE ─────────────────────────────────")

    val aiToolCols = Features.AI_TOOLS.map(t =>
      "uses_" + t.toLowerCase.replaceAll("[^a-z0-9]", "_").replaceAll("_+", "_").stripSuffix("_"))

    val toolAdoptionExprs = aiToolCols.map(c => round(mean(c) * 100, 1).as(c + "_pct"))

    df.groupBy("archetype")
      .agg(
        count("*").as("n"),
        round(mean("ai_tool_count"), 2).as("avg_tools_used"),
        toolAdoptionExprs.head,
        toolAdoptionExprs.tail: _*
      )
      .orderBy(desc("avg_tools_used"))
      .show(truncate = false)

    // ── 5. Tech stack profile per cluster ──────────────────────────────────
    println("── 5. TECH STACK PROFILE BY ARCHETYPE (% using language) ───────────")

    val langCols = Features.LANGUAGES.map(l =>
      "lang_" + l.toLowerCase.replaceAll("[^a-z0-9]", "_").replaceAll("_+", "_").stripSuffix("_"))

    val langExprs = langCols.map(c => round(mean(c) * 100, 1).as(c + "_pct"))

    df.groupBy("archetype")
      .agg(langExprs.head, langExprs.tail: _*)
      .orderBy("archetype")
      .show(truncate = false)

    // ── 6. Experience & company size profile ──────────────────────────────
    println("── 6. EXPERIENCE & COMPANY SIZE PROFILE ─────────────────────────────")
    df.groupBy("archetype")
      .agg(
        round(mean("years_code_pro"),  2).as("avg_years_exp"),
        round(mean("org_size_int"),    2).as("avg_org_size_score"),
        round(mean("ai_acc_int"),      3).as("avg_ai_trust_score"),
        round(mean("ai_threat_int"),   3).as("pct_see_ai_as_threat"),
        round(mean("ai_ben_int"),      3).as("avg_ai_benefit_score")
      )
      .orderBy("archetype")
      .show(truncate = false)

    // ── 7. Country distribution (RDD-based) ───────────────────────────────
    println("── 7. TOP 5 COUNTRIES BY ARCHETYPE (RDD operations) ─────────────────")

    // Use RDD to compute top countries per archetype
    val archetypeCountryRdd = df
      .select("archetype", "Country")
      .filter(col("Country").isNotNull)
      .rdd
      .map(row => ((row.getString(0), row.getString(1)), 1))
      .reduceByKey(_ + _)                              // count per (archetype, country)
      .map { case ((archetype, country), count) => (archetype, (country, count)) }
      .groupByKey()
      .mapValues(iter => iter.toSeq.sortBy(-_._2).take(5))  // top 5 per archetype

    println(f"\n  ${"Archetype"}%-22s  ${"Country"}%-30s  ${"Count"}%6s")
    println("  " + "-" * 62)
    archetypeCountryRdd.collect().sortBy(_._1).foreach { case (archetype, top5) =>
      top5.foreach { case (country, count) =>
        println(f"  $archetype%-22s  $country%-30s  $count%6d")
      }
      println()
    }

    // ── 8. DevType breakdown per archetype (RDD-based) ────────────────────
    println("── 8. TOP DEVELOPER ROLES BY ARCHETYPE (RDD operations) ────────────")

    // DevType is also semicolon-separated — we explode it via RDD flatMap
    val archetypeDevTypeRdd = df
      .select("archetype", "DevType")
      .filter(col("DevType").isNotNull)
      .rdd
      .flatMap { row =>
        val archetype = row.getString(0)
        val devTypes  = row.getString(1).split(";").map(_.trim)
        devTypes.map(dt => ((archetype, dt), 1))
      }
      .reduceByKey(_ + _)
      .map { case ((archetype, devType), count) => (archetype, (devType, count)) }
      .groupByKey()
      .mapValues(iter => iter.toSeq.sortBy(-_._2).take(5))

    println(f"\n  ${"Archetype"}%-22s  ${"Dev Role"}%-40s  ${"Count"}%6s")
    println("  " + "-" * 72)
    archetypeDevTypeRdd.collect().sortBy(_._1).foreach { case (archetype, top5) =>
      top5.foreach { case (devType, count) =>
        println(f"  $archetype%-22s  $devType%-40s  $count%6d")
      }
      println()
    }

    // ── 9. Summary table ──────────────────────────────────────────────────
    println("=" * 70)
    println("  SUMMARY: KEY METRICS PER ARCHETYPE")
    println("=" * 70)
    df.groupBy("archetype")
      .agg(
        count("*").as("developers"),
        round(mean("salary_usd"),     0).as("avg_salary_usd"),
        round(mean("job_sat_int"),    2).as("avg_job_sat_1_5"),
        round(mean("ai_tool_count"),  2).as("avg_ai_tools"),
        round(mean("years_code_pro"), 1).as("avg_years_exp"),
        round(mean("ai_acc_int"),     2).as("avg_ai_trust_0_4")
      )
      .orderBy(desc("avg_salary_usd"))
      .show(truncate = false)

    println("=" * 70)
    println("  Report complete. See output above for full breakdown.")
    println("=" * 70 + "\n")
  }
}