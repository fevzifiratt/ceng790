package divide

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression

/**
 * Multi-section impact report on the labelled cluster DataFrame.
 *
 * Mixes Spark DataFrame API (sections 1–6, 9) with low-level RDD
 * operations (sections 7–8) as required by the course brief.
 *
 * If `outputDir` is provided, the labelled DataFrame plus all summary
 * tables are also persisted as CSV under that directory.
 */
object Analysis {

  def impactReport(df: DataFrame, spark: SparkSession, outputDir: String = null): Unit = {

    import spark.implicits._

    println("\n" + "=" * 70)
    println("  THE AI DIVIDE — DEVELOPER ARCHETYPE IMPACT REPORT")
    println("=" * 70)

    // ── 1. Cluster size ────────────────────────────────────────────────────
    println("\n── 1. CLUSTER SIZE DISTRIBUTION ─────────────────────────────────────")
    val sizeDf = df.groupBy("prediction", "archetype")
      .agg(count("*").as("count"))
      .orderBy("prediction")
    sizeDf.show(truncate = false)

    // ── 2. Salary impact ───────────────────────────────────────────────────
    println("── 2. SALARY IMPACT BY ARCHETYPE (USD/year) ─────────────────────────")
    val salaryDf = df.groupBy("archetype")
      .agg(
        count("salary_usd").as("n"),
        round(mean("salary_usd"),    2).as("mean_salary"),
        round(stddev("salary_usd"),  2).as("stddev_salary"),
        round(expr("percentile(salary_usd, 0.25)"), 2).as("p25_salary"),
        round(expr("percentile(salary_usd, 0.50)"), 2).as("median_salary"),
        round(expr("percentile(salary_usd, 0.75)"), 2).as("p75_salary")
      )
      .orderBy(desc("mean_salary"))
    salaryDf.show(truncate = false)

    // ── 3. Job satisfaction (0–10 numeric in 2024) ────────────────────────
    println("── 3. JOB SATISFACTION BY ARCHETYPE (0–10 scale) ────────────────────")
    val jobsatDf = df.groupBy("archetype")
      .agg(
        count("job_sat_int").as("n"),
        round(mean("job_sat_int"),   3).as("mean_job_sat"),
        round(stddev("job_sat_int"), 3).as("stddev_job_sat"),
        round(expr("percentile(job_sat_int, 0.50)"), 1).as("median_job_sat")
      )
      .orderBy(desc("mean_job_sat"))
    jobsatDf.show(truncate = false)

    // ── 4. AI tool adoption breakdown ──────────────────────────────────────
    println("── 4. AI PRODUCT ADOPTION BY ARCHETYPE (% using each) ──────────────")

    val aiToolCols = Features.AI_TOOLS.map(t => "uses_" + Features.safeName(t))

    val toolAdoptionExprs = aiToolCols.map(c => round(mean(c) * 100, 1).as(c + "_pct"))

    val toolAggExprs: Seq[org.apache.spark.sql.Column] = Seq(
      count("*").as("n"),
      round(mean("ai_product_count"), 2).as("avg_products"),
      round(mean("ai_task_count"),    2).as("avg_tasks")
    ) ++ toolAdoptionExprs

    val toolsDf = df.groupBy("archetype")
      .agg(toolAggExprs.head, toolAggExprs.tail: _*)
      .orderBy(desc("avg_products"))
    toolsDf.show(truncate = false)

    // ── 5. Tech stack profile per cluster ──────────────────────────────────
    println("── 5. TECH STACK PROFILE BY ARCHETYPE (% using language) ───────────")

    val langCols = Features.LANGUAGES.map(l => "lang_" + Features.safeName(l))

    val langExprs = langCols.map(c => round(mean(c) * 100, 1).as(c + "_pct"))

    val stackDf = df.groupBy("archetype")
      .agg(langExprs.head, langExprs.tail: _*)
      .orderBy("archetype")
    stackDf.show(truncate = false)

    // ── 6. Experience & company size profile ──────────────────────────────
    println("── 6. EXPERIENCE / COMPANY-SIZE / AI-SENTIMENT PROFILE ──────────────")
    val profileDf = df.groupBy("archetype")
      .agg(
        round(mean("years_code_pro"),  2).as("avg_years_exp"),
        round(mean("org_size_int"),    2).as("avg_org_size_score"),
        round(mean("ai_acc_int"),      3).as("avg_ai_trust_0_4"),
        round(mean("ai_threat_int"),   3).as("avg_ai_threat_0_2"),
        round(mean("ai_sent_int"),     3).as("avg_ai_sentiment_0_4"),
        round(mean("ai_ben_count"),    3).as("avg_perceived_benefits")
      )
      .orderBy("archetype")
    profileDf.show(truncate = false)

    // ── 6A. Remote-work mix per archetype (% of cluster) ──────────────────
    println("── 6A. REMOTE-WORK MIX BY ARCHETYPE (%) ─────────────────────────────")
    val remoteWorkValues = Seq("Remote", "Hybrid (some remote, some in-person)", "In-person")

    val rwTotals = df
      .filter(col("RemoteWork").isNotNull)
      .groupBy("archetype").agg(count("*").as("total"))

    val rwCounts = df
      .filter(col("RemoteWork").isNotNull)
      .groupBy("archetype")
      .pivot("RemoteWork", remoteWorkValues)
      .count()

    val rwPct = rwCounts.join(rwTotals, "archetype")
    val rwPctFinal = remoteWorkValues.foldLeft(rwPct) { (acc, v) =>
      acc.withColumn(v + "_pct",
        round(coalesce(col(v), lit(0L)) * 100.0 / col("total"), 1))
    }.select(
      col("archetype") +: (col("total") +: remoteWorkValues.map(v => col(v + "_pct"))): _*
    ).orderBy("archetype")
    rwPctFinal.show(truncate = false)

    // ── 6B. MainBranch mix per archetype (%) ──────────────────────────────
    println("── 6B. MAIN-BRANCH (developer self-identification) BY ARCHETYPE (%) ─")
    val mainBranchValues = Seq(
      "I am a developer by profession",
      "I am not primarily a developer, but I write code sometimes as part of my work/studies",
      "I used to be a developer by profession, but no longer am",
      "I code primarily as a hobby",
      "I am learning to code"
    )
    val mbTotals = df
      .filter(col("MainBranch").isNotNull)
      .groupBy("archetype").agg(count("*").as("total"))
    val mbCounts = df
      .filter(col("MainBranch").isNotNull)
      .groupBy("archetype")
      .pivot("MainBranch", mainBranchValues)
      .count()
    val mbPct = mbCounts.join(mbTotals, "archetype")
    // Use short aliases so the table fits on screen
    val mbAliases = Seq(
      "professional_pct", "occasional_pct", "former_pct", "hobbyist_pct", "learning_pct"
    )
    val mbPctFinal = mainBranchValues.zip(mbAliases).foldLeft(mbPct) {
      case (acc, (v, alias)) =>
        acc.withColumn(alias,
          round(coalesce(col(v), lit(0L)) * 100.0 / col("total"), 1))
    }.select(
      col("archetype") +: (col("total") +: mbAliases.map(col)): _*
    ).orderBy("archetype")
    mbPctFinal.show(truncate = false)

    // ── 6C. DevType mix per archetype (% per primary role) ────────────────
    //   DevType is multi-value; we use the FIRST listed role as the primary
    //   role. Top 8 roles globally are reported; everything else is dropped
    //   from the pivot to keep the table readable.
    println("── 6C. PRIMARY DEVELOPER ROLE BY ARCHETYPE (% of cluster) ───────────")
    val withPrimaryRole = df
      .withColumn("primary_role",
        when(col("DevType").isNotNull,
          split(col("DevType"), ";").getItem(0)
        ).otherwise(lit("Unknown"))
      )

    val topRoles: Seq[String] = withPrimaryRole
      .filter(col("primary_role") =!= "Unknown")
      .groupBy("primary_role").count()
      .orderBy(desc("count")).limit(8)
      .select("primary_role").as[String].collect().toSeq

    val dtTotals = withPrimaryRole.groupBy("archetype").agg(count("*").as("total"))
    val dtCounts = withPrimaryRole
      .filter(col("primary_role").isin(topRoles: _*))
      .groupBy("archetype")
      .pivot("primary_role", topRoles)
      .count()
      .na.fill(0L)
    val dtPct = dtCounts.join(dtTotals, "archetype")

    val dtPctFinal = topRoles.foldLeft(dtPct) { (acc, role) =>
      val short = role.replace("Developer, ", "dev_")
        .replaceAll("[^A-Za-z0-9]", "_")
        .replaceAll("_+", "_")
        .stripSuffix("_") + "_pct"
      acc.withColumn(short, round(col(role) * 100.0 / col("total"), 1))
    }
    val dtFinalCols = "archetype" +: "total" +: topRoles.map { role =>
      role.replace("Developer, ", "dev_")
        .replaceAll("[^A-Za-z0-9]", "_")
        .replaceAll("_+", "_")
        .stripSuffix("_") + "_pct"
    }
    val dtPctReady = dtPctFinal.select(dtFinalCols.head, dtFinalCols.tail: _*)
      .orderBy("archetype")
    dtPctReady.show(truncate = false)

    // ── 7. Country distribution (RDD-based) ───────────────────────────────
    println("── 7. TOP 5 COUNTRIES BY ARCHETYPE (RDD operations) ─────────────────")

    val archetypeCountryRdd = df
      .select("archetype", "Country")
      .filter(col("Country").isNotNull)
      .rdd
      .map(row => ((row.getString(0), row.getString(1)), 1))
      .reduceByKey(_ + _)
      .map { case ((arch, country), c) => (arch, (country, c)) }
      .groupByKey()
      .mapValues(_.toSeq.sortBy(-_._2).take(5))

    println(f"\n  ${"Archetype"}%-25s  ${"Country"}%-30s  ${"Count"}%6s")
    println("  " + "-" * 65)
    val countryRows = archetypeCountryRdd.collect().sortBy(_._1)
    countryRows.foreach { case (arch, top5) =>
      top5.foreach { case (country, c) =>
        println(f"  $arch%-25s  $country%-30s  $c%6d")
      }
      println()
    }

    // ── 8. DevType breakdown per archetype (RDD-based) ────────────────────
    println("── 8. TOP 5 DEVELOPER ROLES BY ARCHETYPE (RDD operations) ───────────")

    val archetypeDevTypeRdd = df
      .select("archetype", "DevType")
      .filter(col("DevType").isNotNull)
      .rdd
      .flatMap { row =>
        val arch     = row.getString(0)
        val devTypes = row.getString(1).split(";").map(_.trim).filter(_.nonEmpty)
        devTypes.map(dt => ((arch, dt), 1))
      }
      .reduceByKey(_ + _)
      .map { case ((arch, devType), c) => (arch, (devType, c)) }
      .groupByKey()
      .mapValues(_.toSeq.sortBy(-_._2).take(5))

    println(f"\n  ${"Archetype"}%-25s  ${"Dev Role"}%-40s  ${"Count"}%6s")
    println("  " + "-" * 75)
    val devTypeRows = archetypeDevTypeRdd.collect().sortBy(_._1)
    devTypeRows.foreach { case (arch, top5) =>
      top5.foreach { case (devType, c) =>
        println(f"  $arch%-25s  $devType%-40s  $c%6d")
      }
      println()
    }

    // ── 9. Summary table ──────────────────────────────────────────────────
    println("=" * 70)
    println("  SUMMARY: KEY METRICS PER ARCHETYPE")
    println("=" * 70)
    val summaryDf = df.groupBy("archetype")
      .agg(
        count("*").as("developers"),
        round(mean("salary_usd"),       0).as("avg_salary_usd"),
        round(mean("job_sat_int"),      2).as("avg_job_sat_0_10"),
        round(mean("ai_product_count"), 2).as("avg_ai_products"),
        round(mean("ai_task_count"),    2).as("avg_ai_tasks"),
        round(mean("years_code_pro"),   1).as("avg_years_exp"),
        round(mean("ai_acc_int"),       2).as("avg_ai_trust_0_4")
      )
      .orderBy(desc("avg_salary_usd"))
    summaryDf.show(truncate = false)

    // ── Persist artifacts ─────────────────────────────────────────────────
    if (outputDir != null) {
      println(s"\n── PERSISTING OUTPUTS to $outputDir ────────────────────────────")

      def writeCsv(d: DataFrame, name: String): Unit = {
        val path = s"$outputDir/$name"
        d.coalesce(1).write.mode("overwrite").option("header", "true").csv(path)
        println(s"  wrote $path")
      }

      writeCsv(sizeDf,     "cluster_sizes")
      writeCsv(salaryDf,   "salary_by_archetype")
      writeCsv(jobsatDf,   "jobsat_by_archetype")
      writeCsv(toolsDf,    "ai_tool_adoption_by_archetype")
      writeCsv(stackDf,    "tech_stack_by_archetype")
      writeCsv(profileDf,  "profile_by_archetype")
      writeCsv(rwPctFinal, "remote_work_by_archetype")
      writeCsv(mbPctFinal, "main_branch_by_archetype")
      writeCsv(dtPctReady, "primary_role_by_archetype")
      writeCsv(summaryDf,  "summary_by_archetype")

      val countryCsv = countryRows.flatMap { case (arch, top5) =>
        top5.map { case (country, c) => (arch, country, c) }
      }.toSeq.toDF("archetype", "country", "count")
      writeCsv(countryCsv, "top_countries_by_archetype")

      val devTypeCsv = devTypeRows.flatMap { case (arch, top5) =>
        top5.map { case (devType, c) => (arch, devType, c) }
      }.toSeq.toDF("archetype", "dev_role", "count")
      writeCsv(devTypeCsv, "top_devroles_by_archetype")

      // Persist the per-respondent labelled dataset (Parquet keeps schema)
      val parquetPath = s"$outputDir/labelled_developers.parquet"
      df.drop("features_raw", "features")
        .write.mode("overwrite").parquet(parquetPath)
      println(s"  wrote $parquetPath")
    }

    // ── DEEP DIVE: does "AI Power Users earn less" survive controls? ──────
    deepDive(df, spark, outputDir)

    println("\n" + "=" * 70)
    println("  Report complete.")
    println("=" * 70 + "\n")
  }

  /**
   * Tests the headline finding "AI Power Users earn less" against three
   * obvious confounds: experience, country, and developer role.
   *
   * If salary still drops with AI usage WITHIN each control bucket, the
   * effect is real. If it disappears (or flips), it was a confound.
   *
   * Also reports:
   *   - Pearson correlations between AI-use and outcome variables
   *   - A small Spark MLlib LinearRegression with AI usage + experience +
   *     org size as predictors of salary (so we get a coefficient with sign
   *     and magnitude, not just a binned table).
   */
  def deepDive(df: DataFrame, spark: SparkSession, outputDir: String = null): Unit = {

    import spark.implicits._

    println("\n" + "=" * 70)
    println("  DEEP DIVE — DOES \"AI POWER USERS EARN LESS\" SURVIVE CONTROLS?")
    println("=" * 70)

    // Ordinal bins so the gradient is easy to read
    val withBins = df
      .withColumn("ai_usage_bin",
        when(col("ai_product_count") === 0, "0_None")
          .when(col("ai_product_count") <= 2, "1_Light_1to2")
          .when(col("ai_product_count") <= 4, "2_Moderate_3to4")
          .otherwise("3_Heavy_5plus"))
      .withColumn("experience_bin",
        when(col("years_code_pro") < 5,  "Junior_0to4")
          .when(col("years_code_pro") < 11, "Mid_5to10")
          .otherwise("Senior_11plus"))
      .withColumn("primary_role",
        when(col("DevType").isNotNull,
          split(col("DevType"), ";").getItem(0))
          .otherwise(lit("Unknown")))
      .cache()

    // ── 9. Raw salary gradient by AI usage bin ────────────────────────────
    println("\n── 9. RAW SALARY GRADIENT BY AI USAGE BIN ───────────────────────────")
    val salaryByAi = withBins.groupBy("ai_usage_bin")
      .agg(
        count("*").as("n"),
        round(mean("salary_usd"),   0).as("mean_salary"),
        round(expr("percentile(salary_usd, 0.5)"),  0).as("median_salary"),
        round(mean("years_code_pro"), 1).as("mean_years_exp"),
        round(mean("job_sat_int"),    2).as("mean_job_sat"),
        round(mean("org_size_int"),   2).as("mean_org_size")
      )
      .orderBy("ai_usage_bin")
    salaryByAi.show(truncate = false)

    // ── 10. Salary by (experience × AI usage) — controls for seniority ────
    println("── 10. SALARY BY EXPERIENCE × AI USAGE  (controls for seniority) ────")
    val salaryByExpAi = withBins
      .groupBy("experience_bin")
      .pivot("ai_usage_bin", Seq("0_None", "1_Light_1to2", "2_Moderate_3to4", "3_Heavy_5plus"))
      .agg(round(mean("salary_usd"), 0))
      .orderBy("experience_bin")
    salaryByExpAi.show(truncate = false)

    println("    Cell counts (so you can judge sample size of each bucket):")
    val countsByExpAi = withBins
      .groupBy("experience_bin")
      .pivot("ai_usage_bin", Seq("0_None", "1_Light_1to2", "2_Moderate_3to4", "3_Heavy_5plus"))
      .count()
      .orderBy("experience_bin")
    countsByExpAi.show(truncate = false)

    // ── 11. Salary by (top-5 country × AI usage) — controls for geography ─
    println("── 11. SALARY BY COUNTRY × AI USAGE  (controls for geography) ───────")
    val topCountries = withBins
      .filter(col("Country").isNotNull)
      .groupBy("Country").count()
      .orderBy(desc("count")).limit(5)
      .select("Country").as[String].collect().toSeq
    println(s"    Top-5 countries used: ${topCountries.mkString(", ")}")
    val salaryByCountryAi = withBins
      .filter(col("Country").isin(topCountries: _*))
      .groupBy("Country")
      .pivot("ai_usage_bin", Seq("0_None", "1_Light_1to2", "2_Moderate_3to4", "3_Heavy_5plus"))
      .agg(round(mean("salary_usd"), 0))
      .orderBy(desc("Country"))
    salaryByCountryAi.show(truncate = false)

    // ── 12. Salary by (top dev role × AI usage) — controls for role ───────
    println("── 12. SALARY BY DEV ROLE × AI USAGE  (controls for role mix) ───────")
    val topRoles = withBins
      .groupBy("primary_role").count()
      .orderBy(desc("count")).limit(8)
      .select("primary_role").as[String].collect().toSeq
    val salaryByRoleAi = withBins
      .filter(col("primary_role").isin(topRoles: _*))
      .groupBy("primary_role")
      .pivot("ai_usage_bin", Seq("0_None", "1_Light_1to2", "2_Moderate_3to4", "3_Heavy_5plus"))
      .agg(round(mean("salary_usd"), 0))
      .orderBy("primary_role")
    salaryByRoleAi.show(truncate = false)

    // ── 13. Pearson correlations among numeric variables ──────────────────
    println("── 13. PEARSON CORRELATIONS  (signal direction & strength) ──────────")
    val numericCols = Seq(
      "salary_usd", "ai_product_count", "ai_task_count", "ai_ben_count",
      "years_code_pro", "org_size_int", "job_sat_int",
      "ai_acc_int", "ai_threat_int", "ai_sent_int"
    )
    val pairs = for {
      i <- numericCols.indices
      j <- (i + 1) until numericCols.length
    } yield (numericCols(i), numericCols(j))

    val corrRows = pairs.map { case (a, b) =>
      val c = withBins.stat.corr(a, b)
      (a, b, math.round(c * 1000.0) / 1000.0)
    }

    println(f"  ${"variable A"}%-22s  ${"variable B"}%-22s  ${"corr"}%6s")
    println("  " + "-" * 56)
    corrRows
      .filter { case (a, _, _) => a == "salary_usd" }     // headline first
      .sortBy { case (_, _, c) => -math.abs(c) }
      .foreach { case (a, b, c) =>
        println(f"  $a%-22s  $b%-22s  $c%+6.3f")
      }
    println("  " + "-" * 56)
    corrRows
      .filter { case (a, _, _) => a != "salary_usd" }
      .sortBy { case (_, _, c) => -math.abs(c) }
      .foreach { case (a, b, c) =>
        println(f"  $a%-22s  $b%-22s  $c%+6.3f")
      }

    // ── 14. Linear regression: salary ~ AI usage + controls ───────────────
    println("\n── 14. LINEAR REGRESSION  salary ~ ai_use + experience + org_size ──")
    val regCols = Array(
      "ai_product_count",
      "ai_task_count",
      "years_code_pro",
      "org_size_int"
    )
    val regAssembler = new VectorAssembler()
      .setInputCols(regCols)
      .setOutputCol("reg_features")
      .setHandleInvalid("skip")

    val regDf = regAssembler.transform(withBins)
      .filter(col("salary_usd").isNotNull)

    val lr = new LinearRegression()
      .setFeaturesCol("reg_features")
      .setLabelCol("salary_usd")
      .setPredictionCol("salary_predicted")   // avoid clash with K-Means' "prediction"
      .setMaxIter(40)
      .setRegParam(0.0)
      .setStandardization(true)

    val lrModel = lr.fit(regDf)
    val coefs   = lrModel.coefficients.toArray
    val sm      = lrModel.summary

    println(f"\n  Intercept       : ${lrModel.intercept}%+12.2f USD")
    println(f"  ${"feature"}%-20s  ${"coefficient"}%14s  ${"std error"}%12s  ${"t-stat"}%8s  ${"p-value"}%8s")
    println("  " + "-" * 72)
    val stdErrs = sm.coefficientStandardErrors
    val tStats  = sm.tValues
    val pVals   = sm.pValues
    for (i <- regCols.indices) {
      println(f"  ${regCols(i)}%-20s  ${coefs(i)}%+14.2f  ${stdErrs(i)}%12.2f  ${tStats(i)}%+8.2f  ${pVals(i)}%8.4f")
    }
    println("  " + "-" * 72)
    println(f"  R²              : ${sm.r2}%.4f")
    println(f"  Adj R²          : ${sm.r2adj}%.4f")
    println(f"  RMSE            : ${sm.rootMeanSquaredError}%.0f USD")
    println(f"  Observations    : ${sm.numInstances}")

    val aiCoef = coefs(0)
    val aiP    = pVals(0)
    println("\n  Headline read:")
    if (aiCoef < 0 && aiP < 0.01) {
      println(f"    ✓ ai_product_count coefficient is NEGATIVE ($aiCoef%+.0f USD per extra product)")
      println(f"      and statistically significant (p=$aiP%.4f) AFTER controlling for")
      println( "      years of experience and org size.  The thesis IS supported.")
    } else if (aiCoef < 0 && aiP < 0.05) {
      println(f"    ~ ai_product_count coefficient is negative ($aiCoef%+.0f) and significant at p<0.05")
      println( "      but not at p<0.01.  Weak support.")
    } else if (aiCoef < 0) {
      println(f"    × ai_product_count coefficient is negative ($aiCoef%+.0f) but NOT significant (p=$aiP%.3f).")
      println( "      Cannot reject the null after controls.")
    } else {
      println(f"    × ai_product_count coefficient is POSITIVE ($aiCoef%+.0f) — thesis NOT supported.")
    }

    // ── Persist deep-dive tables ──────────────────────────────────────────
    if (outputDir != null) {
      def writeCsv(d: DataFrame, name: String): Unit = {
        val path = s"$outputDir/$name"
        d.coalesce(1).write.mode("overwrite").option("header", "true").csv(path)
        println(s"  wrote $path")
      }
      println("\n── PERSISTING DEEP-DIVE OUTPUTS ─────────────────────────────────────")
      writeCsv(salaryByAi,        "deep_salary_by_ai_usage")
      writeCsv(salaryByExpAi,     "deep_salary_by_exp_x_ai")
      writeCsv(countsByExpAi,     "deep_counts_by_exp_x_ai")
      writeCsv(salaryByCountryAi, "deep_salary_by_country_x_ai")
      writeCsv(salaryByRoleAi,    "deep_salary_by_role_x_ai")

      val corrCsv = corrRows.toDF("var_a", "var_b", "pearson_corr")
      writeCsv(corrCsv, "deep_correlations")

      val regCsv = (regCols.indices).map { i =>
        (regCols(i), coefs(i), stdErrs(i), tStats(i), pVals(i))
      }.toDF("feature", "coefficient", "std_error", "t_stat", "p_value")
      writeCsv(regCsv, "deep_regression_coefficients")
    }

    withBins.unpersist()
  }
}
