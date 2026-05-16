package ecosystem

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}

/**
 * Builds the feature vector for K-Means clustering.
 *
 * Feature groups:
 *   - Numeric / ordinal: ai_select_int, ai_acc_int, ai_threat_int,
 *                        ai_sent_int, ai_ben_count,
 *                        ai_product_count, ai_task_count,
 *                        years_code_pro, org_size_int
 *   - Binary indicators: uses_<product>  (top AI products in 2024 survey)
 *   - Binary indicators: lang_<language> (top languages in 2024 survey)
 *
 * Everything is z-score normalised so binary 0/1 features don't get
 * dominated by years_code_pro (0–50) or product counts.
 */
object Features {

  // ── Top AI products in the 2024 Stack Overflow survey ───────────────────
  //   Pulled from the AISearchDevHaveWorkedWith column. Note: "Cursor" is
  //   NOT in this list — it wasn't an answer option in the 2024 survey.
  val AI_TOOLS: Seq[String] = Seq(
    "ChatGPT",
    "GitHub Copilot",
    "Google Gemini",
    "Bing AI",
    "Visual Studio Intellicode",
    "Claude",
    "Codeium",
    "Perplexity AI",
    "Tabnine",
    "Amazon Q"
  )

  // ── Top languages in the 2024 survey (LanguageHaveWorkedWith) ──────────
  val LANGUAGES: Seq[String] = Seq(
    "Python",
    "JavaScript",
    "TypeScript",
    "Java",
    "C#",
    "C++",
    "Go",
    "Rust",
    "Kotlin",
    "SQL",
    "HTML/CSS",
    "Bash/Shell (all shells)"
  )

  /** Convert a label like "GitHub Copilot" or "C#" to a unique safe column suffix. */
  def safeName(s: String): String =
    s.toLowerCase
      .replace("c++", "cpp")
      .replace("c#",  "csharp")
      .replace("f#",  "fsharp")
      .replace(".",   "_dot_")
      .replaceAll("[^a-z0-9]", "_")
      .replaceAll("_+", "_")
      .stripPrefix("_")
      .stripSuffix("_")

  def buildFeatures(df: DataFrame): DataFrame = {

    // ── 1. Counts derived from the two multi-value AI columns ──────────────
    val withCounts = df
      .withColumn("ai_product_count",
        when(col("AIProducts").isNull || col("AIProducts") === "NA", lit(0))
          .otherwise(size(split(col("AIProducts"), ";")))
      )
      .withColumn("ai_task_count",
        when(col("AIToolActivities").isNull || col("AIToolActivities") === "NA", lit(0))
          .otherwise(
            size(
              expr("filter(split(AIToolActivities, ';'), x -> x != 'Other (please specify):' AND x != '')")
            )
          )
      )

    // ── 2. Per-product binary indicators ───────────────────────────────────
    val withAIBinary = AI_TOOLS.foldLeft(withCounts) { (acc, tool) =>
      val colName = "uses_" + safeName(tool)
      acc.withColumn(colName,
        when(col("AIProducts").isNotNull &&
             array_contains(split(col("AIProducts"), ";"), lit(tool)), 1)
          .otherwise(0)
      )
    }

    // ── 3. Per-language binary indicators ──────────────────────────────────
    val withLangBinary = LANGUAGES.foldLeft(withAIBinary) { (acc, lang) =>
      val colName = "lang_" + safeName(lang)
      acc.withColumn(colName,
        when(col("LanguageHaveWorkedWith").isNotNull &&
             array_contains(split(col("LanguageHaveWorkedWith"), ";"), lit(lang)), 1)
          .otherwise(0)
      )
    }

    // ── 4. Assemble feature vector ─────────────────────────────────────────
    // NOTE: Per-tool (uses_*) and per-language (lang_*) binary indicators are
    // intentionally KEPT in the dataframe (the report still uses them) but
    // EXCLUDED from the clustering vector. Including them dominated the
    // distance metric and produced two near-identical "Power User" clusters
    // split only by Tabnine vs Codeium loyalty.
    val baseCols = Array(
      "ai_select_int",
      "ai_product_count",
      "ai_task_count",
      "ai_acc_int",
      "ai_threat_int",
      "ai_sent_int",
      "ai_ben_count",
      "years_code_pro",
      "org_size_int"
    )

    val allFeatureCols: Array[String] = baseCols

    val assembler = new VectorAssembler()
      .setInputCols(allFeatureCols)
      .setOutputCol("features_raw")
      .setHandleInvalid("skip")

    val assembled = assembler.transform(withLangBinary)

    // ── 5. Z-score normalisation ───────────────────────────────────────────
    val scaler = new StandardScaler()
      .setInputCol("features_raw")
      .setOutputCol("features")
      .setWithMean(true)
      .setWithStd(true)

    val scalerModel = scaler.fit(assembled)
    val scaled      = scalerModel.transform(assembled)

    println(s"\nFeature vector size: ${allFeatureCols.length}")
    println(s"Feature columns: ${allFeatureCols.mkString(", ")}")
    println(s"Rows after feature engineering: ${scaled.count()}")

    scaled
  }
}
