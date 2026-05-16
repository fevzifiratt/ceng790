package divide

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}

object Features {

  // ── AI tools we create binary indicator columns for ──────────────────────
  //   These are the most common tools in the 2024 survey
  val AI_TOOLS: Seq[String] = Seq(
    "ChatGPT",
    "GitHub Copilot",
    "Bing AI",
    "Google Gemini",
    "Tabnine",
    "AWS CodeWhisperer",
    "Codeium",
    "Cursor",
    "Perplexity AI"
  )

  // ── Languages we create binary indicator columns for ────────────────────
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
    "SQL"
  )

  /**
   * Builds the feature vector for clustering.
   *
   * New columns added:
   *   - ai_tool_count        : total number of AI tools the dev uses
   *   - uses_<tool>          : binary indicator per AI tool
   *   - lang_<language>      : binary indicator per language
   *   - features_raw         : assembled (unscaled) vector
   *   - features             : scaled vector (input to K-Means)
   */
  def buildFeatures(df: DataFrame): DataFrame = {

    // ── Step 1: Explode AIToolCurrentlyUsing into binary columns ─────────
    //   Raw value looks like: "ChatGPT;GitHub Copilot;Tabnine"
    val withAIToolCount = df.withColumn(
      "ai_tool_count",
      when(
        col("AIToolCurrentlyUsing").isNull,
        lit(0)
      ).otherwise(
        size(split(col("AIToolCurrentlyUsing"), ";"))
      )
    )

    val withAIBinary = AI_TOOLS.foldLeft(withAIToolCount) { (accDf, tool) =>
      val colName = "uses_" + tool.toLowerCase
        .replaceAll("[^a-z0-9]", "_")   // safe column name
        .replaceAll("_+", "_")
        .stripSuffix("_")

      accDf.withColumn(
        colName,
        when(
          col("AIToolCurrentlyUsing").isNotNull &&
            array_contains(split(col("AIToolCurrentlyUsing"), ";"), lit(tool)),
          1
        ).otherwise(0)
      )
    }

    // ── Step 2: Explode LanguageHaveWorkedWith into binary columns ────────
    val withLangBinary = LANGUAGES.foldLeft(withAIBinary) { (accDf, lang) =>
      val colName = "lang_" + lang.toLowerCase
        .replaceAll("[^a-z0-9]", "_")
        .replaceAll("_+", "_")
        .stripSuffix("_")

      accDf.withColumn(
        colName,
        when(
          col("LanguageHaveWorkedWith").isNotNull &&
            array_contains(split(col("LanguageHaveWorkedWith"), ";"), lit(lang)),
          1
        ).otherwise(0)
      )
    }

    // ── Step 3: Assemble all numeric features into a single vector ────────
    val aiToolCols  = AI_TOOLS.map(t =>
      "uses_" + t.toLowerCase.replaceAll("[^a-z0-9]", "_").replaceAll("_+", "_").stripSuffix("_"))
    val langCols    = LANGUAGES.map(l =>
      "lang_" + l.toLowerCase.replaceAll("[^a-z0-9]", "_").replaceAll("_+", "_").stripSuffix("_"))

    val baseCols = Array(
      "ai_select_int",    // 0-3: AI tool adoption level
      "ai_tool_count",    // 0-N: number of tools used
      "ai_acc_int",       // 0-4: trust in AI accuracy
      "ai_threat_int",    // 0-1: perceives AI as threat
      "ai_ben_int",       // 0-2: perceived AI benefit
      "years_code_pro",   // 0-50: professional experience
      "org_size_int"      // 1-9: company size
    )

    val allFeatureCols = baseCols ++ aiToolCols.toArray ++ langCols.toArray

    val assembler = new VectorAssembler()
      .setInputCols(allFeatureCols)
      .setOutputCol("features_raw")
      .setHandleInvalid("skip")   // skip rows with any remaining nulls

    val assembled = assembler.transform(withLangBinary)

    // ── Step 4: Normalise with StandardScaler ─────────────────────────────
    //   withMean=true, withStd=true → z-score normalisation
    //   Prevents years_code_pro (0-50) from dominating binary (0-1) features
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