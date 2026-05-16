package divide

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
 * Cleans the raw Stack Overflow 2024 Developer Survey CSV.
 *
 * IMPORTANT (2024 schema notes):
 *   - The column with AI tool products is `AISearchDevHaveWorkedWith`,
 *     NOT `AIToolCurrentlyUsing`.
 *   - The column `AIToolCurrently Using` (with a space!) lists the
 *     ACTIVITIES the developer uses AI for (writing code, debugging, …).
 *   - `JobSat` is a numeric 0–10 score, not text labels.
 *   - `AIBen` is a multi-value semicolon list of perceived benefits,
 *     not a single sentiment label.
 */
object Cleaning {

  def clean(df: DataFrame): DataFrame = {

    // ── 1. Select relevant columns ───────────────────────────────────────────
    val selected = df.select(
      col("ResponseId"),
      col("AISelect"),
      col("AISent"),
      col("AIAcc"),
      col("AIComplex"),
      col("AIThreat"),
      col("AIBen"),
      col("AIToolCurrently Using").as("AIToolActivities"),
      col("AISearchDevHaveWorkedWith").as("AIProducts"),
      col("LanguageHaveWorkedWith"),
      col("OrgSize"),
      col("YearsCodePro"),
      col("ConvertedCompYearly"),
      col("JobSat"),
      col("Country"),
      col("DevType"),
      col("Employment"),
      col("RemoteWork"),
      col("MainBranch")
    )

    // ── 2. Keep only full-time employed developers ──────────────────────────
    val employed = selected.filter(
      col("Employment").isNotNull &&
        col("Employment").contains("Employed, full-time")
    )

    // ── 3. Drop rows with null critical outcomes ────────────────────────────
    val nonNullOutcomes = employed
      .filter(col("ConvertedCompYearly").isNotNull)
      .filter(col("JobSat").isNotNull)
      .filter(col("AISelect").isNotNull)

    // ── 4. Cast salary to Double, remove outliers ───────────────────────────
    val withSalary = nonNullOutcomes
      .withColumn("salary_usd", col("ConvertedCompYearly").cast("double"))
      .filter(col("salary_usd").isNotNull)
      .filter(col("salary_usd") > 5000)
      .filter(col("salary_usd") < 1000000)

    // ── 5. Normalise YearsCodePro ───────────────────────────────────────────
    //   2024 file still uses "More than 50 years" / "Less than 1 year"
    val withYears = withSalary
      .withColumn("years_code_pro",
        when(col("YearsCodePro") === "More than 50 years", 50)
          .when(col("YearsCodePro") === "Less than 1 year",  0)
          .otherwise(col("YearsCodePro").cast("int"))
      )
      .filter(col("years_code_pro").isNotNull)

    // ── 6. Encode OrgSize as ordinal integer (1–9) ─────────────────────────
    //   "I don't know" → null → dropped
    val withOrgSize = withYears
      .withColumn("org_size_int",
        when(col("OrgSize") === "Just me - I am a freelancer, sole proprietor, etc.", 1)
          .when(col("OrgSize") === "2 to 9 employees",          2)
          .when(col("OrgSize") === "10 to 19 employees",        3)
          .when(col("OrgSize") === "20 to 99 employees",        4)
          .when(col("OrgSize") === "100 to 499 employees",      5)
          .when(col("OrgSize") === "500 to 999 employees",      6)
          .when(col("OrgSize") === "1,000 to 4,999 employees",  7)
          .when(col("OrgSize") === "5,000 to 9,999 employees",  8)
          .when(col("OrgSize") === "10,000 or more employees",  9)
          .otherwise(null)
      )
      .filter(col("org_size_int").isNotNull)

    // ── 7. Cast JobSat (0–10 numeric in 2024) ──────────────────────────────
    val withJobSat = withOrgSize
      .withColumn("job_sat_int", col("JobSat").cast("int"))
      .filter(col("job_sat_int").isNotNull)
      .filter(col("job_sat_int") >= 0 && col("job_sat_int") <= 10)

    // ── 8. Encode AISelect as numeric (2024 wording) ───────────────────────
    val withAISelect = withJobSat
      .withColumn("ai_select_int",
        when(col("AISelect") === "Yes",                       2)
          .when(col("AISelect") === "No, but I plan to soon", 1)
          .when(col("AISelect") === "No, and I don't plan to", 0)
          .otherwise(null)
      )
      .filter(col("ai_select_int").isNotNull)

    // ── 9. Encode AI sentiment / trust / threat ────────────────────────────
    val withSentiment = withAISelect
      .withColumn("ai_acc_int",
        when(col("AIAcc") === "Highly trust",                4)
          .when(col("AIAcc") === "Somewhat trust",           3)
          .when(col("AIAcc") === "Neither trust nor distrust", 2)
          .when(col("AIAcc") === "Somewhat distrust",        1)
          .when(col("AIAcc") === "Highly distrust",          0)
          .otherwise(2)
      )
      .withColumn("ai_threat_int",
        when(col("AIThreat") === "Yes",            2)
          .when(col("AIThreat") === "I'm not sure", 1)
          .when(col("AIThreat") === "No",          0)
          .otherwise(0)
      )
      .withColumn("ai_sent_int",
        when(col("AISent") === "Very favorable",   4)
          .when(col("AISent") === "Favorable",     3)
          .when(col("AISent") === "Indifferent",   2)
          .when(col("AISent") === "Unsure",        2)
          .when(col("AISent") === "Unfavorable",   1)
          .when(col("AISent") === "Very unfavorable", 0)
          .otherwise(2)
      )

    // ── 10. AIBen → number of perceived benefits (0..N) ────────────────────
    //   Multi-value semicolon list. Drop "Other (please specify):" noise.
    val withBenefit = withSentiment
      .withColumn("ai_ben_count",
        when(col("AIBen").isNull || col("AIBen") === "NA", lit(0))
          .otherwise(
            size(
              expr(
                "filter(split(AIBen, ';'), x -> x != 'Other (please specify):' AND x != '')"
              )
            )
          )
      )

    withBenefit
  }
}
