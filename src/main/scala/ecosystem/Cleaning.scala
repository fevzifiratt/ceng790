package divide

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object Cleaning {

  /**
   * Cleans the raw Stack Overflow survey CSV.
   *
   * Steps:
   *  1. Select only the columns we actually need
   *  2. Drop rows where critical outcome variables are null
   *  3. Cast salary to Double (it comes in as String due to commas / NAs)
   *  4. Cast YearsCodePro to Int  (survey uses "More than 50 years" / "Less than 1 year" special strings)
   *  5. Encode OrgSize as an ordinal integer (1 = solo, 7 = 10k+)
   *  6. Drop any remaining nulls in clustering features
   */
  def clean(df: DataFrame): DataFrame = {

    import df.sparkSession.implicits._

    // ── 1. Select relevant columns ──────────────────────────────────────────
    val selected = df.select(
      col("ResponseId"),
      col("AISelect"),
      col("AISent"),
      col("AIAcc"),
      col("AIComplex"),
      col("AIThreat"),
      col("AIBen"),
      col("AIToolCurrentlyUsing"),
      col("LanguageHaveWorkedWith"),
      col("OrgSize"),
      col("YearsCodePro"),
      col("ConvertedCompYearly"),
      col("JobSat"),
      col("Country"),
      col("DevType"),
      col("Employment")
    )

    // ── 2. Keep only full-time employed developers ──────────────────────────
    val employed = selected.filter(
      col("Employment").contains("Employed, full-time")
    )

    // ── 3. Drop rows with null outcome variables ────────────────────────────
    val nonNullOutcomes = employed
      .filter(col("ConvertedCompYearly").isNotNull)
      .filter(col("JobSat").isNotNull)
      .filter(col("AISelect").isNotNull)

    // ── 4. Cast salary ──────────────────────────────────────────────────────
    val withSalary = nonNullOutcomes
      .withColumn("salary_usd",
        col("ConvertedCompYearly").cast("double"))
      .filter(col("salary_usd") > 5000)      // remove implausible values
      .filter(col("salary_usd") < 1000000)   // remove outliers

    // ── 5. Normalise YearsCodePro ───────────────────────────────────────────
    //   Survey has "More than 50 years" and "Less than 1 year" as strings
    val withYears = withSalary
      .withColumn("years_code_pro",
        when(col("YearsCodePro") === "More than 50 years", 50)
          .when(col("YearsCodePro") === "Less than 1 year",  0)
          .otherwise(col("YearsCodePro").cast("int"))
      )
      .filter(col("years_code_pro").isNotNull)

    // ── 6. Encode OrgSize as ordinal integer ────────────────────────────────
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

    // ── 7. Encode JobSat as numeric ─────────────────────────────────────────
    val withJobSat = withOrgSize
      .withColumn("job_sat_int",
        when(col("JobSat") === "Very satisfied",       5)
          .when(col("JobSat") === "Slightly satisfied",  4)
          .when(col("JobSat") === "Neither satisfied nor dissatisfied", 3)
          .when(col("JobSat") === "Slightly dissatisfied", 2)
          .when(col("JobSat") === "Very dissatisfied",    1)
          .otherwise(null)
      )
      .filter(col("job_sat_int").isNotNull)

    // ── 8. Encode AISelect as numeric ───────────────────────────────────────
    val withAISelect = withJobSat
      .withColumn("ai_select_int",
        when(col("AISelect") === "I use it myself",              3)
          .when(col("AISelect") === "I use it at work",           2)  // older survey wording
          .when(col("AISelect") === "No, but I plan to soon",     1)
          .when(col("AISelect") === "No, and I don't plan to",    0)
          .otherwise(null)
      )
      .filter(col("ai_select_int").isNotNull)

    // ── 9. Encode AI sentiment columns ──────────────────────────────────────
    val withSentiment = withAISelect
      .withColumn("ai_acc_int",
        when(col("AIAcc") === "Highly trust",     4)
          .when(col("AIAcc") === "Somewhat trust",  3)
          .when(col("AIAcc") === "Indifferent",     2)
          .when(col("AIAcc") === "Somewhat distrust", 1)
          .when(col("AIAcc") === "Highly distrust", 0)
          .otherwise(2)   // default to indifferent if null
      )
      .withColumn("ai_threat_int",
        when(col("AIThreat") === "Yes", 1)
          .when(col("AIThreat") === "No",  0)
          .otherwise(0)
      )
      .withColumn("ai_ben_int",
        when(col("AIBen") === "Mostly positive", 2)
          .when(col("AIBen") === "Indifferent",    1)
          .when(col("AIBen") === "Mostly negative", 0)
          .otherwise(1)
      )

    withSentiment
  }
}