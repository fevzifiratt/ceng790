package divide

import org.apache.spark.sql.SparkSession

object Main extends App {

  System.setProperty("hadoop.home.dir", "C:\\hadoop")

  val spark = SparkSession.builder()
    .appName("AI Divide - Developer Archetype Clustering")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.warehouse.dir", "output/spark-warehouse")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

  import spark.implicits._

  println(s"Spark version: ${spark.version}")
  println(s"Java version: ${System.getProperty("java.version")}")

  // ── Step 1: Load and clean ───────────────────────────────────────────────
  val rawDf = spark.read
    .option("header",      "true")
    .option("inferSchema", "true")
    .option("nullValue",   "NA")
    .option("mode",        "DROPMALFORMED")
    .csv("data/survey_results_public.csv")

  println(s"Raw row count: ${rawDf.count()}")
  val cleanDf = Cleaning.clean(rawDf)
  println(s"Clean row count: ${cleanDf.count()}")

  // ── Step 2: Feature engineering ──────────────────────────────────────────
  //   Expands semicolon-separated AI tool & language columns into
  //   binary indicator columns, then assembles + scales the feature vector
  val featureDf = Features.buildFeatures(cleanDf)
  println(s"Feature columns: ${featureDf.columns.mkString(", ")}")

  // ── Step 3: Find optimal k ───────────────────────────────────────────────
  //   Prints WSSSE for each k so you can apply the Elbow Method manually
  OptimalK.findOptimalK(featureDf, kRange = 2 to 8)

  // ── Step 4: Train final model (set k after inspecting Step 3 output) ─────
  val k           = 4   // change this after reviewing elbow output
  val clusteredDf = Clustering.trainFinal(featureDf, k)

  // ── Step 5: Impact analysis ───────────────────────────────────────────────
  Analysis.impactReport(clusteredDf, spark)

  spark.stop()
  println("\nPipeline complete.")
}