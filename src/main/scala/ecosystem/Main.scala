package ecosystem

import org.apache.spark.sql.SparkSession

object Main extends App {

  // Hadoop home only matters on Windows; harmless on Linux/macOS
  if (sys.props("os.name").toLowerCase.contains("win"))
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
  println(s"Java version:  ${System.getProperty("java.version")}")

  // ── Step 1: Load and clean ───────────────────────────────────────────────
  val csvPath = sys.env.getOrElse("SURVEY_CSV", "data/dataset.csv")

  val rawDf = spark.read
    .option("header",      "true")
    .option("inferSchema", "true")
    .option("nullValue",   "NA")
    .option("mode",        "DROPMALFORMED")
    .option("multiLine",   "true")
    .option("escape",      "\"")
    .csv(csvPath)

  println(s"Raw row count: ${rawDf.count()}")
  val cleanDf = Cleaning.clean(rawDf).cache()
  println(s"Clean row count: ${cleanDf.count()}")

  // ── Step 2: Feature engineering ──────────────────────────────────────────
  val featureDf = Features.buildFeatures(cleanDf).cache()

  // ── Step 3: Find optimal k ───────────────────────────────────────────────
  OptimalK.findOptimalK(featureDf, kRange = 2 to 8)

  // ── Step 4: Train final model ────────────────────────────────────────────
  val k           = sys.env.getOrElse("CLUSTER_K", "3").toInt
  val clusteredDf = Clustering.trainFinal(featureDf, k)

  // ── Step 5: Impact analysis + persist outputs ───────────────────────────
  val outputDir = sys.env.getOrElse("OUTPUT_DIR", "output/report")
  Analysis.impactReport(clusteredDf, spark, outputDir)

  spark.stop()
  println("\nPipeline complete.")
}
