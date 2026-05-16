package divide

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

object OptimalK {

  /**
   * Runs K-Means for each k in kRange and prints:
   *   - WSSSE  (Within-Set Sum of Squared Errors) for the Elbow Method
   *   - Silhouette Score for cluster quality
   *
   * Use the printed output to identify the elbow point and set k in Main.scala.
   *
   * Example output:
   *   k=2  WSSSE=12450.33  Silhouette=0.312
   *   k=3  WSSSE=9834.21   Silhouette=0.401
   *   k=4  WSSSE=7123.88   Silhouette=0.478  ← pick here if elbow
   *   k=5  WSSSE=6987.44   Silhouette=0.461
   */
  def findOptimalK(df: DataFrame, kRange: Range): Unit = {

    println("\n" + "=" * 60)
    println("  ELBOW METHOD — WSSSE & SILHOUETTE SCORES")
    println("=" * 60)
    println(f"${"k"}%5s  ${"WSSSE"}%14s  ${"Silhouette"}%12s")
    println("-" * 60)

    val evaluator = new ClusteringEvaluator()
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMetricName("silhouette")
      .setDistanceMeasure("squaredEuclidean")

    var prevWssse = Double.MaxValue

    kRange.foreach { k =>

      val kmeans = new KMeans()
        .setK(k)
        .setSeed(42L)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setMaxIter(20)

      val model       = kmeans.fit(df)
      val predictions = model.transform(df)

      val wssse      = model.summary.trainingCost
      val silhouette = evaluator.evaluate(predictions)
      val delta      = if (prevWssse == Double.MaxValue) 0.0 else prevWssse - wssse

      println(f"k=$k%2d  WSSSE=$wssse%14.2f  Silhouette=$silhouette%8.4f  ΔCost=$delta%10.2f")

      prevWssse = wssse
    }

    println("=" * 60)
    println("  → Choose k where WSSSE improvement (ΔCost) drops sharply")
    println("  → Silhouette closer to 1.0 = better-separated clusters")
    println("=" * 60 + "\n")
  }
}