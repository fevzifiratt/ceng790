package divide

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

/**
 * Trains the final K-Means model with the chosen k and assigns
 * human-readable archetype labels to each cluster.
 *
 * Feature index map (set by Features.scala, base block):
 *   0 ai_select_int
 *   1 ai_product_count
 *   2 ai_task_count
 *   3 ai_acc_int
 *   4 ai_threat_int
 *   5 ai_sent_int
 *   6 ai_ben_count
 *   7 years_code_pro
 *   8 org_size_int
 */
object Clustering {

  def trainFinal(df: DataFrame, k: Int): DataFrame = {

    println(s"\nTraining final K-Means model with k=$k ...")

    val kmeans = new KMeans()
      .setK(k)
      .setSeed(42L)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMaxIter(40)

    val model       = kmeans.fit(df)
    val predictions = model.transform(df)

    // ── Centroid summary on the most discriminating features ──────────────
    println(
      "\nCluster centroids (z-scores on key features):"
    )
    println(f"${"Cluster"}%8s  ${"ai_sel"}%8s  ${"prod"}%8s  ${"task"}%8s  " +
            f"${"acc"}%8s  ${"threat"}%8s  ${"sent"}%8s  ${"ben"}%8s  " +
            f"${"years"}%8s  ${"orgSz"}%8s")
    println("-" * 100)

    model.clusterCenters.zipWithIndex.foreach { case (c, idx) =>
      println(f"  C$idx%2d     " +
        f"${c(0)}%8.3f  ${c(1)}%8.3f  ${c(2)}%8.3f  ${c(3)}%8.3f  " +
        f"${c(4)}%8.3f  ${c(5)}%8.3f  ${c(6)}%8.3f  ${c(7)}%8.3f  ${c(8)}%8.3f")
    }

    val evaluator = new ClusteringEvaluator()
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMetricName("silhouette")

    val silhouette = evaluator.evaluate(predictions)
    println(f"\nFinal model Silhouette Score: $silhouette%.4f")

    // ── Archetype labelling: rank clusters by ai_product_count (idx 1) ────
    //   This is generic — works for any k. Lowest AI use → "Traditional",
    //   highest → "Power User", middle ones → "Experimenter" with a rank.
    val centersByAdoption = model.clusterCenters.zipWithIndex
      .sortBy { case (c, _) => c(1) }   // ascending by ai_product_count z-score

    val labels = centersByAdoption.zipWithIndex.map { case ((_, clusterId), rank) =>
      val label = (rank, centersByAdoption.length) match {
        case (0, _)                            => "Traditional Engineer"
        case (r, n) if r == n - 1              => "AI Power User"
        case (r, n) if n == 3 && r == 1        => "AI Experimenter"
        case (r, n) if r == 1                  => "Light AI User"
        case (r, _)                            => s"AI Experimenter (tier $r)"
      }
      clusterId -> label
    }.toMap

    println("\nDerived archetype labels (ranked by AI product use):")
    labels.toSeq.sortBy(_._1).foreach { case (id, label) =>
      println(s"  Cluster $id → $label")
    }

    val labelUdf = udf((clusterId: Int) =>
      labels.getOrElse(clusterId, s"Cluster $clusterId"))

    predictions.withColumn("archetype", labelUdf(col("prediction")))
  }
}
