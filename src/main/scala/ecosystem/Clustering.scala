package divide

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

object Clustering {

  /**
   * Trains the final K-Means model with the chosen k.
   *
   * Returns the input DataFrame augmented with:
   *   - prediction     : raw cluster id (0 to k-1)
   *   - archetype      : human-readable label derived from cluster centroid analysis
   *
   * Archetype labelling logic:
   *   After clustering, we inspect each centroid's ai_select_int and ai_tool_count
   *   (positions 0 and 1 in the feature vector) to assign a meaningful label.
   *   Adjust the thresholds below if your data distribution differs.
   */
  def trainFinal(df: DataFrame, k: Int): DataFrame = {

    println(s"\nTraining final K-Means model with k=$k ...")

    val kmeans = new KMeans()
      .setK(k)
      .setSeed(42L)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMaxIter(40)     // more iterations for final model

    val model       = kmeans.fit(df)
    val predictions = model.transform(df)

    // ── Print centroid summary ─────────────────────────────────────────────
    println("\nCluster centroids (first 7 features: ai_select, tool_count, ai_acc, ai_threat, ai_ben, years_exp, org_size):")
    println(f"${"Cluster"}%10s  ${"ai_select"}%10s  ${"tool_count"}%10s  ${"ai_acc"}%8s  ${"years_exp"}%10s  ${"org_size"}%10s")
    println("-" * 70)

    model.clusterCenters.zipWithIndex.foreach { case (center, idx) =>
      // center is a normalised z-score vector; raw magnitudes indicate direction
      println(f"  C$idx%2d       ${center(0)}%10.3f  ${center(1)}%10.3f  ${center(2)}%8.3f  ${center(5)}%10.3f  ${center(6)}%10.3f")
    }

    // ── Silhouette score for final model ───────────────────────────────────
    val evaluator = new ClusteringEvaluator()
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMetricName("silhouette")

    val silhouette = evaluator.evaluate(predictions)
    println(f"\nFinal model Silhouette Score: $silhouette%.4f")

    // ── Assign archetype labels ────────────────────────────────────────────
    //   We derive labels from each centroid's z-score on the two most
    //   discriminating features: ai_select_int (pos 0) and ai_tool_count (pos 1).
    //   High positive z-score = above average adoption.
    //
    //   These thresholds work for k=4. If you change k, re-run and relabel.
    val clusterLabels: Map[Int, String] = model.clusterCenters.zipWithIndex.map {
      case (center, idx) =>
        val adoption  = center(0)   // z-score of ai_select_int
        val toolCount = center(1)   // z-score of ai_tool_count
        val trust     = center(2)   // z-score of ai_acc_int
        val yearsExp  = center(5)   // z-score of years_code_pro

        val label = (adoption, toolCount) match {
          case (a, t) if a > 0.4 && t > 0.4  => "AI Power User"
          case (a, t) if a > 0.0 && t >= 0.0 => "AI Experimenter"
          case (a, _) if a < -0.2 && yearsExp > 0.3 => "Experienced Skeptic"
          case _                               => "Traditional Engineer"
        }
        idx -> label
    }.toMap

    println("\nDerived archetype labels:")
    clusterLabels.toSeq.sortBy(_._1).foreach { case (id, label) =>
      println(s"  Cluster $id → $label")
    }

    // ── Apply labels via UDF ──────────────────────────────────────────────
    val labelUdf = udf((clusterId: Int) => clusterLabels.getOrElse(clusterId, s"Cluster $clusterId"))

    val labelled = predictions.withColumn("archetype", labelUdf(col("prediction")))

    labelled
  }
}