name         := "AIEcosystemClustering"
version      := "0.1.0"
scalaVersion := "2.12.17"

// Spark 3.2.x is the last series with full Java 8 support
val sparkVersion = "3.2.4"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"  % sparkVersion,
  "org.apache.spark" %% "spark-sql"   % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion
)

ThisBuild / evictionErrorLevel := Level.Warn

Compile / run / fork := true

javaOptions ++= Seq(
  "-Dio.netty.tryReflectionSetAccessible=true",
  s"-Dhadoop.home.dir=C:\\hadoop"
)