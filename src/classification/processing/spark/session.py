from pyspark.sql import SparkSession

def create_spark_session() -> SparkSession:
    spark = SparkSession.builder \
        .appName("Data processing") \
        .master("local[*]") \
        .getOrCreate()

    return spark