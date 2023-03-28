from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
# conf = SparkConf().setAppName("appName").setMaster("yarn")
# sc = SparkContext(conf=conf)


# distFile = sc.textFile("/home/jayliu1/dataset/customer_judgment")
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
path = "/home/jayliu1/dataset/customer_judgment"
df = spark.read.csv(path)
# df2 = spark.read.option("delimiter", ";").csv(path)
# df3 = spark.read.option("delimiter", ";").option("header", True).csv(path)
# df4 = spark.read.options(delimiter=";", header=True).csv(path)
df3 = spark.read.option("header", True).csv(path)
df.show()

df3.write.csv("/home/jayliu1/dataset/out")

# 读取文件下所有csv
folderPath = "/home/jayliu1/dataset/out"
df5 = spark.read.csv(folderPath)

