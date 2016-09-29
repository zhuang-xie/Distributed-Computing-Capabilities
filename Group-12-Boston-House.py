# 95-887 Cloud Computing for Business Homework 4
# Code Group 12
# This homework is done with Spark in standalone mode

from pyspark import SparkContext
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD

# --- Specify the file names and paths here ---
# If left as default, the program will read and save files to the current path where the program is executed
dataTrainPath = "boston_house.csv"
dataVerifyPath = "verification.csv"
resultPath = "predicted_results"

# Create a SparkContext
sc = SparkContext("local", "Group-12-Boston-House")

# Load the datasets
dataTrain = sc.textFile(dataTrainPath)
dataVerify = sc.textFile(dataVerifyPath)

# Remove the header
headerTrain = dataTrain.first()
dataTrainHeaderless = dataTrain.filter(lambda line: line != headerTrain)
headerVerify = dataVerify.first()
dataVerifyHeaderless = dataVerify.filter(lambda line: line != headerVerify)

# Split the datasets by comma
dataTrainSplit = dataTrainHeaderless.map(lambda line: line.split(","))
dataVerifySplit = dataVerifyHeaderless.map(lambda line: line.split(","))

# Create RDDs of label(MEDV) and features separately
featureRDD = dataTrainSplit.map(lambda l: l[0:-1])
labelRDD = dataTrainSplit.map(lambda l: l[-1])

# Scale the dataset's features
scaler = StandardScaler(withMean=True, withStd=True).fit(featureRDD)
featureRDDScaled = scaler.transform(featureRDD)

# Create an RDD of LabeledPoint
labeledPointRDD = labelRDD.zip(featureRDDScaled).map(lambda x: LabeledPoint(x[0], x[1]))

# Train the model
lrm = LinearRegressionWithSGD.train(labeledPointRDD, intercept=True)

# Apply the trained model on the verification dataset
dataVerifyScaled = scaler.transform(dataVerifySplit)
predictionRDD = dataVerifyScaled.map(lambda x: lrm.predict(x))

# Save the results
predictionRDD.saveAsTextFile(resultPath);
print("Results saved to " + resultPath)
