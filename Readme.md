# Objective
Use Spark to write a price predictor that is trained from the Boston house dataset.

## Zeppelin Note
Write your code in a Zeppelin note.  Name this note with the following format

    Group-#-Boston-House

Replace # with your group number.

## Upload Github
Export the Zeppelin Note as a JSON file.  Commit this file to a new repository in Github

## Submission to BlackBoard
Submit a document to BlackBoard that contains the following -

* Names of individuals in the group.
* Link to Github project that contains the Zeppelin note as a JSON file.

## How we will Grade
Your notebook will be imported into Zeppelin.  The model will be trained with the Boston house dataset.  A second dataset will be fed into the model for predictions.  We will examine the accuracy of those predictions from an file your note will output to HDFS.

# Boston House Dataset
The boston_house.csv contains a publicly available dataset of house prices
from suburbs around Boston.  This dataset was donated to the public in 1993.

The verification.csv contains three houses that you can feed your model for
price predictions.  This dataset has the MEDV value removed, which is the value we
are trying to predict.

## File Format
The file is a set of comma separated values (csv).  The first line contains the column headers.

## Variables in the Dataset

* CRIM - per capita crime rate by town
* ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
* INDUS -  proportion of non-retail business acres per town
* CHAS - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
* NOX - nitric oxides concentration (parts per 10 million)
* RM - average number of rooms per dwelling
* AGE -  proportion of owner-occupied units built prior to 1940
* DIS - weighted distances to five Boston employment centers
* RAD - index of accessibility to radial highways
* TAX - full-value property-tax rate per $10,000
* PTRATIO - pupil-teacher ratio by town
* B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
* LSTAT - % lower status of the population
* MEDV - Median value of owner-occupied homes in $1000's


# Implementation

## Input
Upload the boston_house.csv and verification.csv datasets to HDFS in the following folder -

    /tmp/boston-data/boston-house.csv


## The model
Below are notes from an implementation in Pyspark.  By no means are they complete.

Load the dataset from HDFS -

    houses = sc.textFile('hdfs://sandbox.hortonworks.com/tmp/boston-data/boston_house.csv')

You might have to remove the header file (unless you are using Dataframes):

    header = houses.first()
    headerless_houses = houses.filter(lambda line: line != header)  

Scale the dataset's features -

    scaler = StandardScaler(withMean=True, withStd=True).fit(features)


Use LinearRegressionWithSGD to train the model.

    model = LinearRegressionWithSGD.train(labeledDataPoints, intercept=True)

## Verification

Load the verification data set and scale the values using the previous scaler.
Push each row into your model.  Below are the predicted rounded results I got -

30.58
24.96
30.71

## Save the predicted data
Save your non-rounded predicted results to /tmp/boston-data/predicted_results

    verify_predictions.saveAsTextFile('hdfs://sandbox.hortonworks.com/tmp/boston-data/predicted_results');

You will need to delete this directory in HDFS when you re-run the note.
