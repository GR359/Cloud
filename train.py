from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import sys
import os

def prepare_data(input_data):
    # Load wine quality dataset
    new_columns = [col.replace('"', '') for col in input_data.columns]
    input_data = input_data.toDF(*new_columns)

    label_column = 'quality'

    # 'quality' is a categorical variable, indexing it
    indexer = StringIndexer(inputCol=label_column, outputCol="label")
    input_data = indexer.fit(input_data).transform(input_data)

    # Selecting relevant feature columns
    feature_columns = [col for col in input_data.columns if col != label_column]

    # Create a VectorAssembler to assemble feature columns into a single 'features' column
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Apply the VectorAssembler to the data
    assembled_data = assembler.transform(input_data)

    return assembled_data

def train_model(train_data_path, validation_data_path, output_model):
    # Initialize Spark session
    spark = SparkSession.builder.appName("WineQualityTraining").getOrCreate()

    training_raw_data = spark.read.csv(train_data_path, header=True, inferSchema=True, sep=";")
    validation_raw_data = spark.read.csv(validation_data_path, header=True, inferSchema=True, sep=";")

    # Load training and validation data
    train_data = prepare_data(training_raw_data)
    validation_data = prepare_data(validation_raw_data)

    # Define multiple models
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    lr = LogisticRegression(labelCol="label", featuresCol="features")
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    models = [rf, lr, dt]

    # Create parameter grids for hyperparameter tuning
    paramGrids = [
        ParamGridBuilder().addGrid(rf.numTrees, [10, 20, 30]).build(),
        ParamGridBuilder().addGrid(lr.maxIter, [10, 20, 30]).build(),
        ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 15]).build()
    ]

    # Define evaluator
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    # Train and evaluate multiple models
    results = []

    for i, model in enumerate(models):
        print(f"Training model {i + 1}")

        # Create a pipeline
        pipeline = Pipeline(stages=[model])

        crossvalidate = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrids[i],
            evaluator=evaluator,
            numFolds=3
        )

        # Fit the models
        cvModel = crossvalidate.fit(train_data)

        # Predict on the validation set
        predictions = cvModel.transform(validation_data)

        # Evaluate the model
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
        f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

        results.append({
            "Model": model.__class__.__name__,
            "Accuracy": accuracy,
            "Recall": recall,
            "F1 Score": f1_score
        })

    # Get the best model
    best_model = cvModel.bestModel

    # Save the best model to the specified output path
    output_model_path = os.path.join(os.getcwd(), output_model)
    best_model.save(output_model_path)

    # Display results for all models
    for result in results:
        print(result)

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    # Input Paths for training, validation data, and output model as command-line arguments
    if len(sys.argv) != 4:
        #spark-submit train.py TrainingDataset.csv ValidationDataset.csv winemodel
        sys.exit(1)

    train_data_path = sys.argv[1]
    validation_data_path = sys.argv[2]
    output_model = sys.argv[3]

    train_model(train_data_path, validation_data_path, output_model)