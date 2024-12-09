from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
import sys
import os

def prepare_data(input_data):
    # Clean column names by removing quotes
    input_data = input_data.toDF(*[col.replace('"', '') for col in input_data.columns])

    label_column = 'quality'

    # Convert the target column to a numeric label
    indexer = StringIndexer(inputCol=label_column, outputCol="label").fit(input_data)
    indexed_data = indexer.transform(input_data)

    # Select feature columns, excluding the label column
    feature_columns = [col for col in input_data.columns if col != label_column]

    # Assemble features into a single vector column
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_data = assembler.transform(indexed_data)

    return assembled_data

def predict_using_model(test_data_path, model_path):
    # Initialize Spark session with a specific app name
    spark = SparkSession.builder.appName("WineQualityModelPrediction").getOrCreate()

    # Load the test dataset
    test_raw_data = spark.read.csv(test_data_path, header=True, inferSchema=True, sep=";")

    # Prepare the test data
    test_data = prepare_data(test_raw_data)

    # Load the trained model
    trained_model = PipelineModel.load(os.path.join(os.getcwd(), model_path))

    # Generate predictions
    predictions = trained_model.transform(test_data)

    # Set up evaluator for model evaluation
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    # Evaluate predictions
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

    # Print evaluation metrics
    print(f"Test Accuracy: {accuracy}")
    print(f"Test F1 Score: {f1_score}")

    # Stop the Spark session
    spark.stop()

if _name_ == "_main_":
    # Check for the required number of command-line arguments
    if len(sys.argv) != 3:
        print("Usage: main.py <test_data_path> <model_path>")
        sys.exit(1)

    test_data_path = sys.argv[1]
    model_path = sys.argv[2]

    predict_using_model(test_data_path, model_path)
