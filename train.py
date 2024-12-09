from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
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

    # Select relevant feature columns excluding the label column
    feature_columns = [col for col in input_data.columns if col != label_column]

    # Assemble features into a single column
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_data = assembler.transform(indexed_data)

    return assembled_data

def train_model(train_data_path, validation_data_path, output_model):
    # Initialize Spark session with a specific app name
    spark = SparkSession.builder.appName("WineQualityClassifier").getOrCreate()

    # Load training and validation datasets
    train_data_raw = spark.read.csv(train_data_path, header=True, inferSchema=True, sep=";")
    validation_data_raw = spark.read.csv(validation_data_path, header=True, inferSchema=True, sep=";")

    # Prepare data for modeling
    train_data = prepare_data(train_data_raw)
    validation_data = prepare_data(validation_data_raw)

    # Define models for training
    models = [
        ("RandomForestClassifier", RandomForestClassifier(labelCol="label", featuresCol="features")),
        ("LogisticRegression", LogisticRegression(labelCol="label", featuresCol="features")),
        ("DecisionTreeClassifier", DecisionTreeClassifier(labelCol="label", featuresCol="features"))
    ]

    # Create parameter grids
    param_grids = [
        ParamGridBuilder().addGrid(models[0][1].numTrees, [10, 20, 30]).build(),
        ParamGridBuilder().addGrid(models[1][1].maxIter, [10, 20, 30]).build(),
        ParamGridBuilder().addGrid(models[2][1].maxDepth, [5, 10, 15]).build()
    ]

    # Define evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    results = []

    for idx, (model_name, model) in enumerate(models):
        print(f"Training {model_name}")

        # Create pipeline
        pipeline = Pipeline(stages=[model])

        # Cross-validation setup
        cross_validator = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grids[idx],
            evaluator=evaluator,
            numFolds=3
        )

        # Train model using cross-validation
        cv_model = cross_validator.fit(train_data)

        # Make predictions on validation data
        predictions = cv_model.transform(validation_data)

        # Evaluate model performance
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
        f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Recall": recall,
            "F1 Score": f1_score
        })

        # Save the best model for the current model type
        if idx == 0:  # Save only the best model of the first classifier
            best_model = cv_model.bestModel
            best_model_path = os.path.join(os.getcwd(), output_model)
            best_model.save(best_model_path)

    # Output results for all models
    for result in results:
        print(result)

    # Stop Spark session
    spark.stop()

if _name_ == "_main_":
    # Ensure the correct number of command-line arguments
    if len(sys.argv) != 4:
        print("Usage: train.py <train_data_path> <validation_data_path> <output_model>")
        sys.exit(1)

    train_data_path = sys.argv[1]
    validation_data_path = sys.argv[2]
    output_model = sys.argv[3]

    train_model(train_data_path, validation_data_path, output_model)
