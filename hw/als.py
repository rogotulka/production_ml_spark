
import requests
import os
import tempfile

import zipfile
import getpass
import traceback
import click
import pandas as pd
from dstools.spark import init_spark2

@click.command()
@click.option("--ratings_data", help='file with the data')
@click.option("--max_iter", default=5)
@click.option("--reg_param", default=0.1)
@click.option("--rank", default=12)
@click.option("--split_prop", default=0.8)
@click.option("--cold_start_strategy", default='drop')
def als(ratings_data, max_iter, reg_param=0.1, rank = 12, split_prop = 0.8, cold_start_strategy='drop'):
    
    spark = init_spark2({'appName': 'als',
                     'master': 'local[2]'})
    # !!! импорты идут специально после вызова нашей кастомной либы - иначе будет разный pyspark и ошибки   
    import mlflow
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.recommendation import ALS
    from pyspark.ml import Pipeline
    ratings_df = spark.read.parquet('file:///' + ratings_data)
    training, test = ratings_df.randomSplit([0.7, 0.3], seed=11)
    als=ALS(maxIter=5, regParam=reg_param, rank=rank, userCol="userId", itemCol="movieId",
                                     ratingCol="rating", coldStartStrategy=cold_start_strategy, nonnegative=True)
    pipeline = Pipeline(stages=[als]) 
    with mlflow.start_run():        
        model_pipeline=pipeline.fit(training)
        mlflow.spark.log_model(model_pipeline, "model_pipeline_asl")
       
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
        
        predictions = model_pipeline.transform(training)
        rmse = evaluator.evaluate(predictions)
        mlflow.log_metric('train_' + evaluator.getMetricName(), rmse)
        
        predictions = model_pipeline.transform(test)
        rmse = evaluator.evaluate(predictions)
        mlflow.log_metric('test_' + evaluator.getMetricName(), rmse)
        
        
            
if __name__ == '__main__':
    als()