import requests
import os
import tempfile
import mlflow
import zipfile
import getpass
import traceback
import click
import pandas as pd
from dstools.spark import init_spark2

@click.command()
@click.option("--ratings_csv", help='file with the data')
@click.option("--max_row_limit", default=10000, help="count of data")

def etl_data(ratings_csv, max_row_limit):
    with mlflow.start_run() as mlrun:
        save_dir = tempfile.mkdtemp()
        ratings_parquet_dir = os.path.join(save_dir, 'ratings.parquet')
        
        
        spark = init_spark2({'appName': 'etl_data',
                     'master': 'local[20]'})
        
        ratings_df = (spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv('file:///'+ ratings_csv).drop('timestamp'))
        
        ratings_df.show()
        if max_row_limit != -1:
            ratings_df = ratings_df.limit(max_row_limit)
        ratings_df.write.parquet('file:///'+ ratings_parquet_dir)
        mlflow.log_artifact(ratings_parquet_dir, 'ratings_parquet_dir')
            
if __name__ == '__main__':
    etl_data()