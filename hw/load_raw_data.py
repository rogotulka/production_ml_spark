import requests
import os
import tempfile
import mlflow
import zipfile
import getpass
import traceback
import click



import urllib3
import sys

@click.command()
@click.option("--proxy", help='mts proxy')
@click.option("--url", default='http://files.grouplens.org/datasets/movielens/ml-latest-small.zip', help="the url of zip data file")
def load_raw_data(proxy, url):
    print(proxy)
    user = getpass.getuser()
    password = getpass.getpass()
    
    proxies = {
            "https": "http://" + user + ":" + password + "@" + proxy + ":3131",
            "http": "http://" + user + ":" + password + "@" + proxy + ":3131",
    }
    with mlflow.start_run() as mlrun:
        dir_to_save = tempfile.mkdtemp()
        file_name = url.split('/')[-1]
        zip_path = os.path.join(dir_to_save, file_name)
        ratings_file = os.path.join(dir_to_save, 'ratings.csv') 
        
        r = requests.get(url, proxies = proxies, stream=True, verify=False)
        try:
            with open(zip_path, 'wb') as fd:
                for chunk in r.iter_content(chunk_size=1024):
                    fd.write(chunk)
            with zipfile.ZipFile(zip_path) as z:
                print(z.namelist())
                with open(os.path.join(dir_to_save, 'ratings.csv'), 'wb') as f:
                    f.write(z.read('ml-latest-small/ratings.csv'))
        except IOError as e:
            print("I/O error(%s): %s" % (e.errno, e.strerror))
        except:
            print("Unexpected error: %s" % (traceback.format_exc()))
        mlflow.log_artifact(ratings_file)
            
if __name__ == '__main__':
    load_raw_data()