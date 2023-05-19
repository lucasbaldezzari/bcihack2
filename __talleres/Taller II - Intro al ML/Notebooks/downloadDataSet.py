import os
import tarfile
import urllib, urllib.request

ruta_descarga = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
path = os.path.join("datasets", "housing")
housing_url = ruta_descarga + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=housing_url, housing_path=HOUSINpathG_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()