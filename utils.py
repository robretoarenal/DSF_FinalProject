import os
import pathlib

class DataFetch(object):
    """
    Class for "Get the Data" in the notebook"
    """

    def __init__(self):
        self.DOWNLOAD_ROOT = "https://raw.githubusercontent.com/vfp1/bts-dsf-2020/main/"
        self.HOUSING_PATH = os.path.join(str(pathlib.Path().absolute()), "housing")
        self.HOUSING_URL = self.DOWNLOAD_ROOT + "data/housing.tgz"
