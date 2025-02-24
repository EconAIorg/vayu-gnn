import pandas as pd
from urllib.request import urlopen
import zipfile, io
import requests
import logging
from vayu_gnn.dbx.dbx_config import DropboxHelper

class Downloader():

    """
    Attributes
    ----------

    update_period : str
        The period for which the data is being updated, formatted as 'YYYYMM'.

    dbx_helper : object
        An instance of the Dropbox helper class.

    urls : dict
        A dictionary containing the URLs for the data sources.
    """

    def __init__(self, update_period:str, dbx_helper:DropboxHelper, urls:dict):

        self.update_period = update_period #str that should be formatted 'YYYYMM'
        self.dbx_helper = dbx_helper
        self.urls = urls