from vayu_gnn.dbx.dbx_config import DropboxHelper
from vayu_gnn.data.downloader import Downloader

def execute(dbx_helper:DropboxHelper, download_data:list, urls:dict, download_params:dict, start_date:str, end_date:str, nodes:dict, cities:list):

    #create an instance of the Downloader class
    dl = Downloader(dbx_helper, urls, start_date, end_date, nodes, cities)

    #execute the download process
    for data in download_data:

        params = download_params[data]

        if params is None:
            #execute the download method
            getattr(dl, data)()
        else:
            #execute the download method with parameters
            getattr(dl, data)(**params)

    return None

