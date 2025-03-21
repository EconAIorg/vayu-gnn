import dropbox
import pandas as pd
import io
import pickle
import os
from dotenv import load_dotenv
import requests
import tempfile
import geopandas as gpd
import logging
import json
import torch 


class DropboxHelper:
    """
    Helper class for reading and writing files to Dropbox.

    Parameters
    ----------
    dbx_token : str
        The Dropbox API token.
    dbx_key : str
        The Dropbox app key.
    dbx_secret : str
        The Dropbox app secret.
    input_path : str
        The base path in Dropbox where the input data is stored.
    output_path : str
        The base path in Dropbox where the output data will be stored.
    custom_paths : bool, optional
        If True, the input_path and output_path will be used as provided.
        If False, additional paths (raw_input_path, clean_input_path, clean_merged_input_path)
        will be created based on the input_path. The default is False.

    Attributes
    ----------
    dbx : dropbox.Dropbox
        The Dropbox object used to interact with the Dropbox API.
    raw_input_path : str
        The path of the Dropbox folder containing the raw data.
    clean_input_path : str
        The path of the Dropbox folder containing the clean data.
    output_path : str
        The path of the Dropbox folder where the output data will be saved.
    """

    def __init__(self, dbx_token: str, dbx_key: str, dbx_secret: str,
                 input_path: str, output_path: str, custom_paths: bool = False):
        self.dbx = dropbox.Dropbox(oauth2_refresh_token=dbx_token,
                                   app_key=dbx_key,
                                   app_secret=dbx_secret)

        if custom_paths:
            self.input_path = input_path
            self.output_path = output_path
        else:
            (self.raw_input_path,
             self.clean_input_path,
             self.clean_merged_input_path,
             self.output_path) = self._initialize_paths(input_path, output_path)

    def _initialize_paths(self, input_path: str, output_path: str):
        """
        Initialize Dropbox folder paths and create necessary folders if they do not exist.

        Parameters
        ----------
        input_path : str
            The base path in Dropbox where the input data is stored.
        output_path : str
            The base path in Dropbox where the output data will be stored.

        Returns
        -------
        tuple
            A tuple containing:
            - raw_input_path (str): Path for raw data.
            - clean_input_path (str): Path for clean data.
            - clean_merged_input_path (str): Path for merged clean data.
            - output_path (str): Path for output data.
        """
        raw_input_path = f"{input_path}/raw"
        clean_input_path = f"{input_path}/clean"
        clean_merged_input_path = f"{input_path}/clean_merged"
        output_path = output_path

        # Create folders if they do not exist.
        for path in [raw_input_path, clean_input_path, clean_merged_input_path, output_path]:
            if not self.folder_exists(path):
                self.create_folder(path)
            else:
                pass

        return raw_input_path, clean_input_path, clean_merged_input_path, output_path

    def folder_exists(self, folder_path: str):
        """
        Check if a folder exists in Dropbox.

        Parameters
        ----------
        folder_path : str
            The Dropbox folder path to check.

        Returns
        -------
        bool
            True if the folder exists, False otherwise.
        """
        try:
            self.dbx.files_get_metadata(folder_path)
            return True
        except dropbox.exceptions.ApiError as err:
            if (isinstance(err.error, dropbox.files.GetMetadataError) and
                err.error.is_path() and
                    err.error.get_path().is_not_found()):
                return False
            else:
                print(f"Failed to check if folder '{folder_path}' exists:", err)

    def create_folder(self, folder_path: str, return_path: bool = False):
        """
        Create a folder in Dropbox.

        Parameters
        ----------
        folder_path : str
            The Dropbox folder path to create.
        return_path : bool, optional
            If True, returns the folder path after creation. The default is False.

        Returns
        -------
        str or None
            The folder path if return_path is True; otherwise, None.
        """
        try:
            self.dbx.files_create_folder_v2(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        except dropbox.exceptions.ApiError as err:
            if (isinstance(err.error, dropbox.files.CreateFolderError) and
                err.error.is_path() and
                    err.error.get_path().is_conflict()):
                print(f"Folder '{folder_path}' already exists.")
            else:
                print(f"Failed to create folder '{folder_path}':", err)

        if return_path:
            return folder_path

    def list_files_in_folder(self, folder_path: str, recursive: bool = False):
        """
        List all files in a specified Dropbox folder.

        Parameters
        ----------
        folder_path : str
            The Dropbox folder path to list files from.
        recursive : bool, optional
            If True, list files recursively in subfolders. The default is False.

        Returns
        -------
        list of str
            A list containing the names of the files in the folder.
        """
        try:
            files = self.dbx.files_list_folder(folder_path, recursive=recursive, limit=2000)
            print(f"Files in folder '{folder_path}':")
            return [f.name for f in files.entries]
        except dropbox.exceptions.ApiError as err:
            print(f"Failed to list files in folder '{folder_path}':", err)

    def read_csv(self, dbx_path: str, directory: str, filename: str, skiprows=None,
                 header='infer', usecols=None, sep=',', index_col=None, parse_dates=None,
                 dtype=None, mb_to_load=None):
        """
        Read a CSV file from Dropbox into a pandas DataFrame.

        Parameters
        ----------
        dbx_path : str
            The base Dropbox path where the file is stored.
        directory : str
            The directory within the base path. If None, the file is assumed to be directly under dbx_path.
        filename : str
            The name of the CSV file.
        skiprows : int or list-like, optional
            Number of lines to skip at the start of the file.
        header : int, list of int, or 'infer', optional
            Row number(s) to use as the column names.
        usecols : list-like, optional
            Return a subset of the columns.
        sep : str, optional
            Delimiter to use. Default is ','.
        index_col : int, str, sequence or False, optional
            Column to use as the row labels.
        parse_dates : list or dict, optional
            List of columns to parse as dates.
        dtype : data type, optional
            Data type for data or columns.
        mb_to_load : int, optional
            Number of megabytes to load (partial download). If None, the entire file is downloaded.

        Returns
        -------
        pandas.DataFrame or None
            The DataFrame containing the CSV data, or None if an error occurs.
        """
        if directory is None:
            file_path = f'{dbx_path}/{filename}'
        else:
            file_path = f'{dbx_path}/{directory}/{filename}'

        try:
            if mb_to_load is None:
                _, res = self.dbx.files_download(file_path)
                df = pd.read_csv(io.BytesIO(res.content), encoding='utf-8',
                                 skiprows=skiprows, header=header, usecols=usecols,
                                 sep=sep, index_col=index_col, parse_dates=parse_dates, dtype=dtype)
            else:
                metadata = self.dbx.files_get_temporary_link(file_path)
                temp_link = metadata.link
                headers = {"Range": f"bytes=0-{mb_to_load * 1_048_576}"}
                res = requests.get(temp_link, headers=headers)

                if res.status_code != 206:
                    print(f"Error: Received status code {res.status_code}")
                    return None

                content_str = res.content.decode('utf-8')
                lines = content_str.splitlines()
                lines = lines[:-1]  # Exclude the potentially incomplete last line
                content_str = "\n".join(lines)
                df = pd.read_csv(io.StringIO(content_str), skiprows=skiprows,
                                 header=header, usecols=usecols, sep=sep,
                                 index_col=index_col, parse_dates=parse_dates, dtype=dtype)
            return df
        except Exception as e:
            print(f"Error: {e}")
            return None

    def read_pickle(self, dbx_path: str, directory: str, filename: str):
        """
        Download a pickle file from Dropbox and deserialize it.

        Parameters
        ----------
        dbx_path : str
            The base Dropbox path where the file is stored.
        directory : str
            The directory within the base path where the file is stored.
        filename : str
            The name of the pickle file (e.g., 'my_object.pkl').

        Returns
        -------
        object or None
            The deserialized Python object, or None if an error occurs.
        """
        file_path = f'{dbx_path}/{directory}/{filename}'
        try:
            _, res = self.dbx.files_download(file_path)
            obj = pickle.loads(res.content)
            return obj
        except Exception as e:
            print(f"Failed to load '{filename}' from Dropbox. Error: {e}")
            return None

    def read_shp(self, dbx_path: str, directory: str):
        """
        Download shapefile components from Dropbox and load them as a GeoDataFrame.

        Parameters
        ----------
        dbx_path : str
            The base Dropbox path where the files are stored.
        directory : str
            The directory within the base path where the shapefile components are stored.

        Returns
        -------
        geopandas.GeoDataFrame or None
            The GeoDataFrame constructed from the shapefile, or None if an error occurs.
        """
        dir_path = f'{dbx_path}/{directory}'
        try:
            files = self.dbx.files_list_folder(dir_path).entries
            shapefile_extensions = {'.shp', '.shx', '.dbf', '.prj', '.cpg'}
            shapefile_files = [f for f in files if any(f.name.endswith(ext) for ext in shapefile_extensions)]

            if not shapefile_files:
                print(f"No shapefile components found in directory: {dir_path}")
                return None

            with tempfile.TemporaryDirectory() as tmpdir:
                for file in shapefile_files:
                    file_path = os.path.join(dir_path, file.name)
                    _, res = self.dbx.files_download(file_path)
                    temp_file_path = os.path.join(tmpdir, file.name)
                    with open(temp_file_path, 'wb') as temp_file:
                        temp_file.write(res.content)

                shp_file = [f for f in shapefile_files if f.name.endswith('.shp')][0]
                shp_file_path = os.path.join(tmpdir, shp_file.name)
                data = gpd.read_file(shp_file_path)
                return data
        except Exception as e:
            print(f"Failed to load shapefile from Dropbox. Error: {e}")
            return None

    def read_tif(self, dbx_path: str, directory: str, filename: str):
        """
        Download a .tif file from Dropbox and return it as an in-memory BytesIO object.

        Parameters
        ----------
        dbx_path : str
            The base Dropbox path where the file is stored.
        directory : str
            The directory within the base path where the file is stored.
        filename : str
            The name of the .tif file.

        Returns
        -------
        io.BytesIO or None
            The BytesIO object containing the .tif file data, or None if an error occurs.
        """
        file_path = f'{dbx_path}/{directory}/{filename}'
        try:
            _, res = self.dbx.files_download(file_path)
            raster_bytes = io.BytesIO(res.content)
            return raster_bytes
        except Exception as e:
            print(f"Failed to load '{filename}' from Dropbox. Error: {e}")
            return None

    def read_json(self, dbx_path: str, directory: str, filename: str):
        """
        Download a JSON file from Dropbox and deserialize it.

        Parameters
        ----------
        dbx_path : str
            The base Dropbox path where the file is stored.
        directory : str
            The directory within the base path where the file is stored.
        filename : str
            The name of the JSON file (e.g., 'data.json').

        Returns
        -------
        dict or list or None
            The deserialized JSON object, or None if an error occurs.
        """
        if directory is None:
            file_path = f'{dbx_path}/{filename}'
        else:
            file_path = f'{dbx_path}/{directory}/{filename}'

        try:
            _, res = self.dbx.files_download(file_path)
            json_obj = json.loads(res.content.decode('utf-8'))
            return json_obj
        except Exception as e:
            print(f"Failed to load '{filename}' from Dropbox. Error: {e}")
            return None

    def read_parquet(self, dbx_path: str, directory: str, filename: str):
        """
        Download a Parquet file from Dropbox and load it into a pandas DataFrame.

        Parameters
        ----------
        dbx_path : str
            The base Dropbox path where the file is stored.
        directory : str
            The directory within the base path where the file is stored.
        filename : str
            The name of the Parquet file (e.g., 'my_dataframe.parquet').

        Returns
        -------
        pandas.DataFrame or None
            The DataFrame loaded from the Parquet file, or None if an error occurs.
        """
        full_dropbox_path = os.path.join(dbx_path, directory, filename)
        try:
            _, res = self.dbx.files_download(full_dropbox_path)
            buffer = io.BytesIO(res.content)
            df = pd.read_parquet(buffer, engine='pyarrow')
            print(f"File '{filename}' successfully downloaded and loaded into a DataFrame.")
            return df
        except Exception as e:
            print(f"Failed to load Parquet file '{filename}' from Dropbox. Error: {e}")
            return None

    def write_csv(self, df: pd.DataFrame, dbx_path: str, directory: str, filename: str,
                  print_success: bool = True, print_size: bool = True):
        """
        Save a DataFrame as a CSV file and upload it to Dropbox.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to save.
        dbx_path : str
            The base Dropbox path where the file will be saved.
        directory : str
            The directory within the base path where the file will be saved.
        filename : str
            The name of the CSV file (e.g., 'my_dataframe.csv').
        print_success : bool, optional
            Whether to print a success message upon upload. Default is True.
        print_size : bool, optional
            Whether to print the file size before upload. Default is True.
        """
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue().encode('utf-8')
        size_in_bytes = len(csv_content)
        size_in_mb = size_in_bytes / 1024**2

        if print_size:
            print(f"Size of the CSV file: {size_in_mb:.2f} MB")

        full_dropbox_path = f'{dbx_path}/{directory}/{filename}'
        try:
            if size_in_mb < 150:
                self.dbx.files_upload(csv_content, full_dropbox_path,
                                      mode=dropbox.files.WriteMode.overwrite)
                if print_success:
                    print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")
            else:
                CHUNK_SIZE = 25 * 1024 * 1024  # 25MB
                upload_session_start_result = self.dbx.files_upload_session_start(csv_content[:CHUNK_SIZE])
                cursor = dropbox.files.UploadSessionCursor(
                    session_id=upload_session_start_result.session_id,
                    offset=len(csv_content[:CHUNK_SIZE])
                )
                remaining_content = csv_content[CHUNK_SIZE:]
                while len(remaining_content) > 0:
                    if len(remaining_content) > CHUNK_SIZE:
                        self.dbx.files_upload_session_append_v2(remaining_content[:CHUNK_SIZE], cursor)
                        cursor.offset += CHUNK_SIZE
                        remaining_content = remaining_content[CHUNK_SIZE:]
                    else:
                        self.dbx.files_upload_session_finish(
                            remaining_content, cursor,
                            dropbox.files.CommitInfo(path=full_dropbox_path,
                                                     mode=dropbox.files.WriteMode.overwrite)
                        )
                        remaining_content = []
                if print_success:
                    print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")
        except Exception as e:
            print(f"Failed to upload '{filename}' to Dropbox. Error: {e}")

    def write_pickle(self, obj, dbx_path: str, directory: str, filename: str):
        """
        Serialize a Python object with pickle and upload it to Dropbox.

        Parameters
        ----------
        obj : any
            The Python object to serialize.
        dbx_path : str
            The base Dropbox path where the file will be saved.
        directory : str
            The directory within the base path.
        filename : str
            The name of the pickle file (e.g., 'my_object.pkl').
        """
        pickle_buffer = io.BytesIO()
        pickle.dump(obj, pickle_buffer)
        pickle_buffer.seek(0)
        full_dropbox_path = f'{dbx_path}/{directory}/{filename}'
        try:
            self.dbx.files_upload(pickle_buffer.getvalue(), full_dropbox_path,
                                  mode=dropbox.files.WriteMode.overwrite)
            print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")
        except Exception as e:
            print(f"Failed to upload '{filename}' to Dropbox. Error: {e}")

    def write_fig(self, buffer, dbx_path: str, directory: str, filename: str):
        """
        Upload a PNG image from an in-memory buffer to Dropbox.

        Parameters
        ----------
        buffer : io.BytesIO
            The buffer containing the PNG image data.
        dbx_path : str
            The base Dropbox path where the file will be saved.
        directory : str
            The directory within the base path.
        filename : str
            The name of the file (e.g., 'my_plot.png').
        """
        full_dropbox_path = f'{dbx_path}/{directory}/{filename}'
        try:
            self.dbx.files_upload(buffer.getvalue(), full_dropbox_path,
                                  mode=dropbox.files.WriteMode.overwrite)
            print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")
        except Exception as e:
            print(f"Failed to upload '{filename}' to Dropbox. Error: {e}")

    def write_shp(self, gdf: gpd.GeoDataFrame, dbx_path: str, directory: str, filename: str):
        """
        Save a GeoDataFrame as a shapefile and upload its components to Dropbox.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            The GeoDataFrame to save.
        dbx_path : str
            The base Dropbox path where the files will be saved.
        directory : str
            The directory within the base path.
        filename : str
            The base name of the shapefile (without extension).
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                shapefile_path = os.path.join(tmpdir, filename + ".shp")
                gdf.to_file(shapefile_path, driver='ESRI Shapefile')
                print(f"Shapefile written to temporary directory: {tmpdir}")
                shapefile_extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
                for ext in shapefile_extensions:
                    local_file_path = os.path.join(tmpdir, filename + ext)
                    full_dropbox_path = f"{dbx_path}/{directory}/{filename}{ext}"
                    if os.path.exists(local_file_path):
                        try:
                            with open(local_file_path, 'rb') as f:
                                self.dbx.files_upload(f.read(), full_dropbox_path,
                                                      mode=dropbox.files.WriteMode.overwrite)
                            print(f"File '{filename}{ext}' successfully uploaded to Dropbox at '{full_dropbox_path}'")
                        except Exception as e:
                            print(f"Failed to upload '{filename}{ext}' to Dropbox. Error: {e}")
                    else:
                        print(f"File '{filename}{ext}' not found in temporary directory, skipping upload.")
        except Exception as e:
            print(f"Error during shapefile writing or uploading: {e}")

    def write_tex(self, latex_content: str, dbx_path: str, directory: str, filename: str):
        """
        Upload LaTeX content as a .tex file to Dropbox.

        Parameters
        ----------
        latex_content : str
            The LaTeX content to save.
        dbx_path : str
            The base Dropbox path where the file will be saved.
        directory : str
            The directory within the base path.
        filename : str
            The name of the .tex file.
        """
        full_dropbox_path = f"{dbx_path}/{directory}/{filename}"
        try:
            self.dbx.files_upload(latex_content.encode('utf-8'), full_dropbox_path,
                                  mode=dropbox.files.WriteMode.overwrite)
            print(f"Successfully uploaded to {full_dropbox_path}")
        except dropbox.exceptions.ApiError as err:
            print(f"Failed to upload to Dropbox: {err}")

    def write_bytes(self, file_bytes, dbx_path: str, directory: str, filename: str, print_success: bool = True):
        """
        Upload raw file bytes to Dropbox using chunked upload for large files.

        Parameters
        ----------
        file_bytes : bytes
            The content of the file to upload.
        dbx_path : str
            The base Dropbox path where the file will be saved.
        directory : str
            The directory within the base path.
        filename : str
            The name of the file (e.g., 'my_file.tif').
        print_success : bool, optional
            Whether to print a success message upon upload. Default is True.
        """
        full_dropbox_path = f"{dbx_path}/{directory}/{filename}"
        CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB
        try:
            file_size = len(file_bytes)
            if file_size <= CHUNK_SIZE:
                self.dbx.files_upload(file_bytes, full_dropbox_path,
                                      mode=dropbox.files.WriteMode.overwrite)
            else:
                f = io.BytesIO(file_bytes)
                session_start_result = self.dbx.files_upload_session_start(f.read(CHUNK_SIZE))
                session_id = session_start_result.session_id
                cursor = dropbox.files.UploadSessionCursor(session_id=session_id, offset=f.tell())
                commit = dropbox.files.CommitInfo(path=full_dropbox_path, mode=dropbox.files.WriteMode.overwrite)
                while f.tell() < file_size:
                    if (file_size - f.tell()) <= CHUNK_SIZE:
                        self.dbx.files_upload_session_finish(f.read(CHUNK_SIZE), cursor, commit)
                    else:
                        self.dbx.files_upload_session_append_v2(f.read(CHUNK_SIZE), cursor)
                        cursor.offset = f.tell()
            if print_success:
                print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")
        except Exception as e:
            print(f"Failed to upload '{filename}' to Dropbox. Error: {e}")

    def write_parquet(self, df: pd.DataFrame, dbx_path: str, directory: str, filename: str,
                      print_success: bool = True, print_size: bool = True, index: bool = True):
        """
        Save a DataFrame as a Parquet file and upload it to Dropbox.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to save.
        dbx_path : str
            The base Dropbox path where the file will be saved.
        directory : str
            The directory within the base path.
        filename : str
            The name of the Parquet file (e.g., 'my_dataframe.parquet').
        print_success : bool, optional
            Whether to print a success message upon upload. Default is True.
        print_size : bool, optional
            Whether to print the file size before upload. Default is True.
        index : bool, optional
            Whether to include the DataFrame index in the file. Default is True.
        """
        full_dropbox_path = f"{dbx_path}/{directory}/{filename}"
        try:
            buffer = io.BytesIO()
            df.to_parquet(buffer, engine='pyarrow', index=index)
            buffer.seek(0)
            parquet_content = buffer.getvalue()
            size_in_bytes = len(parquet_content)
            size_in_mb = size_in_bytes / (1024 ** 2)
            if print_size:
                print(f"Size of the Parquet file: {size_in_mb:.2f} MB")
            if size_in_mb < 150:
                self.dbx.files_upload(parquet_content, full_dropbox_path,
                                      mode=dropbox.files.WriteMode.overwrite)
            else:
                self._chunked_upload_to_dropbox(parquet_content, full_dropbox_path)
            if print_success:
                print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")
        except Exception as e:
            print(f"Failed to upload Parquet file '{filename}' to Dropbox. Error: {e}")

    def write_log(self, log_content: str, dbx_path: str, filename: str):
        """
        Write log content to a file and upload it to Dropbox.

        Parameters
        ----------
        log_content : str
            The log content to upload.
        dbx_path : str
            The Dropbox path where the log will be saved.
        filename : str
            The name of the log file (e.g., 'logs.txt').
        """
        full_dropbox_path = f"{dbx_path}/{filename}"
        try:
            log_bytes = log_content.encode("utf-8")
            self.dbx.files_upload(log_bytes, full_dropbox_path,
                                  mode=dropbox.files.WriteMode.overwrite)
            print(f"Log file '{filename}' successfully uploaded to Dropbox at '{full_dropbox_path}'")
        except Exception as e:
            print(f"Failed to upload log file '{filename}' to Dropbox. Error: {e}")

    def save_torch(self, obj, dbx_path: str, directory: str, filename: str):
        """
        Serialize a PyTorch object using torch.save and upload it to Dropbox.

        Parameters
        ----------
        obj : any
            The PyTorch object to serialize.
        dbx_path : str
            The base Dropbox path.
        directory : str
            The directory within the base path.
        filename : str
            The name of the file (e.g., 'model.pt').
        """
        buffer = io.BytesIO()
        try:
            torch.save(obj, buffer)
            buffer.seek(0)
            full_dropbox_path = f"{dbx_path}/{directory}/{filename}"
            self.dbx.files_upload(buffer.getvalue(), full_dropbox_path,
                                  mode=dropbox.files.WriteMode.overwrite)
            print(f"Torch object successfully uploaded to Dropbox at '{full_dropbox_path}'")
        except Exception as e:
            print(f"Failed to upload torch object to Dropbox. Error: {e}")

    def load_torch(self, dbx_path: str, directory: str, filename: str):
        """
        Download a serialized PyTorch object from Dropbox and deserialize it using torch.load.

        Parameters
        ----------
        dbx_path : str
            The base Dropbox path.
        directory : str
            The directory within the base path.
        filename : str
            The name of the file (e.g., 'model.pt').

        Returns
        -------
        object or None
            The deserialized PyTorch object, or None if an error occurs.
        """
        file_path = f"{dbx_path}/{directory}/{filename}"
        try:
            _, res = self.dbx.files_download(file_path)
            buffer = io.BytesIO(res.content)
            obj = torch.load(buffer)
            return obj
        except Exception as e:
            print(f"Failed to load torch object from Dropbox. Error: {e}")
            return None



load_dotenv(override = True)
dbx_helper = DropboxHelper(
    dbx_token = os.getenv('DROPBOX_TOKEN'),
    dbx_key = os.getenv('DROPBOX_KEY'),
    dbx_secret= os.getenv('DROPBOX_SECRET'),
    input_path = f"{os.getenv('INPUT_PATH')}",
    output_path = os.getenv('OUTPUT_PATH'),
    custom_paths = False
    )