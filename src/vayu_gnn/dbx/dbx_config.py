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

class DropboxHelper:
    """
    This class contains helper functions for reading and writing files to Dropbox.

    Args:
    - dbx_token (str): The Dropbox API token.
    - dbx_key (str): The Dropbox app key.
    - dbx_secret (str): The Dropbox app secret.
    - input_path (str): The base path in Dropbox where the input data is stored.
    - output_path (str): The base path in Dropbox where the output data will be stored.
    - custom_paths (bool): If True, the input_path and output_path will be used as is. If False, the input_path and output_path will be used to create the raw_input_path, clean_input_path and output_path.

    Attributes:
    - dbx (dropbox.Dropbox): The Dropbox object used to interact with the Dropbox API.
    - raw_input_path (str): The path of the Dropbox folder containing the raw data.
    - clean_input_path (str): The path of the Dropbox folder containing the clean data.
    - output_path (str): The path of the Dropbox folder where the output data will be saved.
    
    """
    
    def __init__(self, dbx_token:str, dbx_key:str, dbx_secret:str, input_path:str, output_path:str, custom_paths:bool = False):

        self.dbx = dropbox.Dropbox(oauth2_refresh_token = dbx_token, app_key = dbx_key, app_secret = dbx_secret)

        if custom_paths:
            self.input_path = input_path
            self.output_path = output_path
        
        else:
            self.raw_input_path, self.clean_input_path, self.clean_merged_input_path, self.output_path = self._initialize_paths(input_path, output_path)

    def _initialize_paths(self, input_path:str, output_path:str):

        """
        Function to initialize your Dropbox App and create the necessary folders.

        Args
        ----

        input_path (str): The base path in Dropbox where the input data is stored.
        output_path (str): The base path in Dropbox where the output data will be stored.

        Returns
        -------
        raw_input_path (str): The path of the Dropbox folder containing the raw data.
        clean_input_path (str): The path of the Dropbox folder containing the clean data.
        output_path (str): The path of the Dropbox folder where outputs will be saved.
        """

        raw_input_path = f"{input_path}/raw"
        clean_input_path = f"{input_path}/clean"
        clean_merged_input_path = f"{input_path}/clean_merged"
        output_path = output_path

        #If the folder doesn't exist (i.e. you didn't manually create your input/raw, input/clean and output folders in Dropbox), then make them
        for path in [raw_input_path, clean_input_path, clean_merged_input_path, output_path]:
            if not self.folder_exists(path):
                self.create_folder(path)
            else:
                pass

        return raw_input_path, clean_input_path, clean_merged_input_path, output_path

    """
    Directory and file management functions
    """

    def folder_exists(self, folder_path):

        # Check if the folder exists
        try:
            self.dbx.files_get_metadata(folder_path)
            return True
        except dropbox.exceptions.ApiError as err:
            if isinstance(err.error, dropbox.files.GetMetadataError) and err.error.is_path() and \
                    err.error.get_path().is_not_found():
                return False
            else:
                print(f"Failed to check if folder '{folder_path}' exists:", err)

    """
    Directory and file management functions
    """

    def create_folder(self, folder_path, return_path = False):
        # Try to create the new folder
        try:
            self.dbx.files_create_folder_v2(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        except dropbox.exceptions.ApiError as err:
            # Check if the error is because the folder already exists
            if isinstance(err.error, dropbox.files.CreateFolderError) and err.error.is_path() and \
                    err.error.get_path().is_conflict():
                print(f"Folder '{folder_path}' already exists.")
            else:
                print(f"Failed to create folder '{folder_path}':", err)

        if return_path:
            return folder_path
        
    def list_files_in_folder(self, folder_path, recursive=False):
        # List all files in the folder
        try:
            files = self.dbx.files_list_folder(folder_path, recursive=recursive, limit = 2000)
            print(f"Files in folder '{folder_path}':")
            return [f.name for f in files.entries]
        except dropbox.exceptions.ApiError as err:
            print(f"Failed to list files in folder '{folder_path}':", err)
    
    """
    Reading functions
    """
    def read_csv(self, dbx_path:str, directory:str, filename:str, skiprows=None, header='infer', usecols = None, sep = ',', index_col = None, parse_dates = None, dtype = None, mb_to_load = None):
        """
        This function reads a CSV file from Dropbox into a pandas DataFrame.

        Args:
        - file_path (str): The path of the CSV file in Dropbox.
        - skiprows (int): The number of rows to skip at the beginning of the file.
        - usecols (list): The columns to read from the CSV file.
        - sep (str): The separator used in the CSV file. ',' is the pandas default.

        Returns:
        - df (pandas.DataFrame): The DataFrame containing the data from the CSV file.
        """
        if directory == None:
            file_path = f'{dbx_path}/{filename}'
        else:
            file_path = f'{dbx_path}/{directory}/{filename}'

        try:
            # Use Dropbox's files_download to get the file

            # If mb_to_load is not specified, download the entire file
            if mb_to_load is None:
                _, res = self.dbx.files_download(file_path)
                # Read the content into a pandas DataFrame, starting from the specified row and columns
                df = pd.read_csv(io.BytesIO(res.content), encoding='utf-8', skiprows=skiprows, header=header, usecols = usecols, sep = sep, index_col = index_col, parse_dates = parse_dates, dtype = dtype)
            
            # If mb_to_load is specified, download only the specified number of MB
            else:
                # Get the temporary link to the file
                metadata = self.dbx.files_get_temporary_link(file_path)
                temp_link = metadata.link

                # Use requests to download the specified byte range
                headers = {"Range": f"bytes=0-{mb_to_load*1_048_576}"} # 1MB = 1_048_576 bytes
                res = requests.get(temp_link, headers=headers)

                if res.status_code != 206:  # Expecting a partial content response
                    print(f"Error: Received status code {res.status_code}")
                    return None

                content_str = res.content.decode('utf-8')
                lines = content_str.splitlines()
                # Exclude the last line (assuming it might be incomplete)
                lines = lines[:-1]

                # Rejoin the lines and pass them to pandas
                content_str = "\n".join(lines)
                
                # Read the content into a pandas DataFrame, starting from the specified row and columns
                df = pd.read_csv(io.StringIO(content_str), skiprows=skiprows, header=header, usecols = usecols, sep = sep, index_col = index_col, parse_dates = parse_dates, dtype = dtype)
            
            return df
        
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def read_pickle(self, dbx_path:str, directory:str, filename:str):
        """
        Downloads a pickle file from Dropbox, deserializes it, and returns the Python object.

        Args:
        - dbx_path (str): The base Dropbox path where the file is stored.
        - directory (str): The directory within the base path where the file is stored.
        - filename (str): The name of the pickle file (e.g., 'my_object.pkl').

        Returns:
        The deserialized Python object from the pickle file.
        """
        # Full path where the file is stored in Dropbox
        file_path = f'{dbx_path}/{directory}/{filename}'

        try:
            # Use Dropbox's files_download to get the file
            _, res = self.dbx.files_download(file_path)
            
            # Deserialize the pickle data from the response content
            obj = pickle.loads(res.content)
            
            return obj
        except Exception as e:
            print(f"Failed to load '{filename}' from Dropbox. Error: {e}")
            return None

    def read_shp(self, dbx_path: str, directory: str):
        """
        Downloads all files related to a shapefile from Dropbox, deserializes them, and returns the GeoDataFrame.

        Args:
        - dbx_path (str): The base Dropbox path where the files are stored.
        - directory (str): The directory within the base path where the files are stored.

        Returns:
        The deserialized GeoDataFrame from the shapefile.
        """
        # Full path where the directory is stored in Dropbox
        dir_path = f'{dbx_path}/{directory}'

        try:
            # List files in the directory
            files = self.dbx.files_list_folder(dir_path).entries

            # Filter to include only shapefile-related extensions
            shapefile_extensions = {'.shp', '.shx', '.dbf', '.prj', '.cpg'}
            shapefile_files = [f for f in files if any(f.name.endswith(ext) for ext in shapefile_extensions)]

            if not shapefile_files:
                print(f"No shapefile components found in directory: {dir_path}")
                return None

            # Create a temporary directory to download the files
            with tempfile.TemporaryDirectory() as tmpdir:
                for file in shapefile_files:
                    # Download each file
                    file_path = os.path.join(dir_path, file.name)
                    _, res = self.dbx.files_download(file_path)
                    
                    # Save the file to the temporary directory
                    temp_file_path = os.path.join(tmpdir, file.name)
                    with open(temp_file_path, 'wb') as temp_file:
                        temp_file.write(res.content)
                
                # Find the .shp file to read with GeoPandas
                shp_file = [f for f in shapefile_files if f.name.endswith('.shp')][0]
                shp_file_path = os.path.join(tmpdir, shp_file.name)
                
                # Load the shapefile from the temporary directory
                data = gpd.read_file(shp_file_path)
                
                return data
        except Exception as e:
            print(f"Failed to load shapefile from Dropbox. Error: {e}")
            return None
        
    def read_tif(self, dbx_path: str, directory: str, filename: str):
        """
        Downloads a .tif file from Dropbox and returns it as a BytesIO object.

        Args:
            dbx_path (str): The base Dropbox path where the file is stored.
            directory (str): The directory within the base path where the file is stored.
            filename (str): The name of the .tif file.

        Returns:
            io.BytesIO: The in-memory raster file.
        """
        file_path = f'{dbx_path}/{directory}/{filename}'

        try:
            # Download the file from Dropbox
            _, res = self.dbx.files_download(file_path)
            # Load the content into a BytesIO object
            raster_bytes = io.BytesIO(res.content)
            return raster_bytes
        except Exception as e:
            print(f"Failed to load '{filename}' from Dropbox. Error: {e}")
            return None
        
    def read_json(self, dbx_path: str, directory: str, filename: str):
        """
        Downloads a JSON file from Dropbox, deserializes it, and returns the Python object.

        Args:
            dbx_path (str): The base Dropbox path where the file is stored.
            directory (str): The directory within the base path where the file is stored.
            filename (str): The name of the JSON file (e.g., 'data.json').

        Returns:
            dict or list: The deserialized JSON object.
        """
        # Full path where the file is stored in Dropbox
        if directory == None:
            file_path = f'{dbx_path}/{filename}'
        else:
            file_path = f'{dbx_path}/{directory}/{filename}'

        try:
            # Use Dropbox's files_download to get the file
            _, res = self.dbx.files_download(file_path)

            # Deserialize the JSON data from the response content
            json_obj = json.loads(res.content.decode('utf-8'))

            return json_obj
        except Exception as e:
            print(f"Failed to load '{filename}' from Dropbox. Error: {e}")
            return None
        
    def read_parquet(self, dbx_path: str, directory: str, filename: str):
        """
        Downloads a Parquet file from Dropbox and loads it into a pandas DataFrame.

        Args:
        - dbx_path (str): The base Dropbox path where the file is stored.
        - directory (str): The directory within the base path where the file is stored.
        - filename (str): The name of the file (e.g., 'my_dataframe.parquet').

        Returns:
        - pandas.DataFrame: The DataFrame loaded from the Parquet file.
        """
        full_dropbox_path = os.path.join(dbx_path, directory, filename)
        try:
            # Download the Parquet file from Dropbox
            _, res = self.dbx.files_download(full_dropbox_path)

            # Load the content into a DataFrame
            buffer = io.BytesIO(res.content)
            df = pd.read_parquet(buffer, engine='pyarrow')

            print(f"File '{filename}' successfully downloaded and loaded into a DataFrame.")
            return df
        except Exception as e:
            print(f"Failed to load Parquet file '{filename}' from Dropbox. Error: {e}")
            return None

    """
    Writing functions
    """

    def write_csv(self, df: pd.DataFrame, dbx_path:str, directory:str, filename:str, print_success = True, print_size = True):
        """
        Saves a DataFrame to a CSV file and uploads it to Dropbox.

        Args:
        - df (pandas.DataFrame): The DataFrame to save.
        - write_path (str): The path in Dropbox where the file will be saved.
        - filename (str): The name of the file (e.g., 'my_dataframe.csv').
        """

        # Convert DataFrame to CSV format
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue().encode('utf-8')

        size_in_bytes = len(csv_content)
        size_in_mb = size_in_bytes / 1024**2   # Convert kilobytes to megabytes

        if print_size:
            print(f"Size of the CSV file: {size_in_mb:.2f} MB")

        
        # Full path where the file will be saved in Dropbox
        full_dropbox_path = f'{dbx_path}/{directory}/{filename}'

        # Upload the CSV to Dropbox
        try:
            if size_in_mb < 150: #dropbox API has a limit of 150MB for this method
                self.dbx.files_upload(csv_content, full_dropbox_path, mode=dropbox.files.WriteMode.overwrite)

                if print_success:
                    print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")

            else:
                # Use chunked upload for large files
                CHUNK_SIZE = 25 * 1024 * 1024  # 25MB chunk size
                upload_session_start_result = self.dbx.files_upload_session_start(csv_content[:CHUNK_SIZE])
                cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,
                                                        offset=csv_content[:CHUNK_SIZE].__len__())
                remaining_content = csv_content[CHUNK_SIZE:]
                
                while len(remaining_content) > 0:
                    if len(remaining_content) > CHUNK_SIZE:
                        self.dbx.files_upload_session_append_v2(remaining_content[:CHUNK_SIZE], cursor)
                        cursor.offset += CHUNK_SIZE
                        remaining_content = remaining_content[CHUNK_SIZE:]
                    else:
                        # Move the remaining data in the last chunk
                        self.dbx.files_upload_session_finish(remaining_content, cursor,
                                                            dropbox.files.CommitInfo(path=full_dropbox_path,
                                                                                    mode=dropbox.files.WriteMode.overwrite))
                        remaining_content = []
                if print_success:
                    print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")
            
        except Exception as e:
            print(f"Failed to upload '{filename}' to Dropbox. Error: {e}")

    def write_pickle(self, obj, dbx_path:str, directory:str, filename:str):
        """
        Serializes a Python object using pickle and uploads it to Dropbox.

        Args:
        - obj (Any): The Python object to serialize and save.
        - dbx_path (str): The base Dropbox path where the file will be saved.
        - directory (str): The directory within the base path where the file will be saved.
        - filename (str): The name of the file (e.g., 'my_object.pkl').
        """
        # Serialize the object to a bytes stream
        pickle_buffer = io.BytesIO()
        pickle.dump(obj, pickle_buffer)
        pickle_buffer.seek(0)  # Rewind the buffer to the beginning after writing

        # Full path where the file will be saved in Dropbox
        full_dropbox_path = f'{dbx_path}/{directory}/{filename}'

        # Upload the serialized object to Dropbox
        try:
            self.dbx.files_upload(pickle_buffer.getvalue(), full_dropbox_path, mode=dropbox.files.WriteMode.overwrite)
            print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")
        except Exception as e:
            print(f"Failed to upload '{filename}' to Dropbox. Error: {e}")

    def write_fig(self, buffer, dbx_path:str, directory:str, filename:str):
        """
        Uploads a PNG image from an in-memory buffer to Dropbox.

        Args:
        - buffer (io.BytesIO): The buffer containing the PNG image data.
        - directory (str): The name of the directory where the file will be saved.
        - filename (str): The name of the file (e.g., 'my_plot.png').
        - write_path (str): The path in Dropbox where the file will be saved.
        """
        
        # Full path where the file will be saved in Dropbox
        full_dropbox_path = f'{dbx_path}/{directory}/{filename}'

        # Upload the PNG to Dropbox
        try:
            self.dbx.files_upload(buffer.getvalue(), full_dropbox_path, mode=dropbox.files.WriteMode.overwrite)
            print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")
        except Exception as e:
            print(f"Failed to upload '{filename}' to Dropbox. Error: {e}")

    def write_shp(self, gdf: gpd.GeoDataFrame, dbx_path: str, directory: str, filename: str):
        """
        Saves a GeoDataFrame to a shapefile and uploads it to Dropbox.

        Args:
            gdf (geopandas.GeoDataFrame): The GeoDataFrame to save.
            dbx_path (str): The base Dropbox path where the file will be saved.
            directory (str): The directory within the base path where the file will be saved.
            filename (str): The base name of the shapefile (without extensions).
        """
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write the shapefile to the temporary directory
                shapefile_path = os.path.join(tmpdir, filename + ".shp")
                
                # Ensure the shapefile is written
                gdf.to_file(shapefile_path, driver='ESRI Shapefile')
                print(f"Shapefile written to temporary directory: {tmpdir}")
                
                # List of shapefile extensions to look for
                shapefile_extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']

                # Upload each component of the shapefile to Dropbox
                for ext in shapefile_extensions:
                    local_file_path = os.path.join(tmpdir, filename + ext)
                    full_dropbox_path = f"{dbx_path}/{directory}/{filename}{ext}"

                    # Check if the file exists before attempting to upload
                    if os.path.exists(local_file_path):
                        try:
                            with open(local_file_path, 'rb') as f:
                                self.dbx.files_upload(f.read(), full_dropbox_path, mode=dropbox.files.WriteMode.overwrite)
                            print(f"File '{filename}{ext}' successfully uploaded to Dropbox at '{full_dropbox_path}'")
                        except Exception as e:
                            print(f"Failed to upload '{filename}{ext}' to Dropbox. Error: {e}")
                    else:
                        print(f"File '{filename}{ext}' not found in temporary directory, skipping upload.")
        
        except Exception as e:
            print(f"Error during shapefile writing or uploading: {e}")

    def write_tex(self, latex_content: str, dbx_path: str, directory: str, filename: str):
        """
        Uploads LaTeX content as a .tex file to Dropbox.

        Args:
        - latex_content (str): The LaTeX content to write to the file.
        - dbx_path (str): The base Dropbox path where the file will be saved.
        - directory (str): The directory within the base path where the file will be saved.
        - filename (str): The name of the .tex file.
        """
        
        full_dropbox_path = f"{dbx_path}/{directory}/{filename}"
        try:
            self.dbx.files_upload(latex_content.encode('utf-8'), full_dropbox_path, mode=dropbox.files.WriteMode.overwrite)
            print(f"Successfully uploaded to {full_dropbox_path}")
        except dropbox.exceptions.ApiError as err:
            print(f"Failed to upload to Dropbox: {err}")

    def write_parquet(self, df: pd.DataFrame, dbx_path: str, directory: str, filename: str, print_success=True, print_size=True, index=True):
        """
        Saves a DataFrame to a Parquet file and uploads it to Dropbox.

        Args:
        - df (pandas.DataFrame): The DataFrame to save.
        - dbx_path (str): The base Dropbox path where the file will be saved.
        - directory (str): The directory within the base path where the file will be saved.
        - filename (str): The name of the file (e.g., 'my_dataframe.parquet').
        - print_success (bool): Whether to print a success message upon successful upload.
        - print_size (bool): Whether to print the file size before upload.
        """
        full_dropbox_path = f"{dbx_path}/{directory}/{filename}"
        try:
            # Save DataFrame to a Parquet file in memory
            buffer = io.BytesIO()
            df.to_parquet(buffer, engine='pyarrow', index=index)
            buffer.seek(0)
            parquet_content = buffer.getvalue()

            # Calculate file size
            size_in_bytes = len(parquet_content)
            size_in_mb = size_in_bytes / (1024 ** 2)  # Convert bytes to megabytes
            if print_size:
                print(f"Size of the Parquet file: {size_in_mb:.2f} MB")

            # Upload Parquet file to Dropbox
            if size_in_mb < 150:  # Dropbox API limit for direct upload
                self.dbx.files_upload(parquet_content, full_dropbox_path, mode=dropbox.files.WriteMode.overwrite)
            else:
                self._chunked_upload_to_dropbox(parquet_content, full_dropbox_path)

            if print_success:
                print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")
        except Exception as e:
            print(f"Failed to upload Parquet file '{filename}' to Dropbox. Error: {e}")



load_dotenv(override = True)
dbx_helper = DropboxHelper(
    dbx_token = os.getenv('DROPBOX_TOKEN'),
    dbx_key = os.getenv('DROPBOX_KEY'),
    dbx_secret= os.getenv('DROPBOX_SECRET'),
    input_path = f"{os.getenv('INPUT_PATH')}",
    output_path = os.getenv('OUTPUT_PATH'),
    custom_paths = False
    )