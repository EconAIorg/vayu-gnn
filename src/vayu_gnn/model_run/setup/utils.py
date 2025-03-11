import time
import warnings
import pandas as pd
from typing import List
import re

class StepTimer:
    def __init__(self, start_timer=True, step_name = None):
        self.start_time = None
        if start_timer:
            self.start(step_name=step_name)

    def start(self, step_name = None):
        """Starts or restarts the timer."""
        self.start_time = time.time()
        if step_name is not None:
            print(f'Starting timer for {step_name}')

    def stop(self, step_name="Process"):
        """Stops the timer and prints the elapsed time."""
        if self.start_time is None:
            print("Timer has not been started.")
            return
        elapsed_time = (time.time() - self.start_time) / 60
        print(f"Time taken for {step_name}: {elapsed_time:.2f} minutes")
        self.start_time = None  # Reset the timer after stopping

def check_if_missing_values(df, error_threshold=1.0, warning_threshold=0.0):
    # Check for missing values.
    na_mean = df.isna().mean().sort_values(ascending=False).round(3)
    if na_mean.any() > warning_threshold:
        warnings.warn(f"There are missing values in the DataFrame. \n Proportion of missing values per column:")
        print(na_mean.loc[na_mean > warning_threshold])

    assert (na_mean <= error_threshold).all(), f"Column(s) with proportion of missing greater than error threshold: {na_mean.loc[na_mean > error_threshold]}"

def extract_columns_from_query(query: str) -> List[str]:
    """
    Extracts column names from a query string.

    Parameters:
        query (str): The query string (e.g., used in pandas `.query()`).

    Returns:
        List[str]: A list of column names used in the query. Returns an empty list if the query is "index==index" or equivalent.
    """
    # Handle the special case where the query is "index == index"
    normalized_query = query.replace(" ", "")  # Remove spaces for comparison
    if normalized_query == "index==index":
        return []

    # Regular expression to capture potential column names
    column_pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b'
    
    # Exclude Python keywords, logical operators, and numeric values
    excluded_keywords = {'and', 'or', 'not', 'in', 'if', 'else', 'True', 'False', 'None'}
    logical_operators = {'==', '!=', '>', '<', '>=', '<=', '&', '|', '~', '(', ')', '[', ']'}
    
    # Find all matches for valid identifiers
    potential_columns = set(re.findall(column_pattern, query))
    
    # Filter out excluded keywords and logical operators
    columns = [
        col for col in potential_columns
        if col not in excluded_keywords and col not in logical_operators
    ]
    
    return columns


# This is a lite source names back to original names.
# This is the easiest way to flexibly switch between lite and not lite.
def rewrite_lite_names(data_loader):
    # remove lite from data_loader
    for key, val in data_loader.dfs.copy().items():
        if 'lite' in key:
            removed = key.removesuffix('_lite')
            data_loader.dfs[removed] = data_loader.dfs.pop(key)
    return data_loader

import sys
from io import StringIO
import traceback
from datetime import datetime
import os
class CaptureLogs:

    def __init__(self, dbx_helper=None, dbx_path="logs", filename=None, capture_on=True):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._captured_output = StringIO()
        self.dbx_helper = dbx_helper
        self.dbx_path = dbx_path
        # Generate default filename if not provided
        if filename is None:
            self.filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
        else:
            self.filename = filename
        if capture_on:
            self.capture(on=True)

    class Tee:
        """
        Helper class to send output to both the original stream and the StringIO buffer.
        """
        def __init__(self, original_stream, captured_stream):
            self.original_stream = original_stream
            self.captured_stream = captured_stream

        def write(self, data):
            self.original_stream.write(data)  # Write to the original stream (console)
            self.captured_stream.write(data)  # Write to the captured stream (log)

        def flush(self):
            self.original_stream.flush()
            self.captured_stream.flush()

    def capture(self, on=True):
        if on:
            sys.stdout = self.Tee(self._original_stdout, self._captured_output)
            sys.stderr = self.Tee(self._original_stderr, self._captured_output)
        else:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

    def save_logs_to_local(self, folder="logs"):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, self.filename)
        with open(filepath, 'w') as log_file:
            log_file.write(self._captured_output.getvalue())
        print(f"Logs saved locally to: {filepath}")
        return filepath

    def save_logs_to_dropbox(self):
        if self.dbx_helper is None:
            print("DropboxHelper instance not provided. Logs not saved to Dropbox.")
            return
        try:
            log_content = self._captured_output.getvalue()
            self.dbx_helper.write_log(log_content, self.dbx_path, self.filename)
        except Exception as e:
            print(f"Failed to save logs to Dropbox. Error: {e}")

    def get_logs(self):
        return self._captured_output.getvalue()

    def clear_logs(self):
        self._captured_output = StringIO()
