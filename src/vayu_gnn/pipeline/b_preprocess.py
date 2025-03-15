from vayu_gnn.dbx.dbx_config import DropboxHelper
from vayu_gnn.data.preprocessor import Preprocessor

def execute(dbx_helper:DropboxHelper, preprocess_data:list, preprocessor_params:dict, nodes:dict, cities:list) :
    
        # Initialize the Preprocessor class

        pp = Preprocessor(dbx_helper, nodes, cities)

        # Loop through the data types to be preprocessed
        for data in preprocess_data:
    
            params = preprocessor_params[data]
    
            if params is None:
                # Execute the preprocessing method
                getattr(pp, data)()
            else:
                # Execute the preprocessing method with parameters
                getattr(pp, data)(**params)
    
        return None