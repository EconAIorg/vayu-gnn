#%%
from vayu_gnn.model_run.setup.utils import StepTimer, CaptureLogs
from vayu_gnn.dbx.dbx_config import dbx_helper

# Start capturing logs
capture_logs = CaptureLogs(dbx_helper=dbx_helper, dbx_path="/output/logs")

try:

    from vayu_gnn.model_run.setup.execution_params import download_bool, preprocess_bool

    # Overall time and a timer for each process initialized
    full_timer = StepTimer()

    # STEP 1: Download data
    if download_bool:
        
        download_timer = StepTimer()  
        from vayu_gnn.pipeline.a_download import execute as execute_download
        from vayu_gnn.model_run.setup.download_params import download_data, download_params, urls, start_date, end_date, nodes, cities

        execute_download(dbx_helper, download_data, urls, download_params, start_date, end_date, nodes, cities)

        download_timer.stop('download process')

    else:
        print("You are not downloading any data for this run.")

    # STEP 2: Preprocess data
    if preprocess_bool:

        preprocess_timer = StepTimer() 

        from vayu_gnn.pipeline.b_preprocess import execute as execute_preprocess
        from vayu_gnn.model_run.setup.preprocessor_params import preprocess_data, preprocess_params

        execute_preprocess(dbx_helper=dbx_helper, preprocess_data=preprocess_data, preprocessor_params=preprocess_params)

        preprocess_timer.stop('preprocessing')

    else:
        print("You are not preprocessing any data for this run.")


    full_timer.stop('full pipeline')

except Exception as e:
    print(f"Exception occurred: {e}")
    
# Stop capturing logs
capture_logs.capture(on=False)

# Save logs to Dropbox
capture_logs.save_logs_to_dropbox()
# %%
