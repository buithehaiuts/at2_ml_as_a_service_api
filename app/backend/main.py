from fastapi import FastAPI
from api import SalesAPI
import uvicorn

if __name__ == "__main__":
    sales_api = SalesAPI()  # Create an instance of SalesAPI

    # Start the FastAPI app in a separate thread
    import threading

    def run_app():
        uvicorn.run(sales_api.app, host="127.0.0.1", port=8000, log_level="info")

    # Start the FastAPI app
    app_thread = threading.Thread(target=run_app)
    app_thread.start()

    # Wait for the server to start
    import time
    time.sleep(1)

    # Directly access and display the training DataFrame
    print("Training DataFrame:")
    print(sales_api.data_loader.display_dataframe(sales_api.data_loader.train_df))

    app_thread.join()  # Wait for the FastAPI app to finish (if needed)
