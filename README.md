# Weekly Sales Forecasting Model

This project contains the pipeline for forecasting weekly sales quantities. It preprocesses raw transaction, store, and product data, uses a pre-trained LightGBM model to make predictions, and evaluates its performance on new, unseen data using the Weighted Mean Absolute Percentage Error (WMAPE) metric.

---
## Setup

Before running the scripts, set up the required environment.

1.  **Clone the repository** (or download the files).
2.  **Install the dependencies** by running the following command in your terminal from the project's root directory:
    ```bash
    pip install -r requirements.txt
    ```

---
## Workflow

Follow these steps to evaluate the model on a new set of data.

### Step 1: Add Your Data

1.  Extract your data to a folder in the project's directory.
2.  Unzip your new raw data and place the three Parquet files (e.g., `part27.snappy.parquet`, `part51.snappy.parquet`, `part71.snappy.parquet`) inside this folder, it can be the data folder.
3. Update the info in the preprocessing script to use the same path you unzipped it to, change the file names too to the ones in your zipped file:

```bash
    # Load the three Parquet files from the specified folder ---
    try:
        df_stores = pd.read_parquet('simulated_january_data/part27.snappy.parquet')
        df_transactions = pd.read_parquet('simulated_january_data/part51.snappy.parquet')
        df_products = pd.read_parquet('simulated_january_data/part71.snappy.parquet')
        print("Successfully loaded the three Parquet files from 'simulated_january_data/'.")
    except FileNotFoundError:
        print("Error: Could not find the required Parquet files in the 'simulated_january_data/' folder.")
        return

    final_df = create_modeling_dataset(df_stores, df_transactions, df_products)
```

### Step 2: Run the Preprocessing Script

This script will find the raw Parquet files in your folder, process them, and save the result.

In your terminal, run:
```bash
python preprocess.py
```
This will create a single, model-ready test.csv file inside the data/ folder.

### Step 3: Run the Evaluation Script

This script uses the test.csv file you just created to score the model. In your terminal, run:

```bash
python evaluate.py
```
