import pandas as pd
import numpy as np
import os

def create_modeling_dataset(df_stores, df_transactions, df_products):
    # Rename Keys for Clean Merging
    df_stores.rename(columns={'pdv': 'internal_store_id'}, inplace=True)
    df_products.rename(columns={'produto': 'internal_product_id'}, inplace=True)

    # Merge the Three DataFrames
    train = pd.merge(df_transactions, df_stores, on='internal_store_id', how='left')
    train = pd.merge(train, df_products, on='internal_product_id', how='left')

    # Initial Cleaning and Type Conversion
    train['transaction_date'] = pd.to_datetime(train['transaction_date'])
    train.dropna(subset=['subcategoria', 'premise'], inplace=True)

    # Feature Engineering (on Raw Data)
    train['semana'] = train['transaction_date'].dt.isocalendar().week
    train['ano'] = train['transaction_date'].dt.isocalendar().year
    train['mes'] = train['transaction_date'].dt.month

    # Aggregate to the Weekly Level
    df_weekly = train.groupby([
        'ano', 'semana', 'internal_store_id', 'internal_product_id'
    ]).agg(
        quantidade=('quantity', 'sum'),
        avg_net_value=('net_value', 'mean'),
        total_discount=('discount', 'sum'),
        n_transacoes=('transaction_date', 'nunique'),
        mes=('mes', 'first')
    ).reset_index()

    # Add Back Static Features
    product_features = train[['internal_product_id', 'marca', 'categoria', 'fabricante']].drop_duplicates()
    store_features = train[['internal_store_id', 'premise', 'categoria_pdv', 'zipcode']].drop_duplicates()
    df_model = pd.merge(df_weekly, product_features, on='internal_product_id', how='left')
    df_model = pd.merge(df_model, store_features, on='internal_store_id', how='left')

    # Create Advanced Time-Series Features
    df_model.sort_values(by=['internal_store_id', 'internal_product_id', 'ano', 'semana'], inplace=True)
    
    # Use the specific name 'quantidade_semana_passada' for the 1-week lag
    df_model['quantidade_semana_passada'] = df_model.groupby(
        ['internal_store_id', 'internal_product_id']
    )['quantidade'].shift(1)
    
    df_model['media_movel_4_semanas'] = df_model.groupby(
        ['internal_store_id', 'internal_product_id']
    )['quantidade'].transform(lambda x: x.shift(1).rolling(4).mean())
    df_model.fillna(0, inplace=True)

    print("Preprocessing complete.")
    return df_model

def main():
    """Main function to run the data processing pipeline."""
    print("Starting the preprocessing script...")
    
    # Load the three Parquet files from the specified folder
    try:
        df_stores = pd.read_parquet('data/simulated_january_data/part27.snappy.parquet')
        df_transactions = pd.read_parquet('data/simulated_january_data/part51.snappy.parquet')
        df_products = pd.read_parquet('data/simulated_january_data/part71.snappy.parquet')
        print("Successfully loaded the three Parquet files from 'simulated_january_data/'.")
    except FileNotFoundError:
        print("Error: Could not find the required Parquet files in the 'simulated_january_data/' folder.")
        return

    final_df = create_modeling_dataset(df_stores, df_transactions, df_products)

    output_folder = 'data'
    output_path = os.path.join(output_folder, 'test.csv')
    os.makedirs(output_folder, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nPreprocessing complete! Saved final data to: {output_path}")

if __name__ == "__main__":
    main()