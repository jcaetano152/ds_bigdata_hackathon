import pandas as pd
import numpy as np
import joblib
import os

def wmape(y_true, y_pred):
    """Calculates the Weighted Mean Absolute Percentage Error (WMAPE)."""
    sum_abs_err = np.sum(np.abs(y_true - y_pred))
    sum_actual = np.sum(np.abs(y_true))
    return sum_abs_err / (sum_actual + 1e-10)

def main():
    print("Starting evaluation script...")

    model_assets_path = os.path.join('models', 'lgbm_model.pkl')
    data_path = os.path.join('data', 'test.csv')
    results_folder = 'results'
    
    try:
        loaded_assets = joblib.load(model_assets_path)
        model = loaded_assets['model']
        train_categories = loaded_assets['categories']
        print(f"Successfully loaded model and categories from '{model_assets_path}'")
        
        df_to_evaluate = pd.read_csv(data_path)
        print(f"Successfully loaded preprocessed data from '{data_path}'")
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    features = model.feature_name_
    target = 'quantidade'
    
    X_eval = df_to_evaluate[features].copy()
    y_actual = df_to_evaluate[target].copy()
    
    print("Aligning categorical features with training data...")
    for col, cats in train_categories.items():
        if col in X_eval.columns:
            X_eval[col] = pd.Categorical(X_eval[col], categories=cats)

    numerical_cols = X_eval.select_dtypes(include=np.number).columns
    X_eval[numerical_cols] = X_eval[numerical_cols].fillna(0)
    
    print("Making predictions on the new data...")
    predictions = model.predict(X_eval)

    wmape_score = wmape(y_actual, predictions)
    print("\n--- EVALUATION COMPLETE ---")
    print(f"Model Performance on New Data (WMAPE): {wmape_score:.4f}")

    # Save results for inspection
    results_df = df_to_evaluate[['ano', 'semana', 'internal_store_id', 'internal_product_id']].copy()
    results_df['actual_quantity'] = y_actual
    results_df['predicted_quantity'] = predictions.round().astype(int)
    
    os.makedirs(results_folder, exist_ok=True)
    results_path = os.path.join(results_folder, 'evaluation_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8')
    print(f"Saved evaluation results to '{results_path}'")

if __name__ == "__main__":
    main()