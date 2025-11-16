import pandas as pd
import numpy as np

# Set the seed for reproducibility
np.random.seed(42)

# --- Define Parameters ---
N_SAMPLES = 150 # Increased sample size for better training
MIN_COST = 200000
MAX_COST = 500000
MODELS = ['Lathe X-A1', 'Lathe Y-B2', 'Lathe Z-C3'] # Simulating 3 competing models

# --- 1. Generate Input Features (X) ---
# Create a machine model identifier
model_assignment = np.random.choice(MODELS, N_SAMPLES)
data = {
    'Machine_Model': model_assignment,
    'Initial_Cost_USD': np.random.randint(MIN_COST, MAX_COST, N_SAMPLES),
    'Power_Consumption_kWh_hr': np.round(np.random.uniform(10, 50, N_SAMPLES), 1),
    'Max_Capacity_Units_hr': np.random.randint(50, 200, N_SAMPLES),
    'Scheduled_Maint_Interval_Days': np.random.choice([30, 60, 90, 180], N_SAMPLES),
    'Historical_Failure_Rate': np.round(np.random.uniform(0.01, 0.15, N_SAMPLES), 3),
}

df = pd.DataFrame(data)

# --- Add inherent bias based on the model type for more realistic prediction targets ---
model_lifespan_bias = {'Lathe X-A1': -1.5, 'Lathe Y-B2': 0, 'Lathe Z-C3': 1.5}
df['Lifespan_Bias'] = df['Machine_Model'].map(model_lifespan_bias)

model_maint_bias = {'Lathe X-A1': 30000, 'Lathe Y-B2': 0, 'Lathe Z-C3': -30000}
df['Maint_Cost_Bias'] = df['Machine_Model'].map(model_maint_bias)


# --- 2. Generate Target Outputs (y) based on Features and Bias ---
# Lifespan: High cost, low failure rate, and low power lead to longer life, plus model bias
df['Actual_Lifespan_Yrs'] = np.round(
    15 - (df['Initial_Cost_USD'] / MAX_COST * 5) - (df['Historical_Failure_Rate'] * 30) + (df['Power_Consumption_kWh_hr'] / 50 * 2) + df['Lifespan_Bias'] + np.random.normal(0, 1.5, N_SAMPLES), 1
)
df['Actual_Lifespan_Yrs'] = np.clip(df['Actual_Lifespan_Yrs'], 5, 15)

# Maintenance Cost: High failure rate, high power, and long interval lead to higher cost, plus model bias
df['Historical_Total_Maint_Cost'] = np.round(
    (df['Historical_Failure_Rate'] * 500000) + (df['Power_Consumption_kWh_hr'] * 1500) + (1 / df['Scheduled_Maint_Interval_Days'] * 500000) + df['Maint_Cost_Bias'] + np.random.normal(0, 20000, N_SAMPLES), 0
)

# Task Completion Time: Inverse of capacity, adjusted by power and failure rate
df['Task_Completion_Time_Hrs'] = np.round(
    20000 / df['Max_Capacity_Units_hr'] + (df['Power_Consumption_kWh_hr'] * 0.5) + (df['Historical_Failure_Rate'] * 50) + np.random.normal(0, 5, N_SAMPLES), 1
)

# Remove temporary bias columns before saving
df = df.drop(columns=['Lifespan_Bias', 'Maint_Cost_Bias'])

# --- 3. Save the Data ---
OUTPUT_FILE = 'bulk_investment_data.csv'
df.to_csv(OUTPUT_FILE, index=False)
print(f"Successfully generated {N_SAMPLES} samples including {len(MODELS)} models and saved to: {OUTPUT_FILE}")

# Display the first few rows to verify
print("\nFirst 5 rows of the generated data:")
print(df.head())