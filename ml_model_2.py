import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# --- 1. Load the Data ---
try:
    # Ensure this is the file generated in the previous step
    df = pd.read_csv('bulk_investment_data.csv') 
except FileNotFoundError:
    print("Error: 'bulk_investment_data.csv' not found. Please ensure it's in the correct folder.")
    exit()

# --- 2. Data Preprocessing: Convert Machine_Model (Text) to Numbers ---
# This step is crucial for the ML model to understand the different machine types
df = pd.get_dummies(df, columns=['Machine_Model'], drop_first=True)


# --- 3. Define Features (X) and Multiple Targets (y) ---
# The features now include the new one-hot encoded columns (e.g., Machine_Model_Lathe Y-B2)
# We exclude the target columns when defining features (X)
target_cols = [
    'Actual_Lifespan_Yrs',
    'Historical_Total_Maint_Cost',
    'Task_Completion_Time_Hrs'
]
X = df.drop(columns=target_cols)
y = df[target_cols]


# --- 4. Split the Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")


# --- 5. Initialize and Train the Multi-Output Model ---
base_estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
multi_output_model = MultiOutputRegressor(base_estimator)

print("\nStarting model training...")
multi_output_model.fit(X_train, y_train)
print("Model training complete!")


# --- 6. Make Predictions ---
y_pred = multi_output_model.predict(X_test)


# --- 7. Evaluate the Model (Per Target) ---
print("\n--- Model Performance on Test Data ---")
for i, target in enumerate(target_cols):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"**Target: {target}**")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  R-squared (R2 Score): {r2:.4f}")


# --- 8. Example Prediction for the Comparison/Formula Step ---
# --- IMPORTANT: These test inputs MUST match the structure used for training! ---
# We simulate data for three competing machines (A, B, C) corresponding to Lathe X-A1, Y-B2, and Z-C3
new_machine_data = pd.DataFrame({
    'Initial_Cost_USD': [250000, 350000, 450000],
    'Power_Consumption_kWh_hr': [25.0, 30.0, 40.0],
    'Max_Capacity_Units_hr': [120, 150, 180],
    'Scheduled_Maint_Interval_Days': [60, 90, 180],
    'Historical_Failure_Rate': [0.10, 0.05, 0.02],
    # Features for one-hot encoding (All models except the base one are set to 0 initially)
    'Machine_Model': ['Lathe X-A1', 'Lathe Y-B2', 'Lathe Z-C3']
})

# Preprocess the test data the same way the training data was processed
new_machine_data_processed = pd.get_dummies(new_machine_data.drop(columns=['Machine_Model']), 
                                            columns=[], # No columns to encode as we handle it manually below
                                            drop_first=True)

# Manually add the one-hot encoded columns for the three models
# If the model is 'Lathe Z-C3', set that column to 1, others to 0.
new_machine_data_processed['Machine_Model_Lathe Y-B2'] = [0, 1, 0] 
new_machine_data_processed['Machine_Model_Lathe Z-C3'] = [0, 0, 1] 

# The remaining model, 'Lathe X-A1', is represented by both columns being 0.

predicted_outputs = multi_output_model.predict(new_machine_data_processed)
predictions_df = pd.DataFrame(predicted_outputs, columns=target_cols)
predictions_df.insert(0, 'Machine_Model', ['Model A (X-A1)', 'Model B (Y-B2)', 'Model C (Z-C3)'])

print("\n--- ML Predicted Outputs for Financial Comparison ---")
print(predictions_df.to_string(index=False))

# --- 9. Financial Formulas and Comparison ---

# --- Define fixed financial parameters for comparison (can be adjusted by user/company) ---
BULK_QUANTITY = 1000  # As specified in the scope
TIME_HORIZON_YRS = 5  # Standard period for comparison
TARGET_REVENUE_PER_UNIT_YR = 80000 # NEW VALUE: Estimated annual revenue per machine unit
ANNUAL_OPERATING_COST_USD = 5000  # Annual cost (labor, utilities, etc.) per machine

# Create a consolidated DataFrame for easy comparison
comparison_df = new_machine_data.copy()
comparison_df['Machine_Model'] = ['Model A (X-A1)', 'Model B (Y-B2)', 'Model C (Z-C3)']
comparison_df['Predicted_Lifespan'] = predictions_df['Actual_Lifespan_Yrs']
comparison_df['Predicted_Maint_Cost'] = predictions_df['Historical_Total_Maint_Cost']
comparison_df['Predicted_Task_Time'] = predictions_df['Task_Completion_Time_Hrs']

# --- Calculate Key Financial Metrics ---

# Total Cost of Ownership (TCO) over the TIME_HORIZON_YRS
# TCO = Initial_Cost + (Annual_Op_Cost * Time_Horizon) + Predicted_Maint_Cost_Annualized
comparison_df['Annual_Maint_Cost'] = comparison_df['Predicted_Maint_Cost'] / comparison_df['Predicted_Lifespan']
comparison_df['TCO_5Yrs_Per_Unit'] = (
    comparison_df['Initial_Cost_USD'] + 
    (ANNUAL_OPERATING_COST_USD * TIME_HORIZON_YRS) + 
    (comparison_df['Annual_Maint_Cost'] * TIME_HORIZON_YRS)
)

# Total Revenue over the TIME_HORIZON_YRS
comparison_df['Total_Revenue_5Yrs_Per_Unit'] = TARGET_REVENUE_PER_UNIT_YR * TIME_HORIZON_YRS

# Net Profit / Net Savings
comparison_df['Net_Profit_5Yrs_Per_Unit'] = (
    comparison_df['Total_Revenue_5Yrs_Per_Unit'] - 
    comparison_df['TCO_5Yrs_Per_Unit']
)

# Return on Investment (ROI) - calculated over the initial cost
comparison_df['ROI_5Yrs'] = (
    comparison_df['Net_Profit_5Yrs_Per_Unit'] / 
    comparison_df['Initial_Cost_USD']
) * 100 # ROI in percent

# --- Scale Metrics to the Bulk Quantity ---
comparison_df['Bulk_Total_TCO_5Yrs'] = comparison_df['TCO_5Yrs_Per_Unit'] * BULK_QUANTITY
comparison_df['Bulk_Total_Profit_5Yrs'] = comparison_df['Net_Profit_5Yrs_Per_Unit'] * BULK_QUANTITY


# --- Final Recommendation Logic ---
best_model = comparison_df.loc[comparison_df['Net_Profit_5Yrs_Per_Unit'].idxmax()]


print("\n=======================================================")
print("  💰 STEP 4: FINANCIAL COMPARISON (5-Year Horizon) 💰")
print("=======================================================")
print(f"ASSUMPTIONS: Bulk Quantity={BULK_QUANTITY}, Time Horizon={TIME_HORIZON_YRS} Yrs, Annual Revenue/Unit=${TARGET_REVENUE_PER_UNIT_YR}")

# Print Comparison Table
comparison_output = comparison_df[['Machine_Model', 'Initial_Cost_USD', 'Predicted_Lifespan', 'Annual_Maint_Cost', 'Net_Profit_5Yrs_Per_Unit', 'ROI_5Yrs']].copy()
comparison_output['Initial_Cost_USD'] = comparison_output['Initial_Cost_USD'].map('${:,.0f}'.format)
comparison_output['Annual_Maint_Cost'] = comparison_output['Annual_Maint_Cost'].map('${:,.0f}'.format)
comparison_output['Net_Profit_5Yrs_Per_Unit'] = comparison_output['Net_Profit_5Yrs_Per_Unit'].map('${:,.0f}'.format)
comparison_output['ROI_5Yrs'] = comparison_output['ROI_5Yrs'].map('{:.2f}%'.format)
comparison_output.rename(columns={
    'Initial_Cost_USD': 'Initial Cost',
    'Predicted_Lifespan': 'Pred. Life (Yrs)',
    'Annual_Maint_Cost': 'Annual Maint.',
    'Net_Profit_5Yrs_Per_Unit': 'Net Profit/Unit',
    'ROI_5Yrs': '5-Yr ROI'
}, inplace=True)
print("\n--- Comparative Financial Metrics (Per Unit) ---")
print(comparison_output.to_string(index=False))


print("\n--- FINAL INVESTMENT RECOMMENDATION ---")
print(f"The recommended model for a bulk investment of {BULK_QUANTITY} units is: **{best_model['Machine_Model']}**")
print(f"This model is projected to yield a **Net Profit** of **${best_model['Bulk_Total_Profit_5Yrs']:,.0f}** over 5 years.")
print(f"Its **5-Year ROI** is **{best_model['ROI_5Yrs']:.2f}%**.")
print("=======================================================")