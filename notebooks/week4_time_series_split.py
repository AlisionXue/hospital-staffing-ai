import pandas as pd

# 1. Load the dataset from Week 3
df = pd.read_csv('data/hospital_week3_clustered.csv')

# 2. Convert quarter to datetime format for sorting and lag features
df['quarter'] = df['quarter'].astype(str)
df['quarter_dt'] = pd.PeriodIndex(df['quarter'], freq='Q').to_timestamp()

# 3. Sort data by hospital_branch and quarter
df = df.sort_values(['hospital_branch', 'quarter_dt'])

# 4. Create lag features
df['treatment_count_lag1'] = df.groupby('hospital_branch')['treatment_count'].shift(1)
df['efficiency_lag1'] = df.groupby('hospital_branch')['efficiency'].shift(1)

# 5. Drop rows with missing lag values (i.e., Q1s)
df_cleaned = df.dropna(subset=['treatment_count_lag1', 'efficiency_lag1']).copy()

# 6. Create time-based splits
# Train = Q1 & Q2 | Validation = Q3 | Test = Q4
df_cleaned['dataset_split'] = df_cleaned['quarter'].apply(lambda q: 
    'train' if q in ['2023Q1', '2023Q2'] else (
        'val' if q == '2023Q3' else (
            'test' if q == '2023Q4' else 'unknown'
        )
    )
)

# 7. Save the processed dataset
output_path = 'data/hospital_week4_timeseries_lagged.csv'
df_cleaned.to_csv(output_path, index=False)

# 8. Show summary
print("✅ Week 4 preprocessing completed.")
print("Saved to:", output_path)
print("\nSplit counts:")
print(df_cleaned['dataset_split'].value_counts())
