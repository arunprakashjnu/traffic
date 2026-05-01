import pandas as pd

# Read Excel file with correct header
excel_file = 'GEN_ALLOTMENT_ROUND_2_2025.xlsx'
df = pd.read_excel(excel_file, sheet_name=0, skiprows=7)

# Clean column names
df.columns = ['SNO', 'APPLN_NO', 'CUTOFF', 'RANK', 'COMMUNITY', 'COLLEGE_CODE', 'BRANCH_CODE', 'ALLOTTED_COMMUNITY', 'Extra']
df = df.drop('Extra', axis=1)

# Remove any completely empty rows
df = df.dropna(how='all')

print(f'Shape: {df.shape[0]} rows × {df.shape[1]} columns')
print(f'\nColumn Names: {list(df.columns)}')
print(f'\nFirst 5 rows:')
print(df.head().to_string())
print(f'\n--- Value Analysis ---')
print(f'COMMUNITY unique values: {sorted(df["COMMUNITY"].dropna().unique())}')
print(f'CUTOFF range: {df["CUTOFF"].min()} to {df["CUTOFF"].max()}')
print(f'Total records: {len(df)}')

# Save to CSV
csv_file = 'allotment_data.csv'
df.to_csv(csv_file, index=False)
print(f'\n✓ Converted to {csv_file}')
