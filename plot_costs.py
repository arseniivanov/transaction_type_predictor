import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

label_mapping = {
    'ent': 'Entertainment, alcohol',
    'food': 'Groceries/Food',
    'hobby': 'Hobbies and Home',
    'loan': 'Loan',
    'out': 'Eating out',
    'rent': 'Rent',
    'sub': 'Subscriptions',
    'trans': 'Transport',
    'trip': 'Travel'
}

# Load the data
file_path = 'Transactions.ods'  # Update to your actual file path
data = pd.read_excel(file_path, engine='odf')

# Set the correct headers and remove the header row from the data
data.columns = data.iloc[0]
data = data[1:]

# Convert 'Transaktionsdag' to datetime and 'Belopp' to absolute values
data['Transaktionsdag'] = pd.to_datetime(data['Transaktionsdag'])
data['Belopp'] = data['Belopp'].astype(float).abs()

# Create a new column for Year-Month
data['YearMonth'] = data['Transaktionsdag'].dt.to_period('M').astype(str)

shared_categories = ['food', 'ent', 'trip', 'out']  # Add more as needed
categories_to_drop = ['meds', 'pb', 'save']  # Add more as needed

for category in shared_categories:
    data.loc[data['Label'] == category, 'Belopp'] /= 2

# Group by 'YearMonth' and 'Label', then sum 'Belopp'
monthly_costs = data.groupby(['YearMonth', 'Label'])['Belopp'].sum().reset_index()

monthly_costs = monthly_costs[~monthly_costs['Label'].isin(categories_to_drop)]

monthly_costs['Full Label'] = monthly_costs['Label'].map(label_mapping)

# Pivot the data for plotting
pivot_table = monthly_costs.pivot(index='YearMonth', columns='Full Label', values='Belopp')

# Fill NA values with 0 for plotting and cumulatively sum across each category
pivot_table_filled = pivot_table.fillna(0)

pivot_table_filled = pivot_table_filled[:-1]

# Plotting
plt.figure(figsize=[15,7])
plt.stackplot(pivot_table_filled.index, pivot_table_filled.T, labels=pivot_table_filled.columns, alpha=0.8)

plt.axhline(y=30000, color='r', linestyle='--', linewidth=2, label='Income')

plt.title('Cumulative Monthly Costs by Category')
plt.xlabel('Month')
plt.ylabel('Cumulative Costs (SEK)')
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
