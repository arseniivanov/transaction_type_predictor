import pandas as pd

file_path = 'Transactions.ods'  # Replace 'YourFileName.ods' with the actual file name
data = pd.read_excel(file_path, engine='odf')

# Correcting the header row and parsing the data correctly
data.columns = data.iloc[0] # Set the first row as column names
data = data[1:] # Remove the first row from the data

# Now, let's convert the "Label" field into a dictionary that counts the occurrence of each entry
label_counts = data['Label'].value_counts().to_dict()

# Display the resulting dictionary
print(label_counts)

