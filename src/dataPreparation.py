import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset 
file_path = 'Path-to-dataset'

# Simple error checking
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
except pd.errors.ParserError:
    print(f"Error: There was a problem parsing the file at {file_path}.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Copy the dataset to avoid modifying the original data
data_processed = data.copy()

# Step 1: Encode binary categorical features (Yes/No columns) as 1 and 0
binary_columns = data_processed.columns[data_processed.isin(['Yes', 'No']).any()]  # Columns with Yes/No values
for col in binary_columns:
    data_processed[col] = data_processed[col].map({'Yes': 1, 'No': 0})

# Step 2: Encode non-binary categorical features, specifically "Gender" and "class"
# Encoding "Gender" column: Male=1, Female=0
data_processed['Gender'] = data_processed['Gender'].map({'Male': 1, 'Female': 0})

# Encoding "class" column: Positive=1, Negative=0
data_processed['class'] = data_processed['class'].map({'Positive': 1, 'Negative': 0})

# Step 3: Splitting data into features (X) and target (y)
X = data_processed.drop(columns='class')
y = data_processed['class']

# Step 4: Splitting the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
