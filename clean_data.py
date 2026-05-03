import pandas as pd

print("Loading raw Kaggle dataset...")
# Load the messy Kaggle data
df = pd.read_csv('dataset.csv')

# Fill empty symptom boxes with blank spaces
df = df.fillna('')

print("Combining and cleaning symptoms...")
# Grab all the columns that contain the word 'Symptom'
symptom_columns = [col for col in df.columns if 'Symptom' in col]

# Combine all symptoms into a single text string for each patient
df['symptoms_text'] = df[symptom_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Clean up the text (Kaggle data often has underscores like 'skin_rash')
df['symptoms_text'] = df['symptoms_text'].str.replace('_', ' ').str.strip()

# Create a clean, final dataframe with just what our ML model needs
final_df = pd.DataFrame({
    'symptoms': df['symptoms_text'],
    'disease': df['Disease']
})

# Save it over our dataset file
final_df.to_csv('dataset.csv', index=False)
print("Data cleaned successfully! Your dataset is ready for training.")