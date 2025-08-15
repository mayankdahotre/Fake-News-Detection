import pandas as pd

# Load datasets
df_true = pd.read_csv('True.csv')
df_fake = pd.read_csv('Fake.csv')

# Add labels
df_true['label'] = 'REAL'
df_fake['label'] = 'FAKE'

# If the text column is named differently (e.g., 'title' + 'text'), combine them
if 'text' in df_true.columns:
    df_true['text'] = df_true['text'].astype(str)
else:
    df_true['text'] = df_true['title'].astype(str) + " " + df_true['content'].astype(str)

if 'text' in df_fake.columns:
    df_fake['text'] = df_fake['text'].astype(str)
else:
    df_fake['text'] = df_fake['title'].astype(str) + " " + df_fake['content'].astype(str)

# Select only 'text' and 'label'
df_true = df_true[['text', 'label']]
df_fake = df_fake[['text', 'label']]

# Combine and shuffle
df = pd.concat([df_true, df_fake], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Save as dataset.csv
df.to_csv('dataset.csv', index=False)

print(f"✅ dataset.csv created with {len(df)} records.")
