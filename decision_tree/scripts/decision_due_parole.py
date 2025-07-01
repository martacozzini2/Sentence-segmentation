#This script performs supervised classification to detect segmentation points in tokenized text
    #It includes the following steps:
        #- Data shuffling and sentence-level split into train, validation, and test sets
        #- Feature scaling using StandardScaler for numerical features
        #- Balancing the training set with SMOTE (Synthetic Minority Over-sampling Technique)
        #- Training a Decision Tree Classifier
        #- Evaluation on both validation and test sets with classification reports and confusion matrices

import pickle

# Load the preprocessed DataFrame from a pickle file
with open('/content/parole.pkl', 'rb') as file:
    df_caricato = pickle.load(file)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#Shuffle the full DataFrame to remove any ordering bias
df_caricato = shuffle(df_caricato, random_state=42)

#Get unique sentence indices and shuffle them
frasi_idx = df_caricato['frase_idx'].unique()
frasi_idx = shuffle(frasi_idx, random_state=42)

#Split sentence indices: 90% for training + validation, 10% for test
frasi_trainval, frasi_test = train_test_split(frasi_idx, test_size=0.1, random_state=42)

#From training+validation, split 10% for validation (i.e. 9% of total data)
frasi_train, frasi_val = train_test_split(frasi_trainval, test_size=0.1, random_state=42)

#Filter DataFrames by sentence indices
df_train = df_caricato[df_caricato['frase_idx'].isin(frasi_train)]
df_val   = df_caricato[df_caricato['frase_idx'].isin(frasi_val)]
df_test  = df_caricato[df_caricato['frase_idx'].isin(frasi_test)]

#Separate features (X) and labels (y)
X_train = df_train.drop(columns=['segmenta', 'frase_idx', 'token'])
y_train = df_train['segmenta']

X_val = df_val.drop(columns=['segmenta', 'frase_idx', 'token'])
y_val = df_val['segmenta']

X_test = df_test.drop(columns=['segmenta', 'frase_idx', 'token'])
y_test = df_test['segmenta']

# 7. Define which columns are numerical and which are binary (for POS tags)
numerical_cols = ['frase_len_token', 'frase_len_char', 'token_len_char', 'distanza_da_prima_parola']
binary_cols = ['NOUN', 'VERB', 'ADJ', 'DET', 'ADP', 'PRON', 'PUNCT', 'ADV', 'NUM', 'CCONJ']

# Create a column transformer to scale only the numerical columns
preprocessor = ColumnTransformer([
    ('scaler', StandardScaler(), numerical_cols)
], remainder='passthrough')  # Keep other columns (binary) as they are

# Apply preprocessing: fit on training, transform all sets
X_train_scaled = preprocessor.fit_transform(X_train)
X_val_scaled = preprocessor.transform(X_val)
X_test_scaled = preprocessor.transform(X_test)

# Reconvert the results to DataFrames with column names and matching indices
X_train_scaled = pd.DataFrame(X_train_scaled, columns=numerical_cols + binary_cols, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=numerical_cols + binary_cols, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=numerical_cols + binary_cols, index=X_test.index)

# 8. Apply SMOTE only on the training set to balance class distribution
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# 9. Train a Decision Tree classifier on the resampled training set
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_res, y_train_res)

# 10. Evaluate on the validation set
y_pred_val = clf.predict(X_val_scaled)

print("\nClassification Report (Validation Set):")
print(classification_report(y_val, y_pred_val))

# Predict on the test set
y_test_pred = clf.predict(X_test_scaled)

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# 11. Plot confusion matrix for the validation set
cm = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Validation')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

# Plot confusion matrix for the test set
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title('Confusion Matrix - Test')
plt.xlabel('Predicted Classe')
plt.ylabel('True Class')
plt.show()

#This second part of the code uses the 'frase_idx' values from the test set to extract only the corresponding sentences from the full corpus.
#These sentences form the due parole test set, which will later be used to evaluate our LLM model
#The selected sentences are written to a new file

frasi_test_set = df_test['frase_idx'].unique()
print(frasi_test_set)

frasi_originali = {}
with open("/content/corpus_in_frasi_no_seg.txt", "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        if "\t" not in line:
            print(f"Riga {line_num} senza tab: '{line}'")
            continue
        idx_str, frase = line.split("\t", 1)
        frasi_originali[int(idx_str)] = frase

indici_da_estrarre = df_test['frase_idx'].unique()

frasi_selezionate = {idx: frasi_originali[idx] for idx in indici_da_estrarre if idx in frasi_originali}

for idx, frase in frasi_selezionate.items():
    print(f"{idx}\t{frase}")

with open("/content/frasi_test_set_due_parole.txt", "w", encoding="utf-8") as f_out:
    for idx, frase in frasi_selezionate.items():
        f_out.write(f"{idx}\t{frase}\n")
