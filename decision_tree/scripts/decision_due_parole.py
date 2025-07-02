#This script performs supervised classification to detect segmentation points in tokenized text
    #It includes the following steps:
        #- Data shuffling and sentence-level split into train, validation, and test sets
        #- Feature scaling using StandardScaler for numerical features
        #- Balancing the training set with SMOTE (Synthetic Minority Over-sampling Technique)
        #- Training a Decision Tree Classifier
        #- Evaluation on both validation and test sets with classification reports and confusion matrices

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load the preprocessed DataFrame from a pickle file
    with open('data/output_preprocessing/dueparole/parole.pkl', 'rb') as file:
        df_caricato = pickle.load(file)

    # Shuffle the full DataFrame to remove any ordering bias
    df_caricato = shuffle(df_caricato, random_state=42)

    # Get unique sentence indices and shuffle them
    frasi_idx = df_caricato['frase_idx'].unique()
    frasi_idx = shuffle(frasi_idx, random_state=42)

    # Split sentence indices: 90% for training+validation, 10% for testing
    frasi_trainval, frasi_test = train_test_split(frasi_idx, test_size=0.1, random_state=42)

    # From training+validation, split 10% for validation (i.e. 9% of total data)
    frasi_train, frasi_val = train_test_split(frasi_trainval, test_size=0.1, random_state=42)

    # Filter DataFrames by sentence indices
    df_train = df_caricato[df_caricato['frase_idx'].isin(frasi_train)]
    df_val = df_caricato[df_caricato['frase_idx'].isin(frasi_val)]
    df_test = df_caricato[df_caricato['frase_idx'].isin(frasi_test)]

    # Separate features (X) and labels (y)
    X_train = df_train.drop(columns=['segmenta', 'frase_idx', 'token'])
    y_train = df_train['segmenta']

    X_val = df_val.drop(columns=['segmenta', 'frase_idx', 'token'])
    y_val = df_val['segmenta']

    X_test = df_test.drop(columns=['segmenta', 'frase_idx', 'token'])
    y_test = df_test['segmenta']

    # Print the size of each dataset
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # Define numerical and binary columns (POS tags)
    numerical_cols = ['frase_len_token', 'frase_len_char', 'token_len_char', 'distanza_da_prima_parola']
    binary_cols = ['NOUN', 'VERB', 'ADJ', 'DET', 'ADP', 'PRON', 'PUNCT', 'ADV', 'NUM', 'CCONJ']

    # Create a column transformer to scale only numerical columns
    preprocessor = ColumnTransformer([
        ('scaler', StandardScaler(), numerical_cols)
    ], remainder='passthrough')  # Keep binary columns unchanged

    # Fit scaler on training data and transform all sets
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    # Convert back to DataFrames for convenience
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=numerical_cols + binary_cols, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=numerical_cols + binary_cols, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=numerical_cols + binary_cols, index=X_test.index)

    # Apply SMOTE on training data to balance class distribution
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    print(f"Training set size after SMOTE: {len(X_train_res)}")

    # Train a Decision Tree classifier on the resampled training set
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_res, y_train_res)

    # Evaluate the model on the validation set
    y_pred_val = clf.predict(X_val_scaled)
    print("\nClassification Report (Validation Set):")
    print(classification_report(y_val, y_pred_val))

    # Evaluate the model on the test set
    y_test_pred = clf.predict(X_test_scaled)
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    # Plot confusion matrix for validation set
    cm_val = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Validation Set')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

    # Plot confusion matrix for test set
    cm_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

    # Extract original sentences corresponding to test set indices

    # Get unique sentence indices from test set
    frasi_test_set = df_test['frase_idx'].unique()
    print("\nSentence indices in test set:", frasi_test_set)

    # Load original sentences from the corpus file into a dictionary
    frasi_originali = {}
    with open("data/output_preprocessing/dueparole/corpus_in_frasi_no_seg.txt", "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                print(f"Line {line_num} without tab: '{line}'")
                continue
            idx_str, frase = line.split("\t", 1)
            frasi_originali[int(idx_str)] = frase

    # Select only sentences whose indices are in the test set and exist in the dictionary
    frasi_selezionate = {idx: frasi_originali[idx] for idx in frasi_test_set if idx in frasi_originali}

    # Print some selected sentences as a check
    for idx, frase in frasi_selezionate.items():
        print(f"{idx}\t{frase}")

    # Save selected sentences to a new file
    with open("data/output_preprocessing/due_parole/frasi_test_set_due_parole.txt", "w", encoding="utf-8") as f_out:
        for idx, frase in frasi_selezionate.items():
            f_out.write(f"{idx}\t{frase}\n")


if __name__ == "__main__":
    main()
