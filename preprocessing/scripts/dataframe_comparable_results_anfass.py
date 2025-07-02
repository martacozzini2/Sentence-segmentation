
# ==========================================================
# Script for processing segmented sentences with spaCy
# - Loads sentences with segments marked by <seg>
# - Analyzes sentences with spaCy
# - Builds a DataFrame with token-level features
# - Saves the "clean" sentences without <seg> tags in a separate file
# ==========================================================

import spacy
import pandas as pd
import pickle

# Load the spaCy model for Italian
nlp = spacy.load("it_core_news_sm")

# POS tags of interest
pos_tags = ["NOUN", "VERB", "ADJ", "ADV", "CCONJ", "DET", "NUM", "PUNCT", "PRON", "ADP"]

file_input = "data/output_preprocessing/anfass/corpus_in_frasi_anfass.txt"
frasi_pulite_path = "data/output_preprocessing/anfass/frasi.txt"
df_pkl_path = "data/output_preprocessing/anfass/anfass.pkl"
df_csv_path = "data/output_preprocessing/anfass/anfass.csv"

# Function to load numbered sentences from a text file
# Each line of the input file should contain a sentence in the format: index<TAB>sentence
def carica_frasi_numerate(file_path):
    # Opens the file in read mode, using UTF-8 and replacing any faulty characters
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        righe = f.readlines()

    frasi = []

    # Iterate all lines in the file one by one
    for i, riga in enumerate(righe):
        riga = riga.strip()  # removes leading/trailing whitespace

        # Skip lines that do not contain a tab (malformed lines)
        if "\t" not in riga:
            print(f"[LINE {i+1}] Ignored line (no tab): {riga!r}")
            continue

        try:
            # Split the line into two parts: numeric index and sentence
            idx, frase = riga.split("\t", 1)

            # Append a tuple (integer index, sentence) to the list
            frasi.append((int(idx), frase))
        except Exception as e:
            # Print error if something goes wrong during parsing
            print(f"[LINE {i+1}] Parsing error: {e}")

    return frasi  # Returns the list of tuples (index, sentence)

if __name__ == "__main__":
    tutti_dati = []
    frasi_segmentate = []

    # Load numbered sentences from the file
    frasi_numerate = carica_frasi_numerate(file_input)

    # For loop over sentences with <seg> tags
    for frase_idx, frase_con_seg in frasi_numerate:
        # Split the sentence into segments, remove unnecessary spaces
        segmenti = [s.strip() for s in frase_con_seg.split("<seg>")]

        # Rebuild the sentence without <seg>
        frase_pulita = " ".join(segmenti)

        # Apply spaCy to the complete sentence
        doc = nlp(frase_pulita)

        # Save the clean sentence with its index
        frasi_segmentate.append((frase_idx, frase_pulita))

        # Tokenize each segment separately
        segmenti_tokenizzati = [nlp(seg) for seg in segmenti]

        # Calculate lengths (in tokens) of each segment
        lunghezze = [len(seg) for seg in segmenti_tokenizzati]

        # Calculate boundaries between segments
        confini = set()
        offset = 0
        for lung in lunghezze[:-1]:  # exclude the last segment
            offset += lung
            confini.add(offset - 1)  # last token of the segment

        # Iterate over tokens in the complete sentence
        for i_token, token in enumerate(doc):
            if token.text.strip() == "":
                continue  # Skip empty/space tokens

            token_text = token.text
            pos = token.pos_

            # Normalize some POS categories
            if pos == "AUX":
                pos = "VERB"
            elif pos == "SCONJ":
                pos = "CCONJ"
            elif pos not in pos_tags:
                pos = "OTHER"

            # Indicate if the token is at the end of a segment
            segmenta = 1 if i_token in confini else 0

            # Save relevant information into a list of dictionaries
            tutti_dati.append({
                "token": token_text,
                "segmenta": segmenta,
                "frase_idx": frase_idx,
                "frase_len_token": len(doc),
                "frase_len_char": len(frase_pulita),
                "token_len_char": len(token_text),
                "distanza_da_prima_parola": i_token,
                "pos": pos
            })

    # Create a DataFrame with all token-level data
    df = pd.DataFrame(tutti_dati)

    # Create binary columns for each POS tag
    for pos_tag in pos_tags:
        df[pos_tag] = df['pos'].apply(lambda x: 1 if x == pos_tag else 0)

    # Reorder the column order in the DataFrame
    cols_pos = pos_tags
    other_cols = [col for col in df.columns if col not in cols_pos + ["frase_idx", "token", "pos"]]
    df = df[["frase_idx", "token"] + cols_pos + other_cols]

    # Save the clean sentences (without <seg>) to a file
    with open(frasi_pulite_path, "w", encoding="utf-8") as f:
        for idx, frase in frasi_segmentate:
            f.write(f"{idx}\t{frase}\n")

    # Save the dataframe to a pickle file (automatically creates the file if it doesn't exist)
    with open(df_pkl_path, 'wb') as file:
        pickle.dump(df, file)  # save (dump) the dataframe df into the opened file

    # Save the DataFrame as CSV
    df.to_csv(df_csv_path, index=False)  # 'index=False' avoids saving the index as a column in the CSV


import spacy
import pandas as pd

# Load the spaCy model for Italian
nlp = spacy.load("it_core_news_sm")

# List to collect token-level data
tutti_dati = []

# List to save the clean sentences (without <seg> tags)
frasi_segmentate = []

# POS tags of interest
pos_tags = ["NOUN", "VERB", "ADJ", "ADV", "CCONJ", "DET", "NUM", "PUNCT", "PRON", "ADP"]

# Function to load numbered sentences from a text file
# Each line in the input file should contain a sentence in the format: index<TAB>sentence

def carica_frasi_numerate(file_path):
    # Opens the file in read mode, using UTF-8 and replacing any invalid characters
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        righe = f.readlines()

    frasi = []

    # Iterates over each line in the file
    for i, riga in enumerate(righe):
        riga = riga.strip()  # removes leading/trailing whitespace

        # Skips lines that don't contain a tab (not properly formatted)
        if "\t" not in riga:
            print(f"[LINE {i+1}] Line ignored (no tab): {riga!r}")
            continue

        try:
            # Splits the line into two parts: numeric index and sentence
            idx, frase = riga.split("\t", 1)

            # Adds a tuple (int index, sentence) to the list
            frasi.append((int(idx), frase))
        except Exception as e:
            # Prints an error if something goes wrong during parsing
            print(f"[LINE {i+1}] Parsing error: {e}")

    return frasi  # Returns the list of (index, sentence) tuples

# Specify the input file path
file_input = "/content/corpus_in_frasi_anfass_tagliato_perdecision.txt"

# Load numbered sentences from the file
frasi_numerate = carica_frasi_numerate(file_input)

# For loop over sentences with <seg> tags
for frase_idx, frase_con_seg in frasi_numerate:
    # Split the sentence into segments, removing unnecessary spaces
    segmenti = [s.strip() for s in frase_con_seg.split("<seg>")]

    # Reconstruct the sentence without <seg> tags
    frase_pulita = " ".join(segmenti)

    # Apply spaCy to the complete sentence
    doc = nlp(frase_pulita)

    # Save the clean sentence with its index
    frasi_segmentate.append((frase_idx, frase_pulita))

    # Tokenize each segment separately
    segmenti_tokenizzati = [nlp(seg) for seg in segmenti]

    # Compute the length (in tokens) of each segment
    lunghezze = [len(seg) for seg in segmenti_tokenizzati]

    # Compute the boundaries between segments
    confini = set()
    offset = 0
    for lung in lunghezze[:-1]:  # exclude the last segment
        offset += lung
        confini.add(offset - 1)  # last token of the segment

    # Iterate over tokens in the complete sentence
    for i_token, token in enumerate(doc):
        if token.text.strip() == "":
            continue  # Skip empty/whitespace tokens

        token_text = token.text
        pos = token.pos_

        # Normalize some POS categories
        if pos == "AUX":
            pos = "VERB"
        elif pos == "SCONJ":
            pos = "CCONJ"
        elif pos not in pos_tags:
            pos = "ALTRO"

        # Indicates whether the token is at the end of a segment
        segmenta = 1 if i_token in confini else 0

        # Save relevant information in a list of dictionaries
        tutti_dati.append({
            "token": token_text,
            "segmenta": segmenta,
            "frase_idx": frase_idx,
            "frase_len_token": len(doc),
            "frase_len_char": len(frase_pulita),
            "token_len_char": len(token_text),
            "distanza_da_prima_parola": i_token,
            "pos": pos
        })

# Create a DataFrame with all token-level data
df = pd.DataFrame(tutti_dati)

# Create binary columns for each POS tag
for pos_tag in pos_tags:
    df[pos_tag] = df['pos'].apply(lambda x: 1 if x == pos_tag else 0)

# Reorganize column order in the DataFrame
cols_pos = pos_tags
other_cols = [col for col in df.columns if col not in cols_pos + ["frase_idx", "token", "pos"]]
df = df[["frase_idx", "token"] + cols_pos + other_cols]

# Save the clean sentences (without <seg>) to file
with open("corpus_in_frasi_no_seg_anfass_tagliato", "w", encoding="utf-8") as f:
    for idx, frase in frasi_segmentate:
        f.write(f"{idx}\t{frase}\n")

# Download the file
from google.colab import files
files.download("corpus_in_frasi_no_seg_anfass_tagliato")

# Save the dataframe to a pickle file
import pickle

# Save the dataframe to a pickle file (automatically creates the file if it doesn’t exist)
with open('anfass_tagliato.pkl', 'wb') as file:
    pickle.dump(df, file)  # dump the df dataframe into the opened file

from google.colab import files

# File path to download
file_path = "/content/anfass_tagliato.pkl"

# Download the file
files.download(file_path)

import pandas as pd

# Save the DataFrame as CSV
df.to_csv('anfass_tagliato.csv', index=False)  # 'index=False' avoids saving the index as a column

from google.colab import files

# Path of the file to download
file_path_2 = "/content/anfass_tagliato.csv"

# Download the file
files.download(file_path_2)
