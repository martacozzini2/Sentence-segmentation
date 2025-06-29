
# ==========================================================
# Script for processing segmented sentences with spaCy
# - Loads sentences with segments marked by <seg>
# - Analyzes sentences with spaCy
# - Builds a DataFrame with token-level features
# - Saves "clean" sentences without <seg> tags to a separate file
# We will use the output DataFrame to train and evaluate a Decision Tree
# ==========================================================

import spacy
import pandas as pd

#Load the spaCy model for Italian
nlp = spacy.load("it_core_news_sm")

# List to collect token-level data
tutti_dati = []

# List to save clean sentences (without <seg> tags)
frasi_segmentate = []

# POS tags of interest
pos_tags = ["NOUN", "VERB", "ADJ", "ADV", "CCONJ", "DET", "NUM", "PUNCT", "PRON", "ADP"]

# Function to load numbered sentences from a text file
def carica_frasi_numerate(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        righe = f.readlines()

    frasi = []

  # Iterates all lines in the file one by one
    for i, riga in enumerate(righe):
        riga = riga.strip()

# Skips lines that do not contain a tab (malformed lines)
        if "\t" not in riga:
            print(f"[RIGA {i+1}] Riga ignorata (niente tab): {riga!r}")
            continue

        try:
            # Splits the line into two parts: numeric index and sentence
            idx, frase = riga.split("\t", 1)

# Adds a tuple (integer index, sentence) to the list
            frasi.append((int(idx), frase))
        except Exception as e:
            # Prints error if something goes wrong during parsing
            print(f"[RIGA {i+1}] Errore nel parsing: {e}")

    return frasi

# Specify the input file path
file_input = "/content/output_corpus_in_no_ricette.txt"

# Loads numbered sentences from the file
frasi_numerate = carica_frasi_numerate(file_input)

# For loop over sentences with <seg> tags
for frase_idx, frase_con_seg in frasi_numerate:
    # Splits the sentence into segments, removes unnecessary spaces
    segmenti = [s.strip() for s in frase_con_seg.split("<seg>")]

    # Rebuilds the sentence without <seg>
    frase_pulita = " ".join(segmenti)

    # Applies spaCy to the complete sentence
    doc = nlp(frase_pulita)

    # Saves the clean sentence with the index
    frasi_segmentate.append((frase_idx, frase_pulita))

    # Tokenizes each segment separately
    segmenti_tokenizzati = [nlp(seg) for seg in segmenti]

    # Calculates the length (in tokens) of each segment
    lunghezze = [len(seg) for seg in segmenti_tokenizzati]

    # Calculates boundaries between segments
    confini = set()
    offset = 0
    for lung in lunghezze[:-1]:
        offset += lung
        confini.add(offset - 1)

 # Iterates over tokens in the complete sentence
    for i_token, token in enumerate(doc):
        if token.text.strip() == "":
            continue

        token_text = token.text
        pos = token.pos_

        # Normalizes some POS categories
        if pos == "AUX":
            pos = "VERB"
        elif pos == "SCONJ":
            pos = "CCONJ"
        elif pos not in pos_tags:
            pos = "ALTRO"

        # Indicates if the token is at the end of a segment
        segmenta = 1 if i_token in confini else 0

        # Saves relevant information into a list of dictionaries
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

# Creates a DataFrame with all token-level data
df = pd.DataFrame(tutti_dati)

# Creates binary columns for each POS tag
for pos_tag in pos_tags:
    df[pos_tag] = df['pos'].apply(lambda x: 1 if x == pos_tag else 0)

# Reorganizes the column order in the DataFrame
cols_pos = pos_tags
other_cols = [col for col in df.columns if col not in cols_pos + ["frase_idx", "token", "pos"]]
df = df[["frase_idx", "token"] + cols_pos + other_cols]

# # Saves clean sentences (without <seg>) to a file
with open("corpus_in_frasi_no_seg", "w", encoding="utf-8") as f:
    for idx, frase in frasi_segmentate:
        f.write(f"{idx}\t{frase}\n")

# Saves the dataframe to a pickle file (automatically creates the file if it doesn't exist)
with open('parole.pkl', 'wb') as file:
    pickle.dump(df, file)
from google.colab import files

# Saves the DataFrame as CSV
df.to_csv('parole.csv', index=False)
