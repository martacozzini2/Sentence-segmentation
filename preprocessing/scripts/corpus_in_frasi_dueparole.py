# ==========================================================
# Script for segmenting a corpus into sentences using spaCy
#
# Features:
# - Loads a text corpus from file
# - Splits the text into blocks using a pattern based on lines separated by ---
# - For each block:
#     - Replaces line breaks (\n) with <seg> to indicate internal segments
#     - Segments the text into sentences using the Italian spaCy model
#     - Removes any <seg> tags at the beginning of sentences
# - Saves all numbered sentences (one per line, with index) to an output file
# - The output file will be manually modified before of being used 
# ==========================================================

import spacy
import re

# Load the Italian spaCy language model
nlp = spacy.load("it_core_news_sm")

# Functions

# Reads the content of a text file
def carica_testo(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        testo = f.read()
    return testo

# Removes <seg> at the beginning of sentences
def rimuovi_seg_inizio_frase(frasi):
    frasi_pulite = []
    for frase in frasi:
        frase_pulita = re.sub(r'^<seg>\s*', '', frase)
        frasi_pulite.append(frase_pulita)
    return frasi_pulite

# Segments the text into sentences using spaCy
def segmenta_testo_spacy(testo):
    paragrafi = testo.split("\n\n")  # Splits into paragraphs
    frasi = []

    for p in paragrafi:
        p = p.strip()
        if not p:
            continue
        # Replaces \n with <seg> to mark internal breaks
        p = p.replace("\n", " <seg> ")
        doc = nlp(p)
        # Extracts and cleans the segmented sentences
        frasi.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])

    # Removes <seg> if it appears at the beginning of the sentence
    frasi = rimuovi_seg_inizio_frase(frasi)
    return frasi

# Splits the text into blocks and segments each block into sentences
def segmenta_blocchi(testo):
    blocchi = re.split(r"-{3,} \d+ .+?\.txt -{3,}", testo)  # Splits based on block pattern
    blocchi = [b.strip() for b in blocchi if b.strip()]

    frasi_totali = []
    for blocco in blocchi:
        frasi_blocco = segmenta_testo_spacy(blocco)
        frasi_totali.extend(frasi_blocco)
    return frasi_totali

# Saves the numbered sentences to a file
def salva_frasi_con_indice(frasi, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, frase in enumerate(frasi, 1):
            f.write(f"{i}\t{frase}\n")


# Main block
if __name__ == "__main__":
    input_file = "data/corpora/corpus_dueparole.txt"
    output_file = "data/output_preprocessing/dueparole/output_corpus_in_frasi.txt"

    testo = carica_testo(input_file)
    frasi = segmenta_blocchi(testo)
    salva_frasi_con_indice(frasi, output_file)

# From here, we manually remove the recipes
