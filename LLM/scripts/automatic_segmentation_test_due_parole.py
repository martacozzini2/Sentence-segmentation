# Authentication with Hugging Face

from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()  # Loads environment variables from .env
hf_token = os.getenv("HF_TOKEN")  # Gets the token from the environment variable
login(token=hf_token)  # Logs in to Hugging Face using the token

import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set the seed for reproducibility
seed = 42

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load the tokenizer and model from Hugging Face Hub

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Define paths for input and output files
input_path = "data/output_preprocessing/dueparole/frasi_test_set_due_parole.txt"
output_path = "data/output_LLM/original_outputs/frasi_segmentate_test_due_parole_1p.txt"

simplified_results = []

# Load sentences from input file
frasi = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            print("Riga non valida:", line)
            continue
        id_, frase = parts[0], parts[1]
        frasi.append({"id": id_, "frase": frase})

# Load already processed sentence IDs to avoid duplicates
processed_ids = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f_out:
        for line in f_out:
            parts = line.strip().split("\t", maxsplit=1)
            if len(parts) >= 1:
                processed_ids.add(parts[0])

with open(output_path, "a", encoding="utf-8") as f_out:
    for entry in frasi:
        if entry["id"] in processed_ids:
            print(f"Già processata: {entry['id']}, salto.")
            continue

        prompt = f"""Dividi la seguente frase in segmenti separati, inserendo un ritorno a capo dove le persone farebbero una pausa leggendo la frase ad alta voce.
Ogni segmento di testo dovrebbe contenere tra le 5 e le 15 parole.
Il contenuto della frase originale non deve essere alterato in nessun modo; pertanto non deve essere aggiunta nuova informazione di alcun tipo.
Scrivi ogni segmento su una nuova riga, senza numerazione o simboli all'inizio.
Non generare altro testo ad eccezione del testo originale segmentato.

Testo: {entry['frase']}

Risultato:"""

        try:
            input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

            outputs = model.generate(
                **input_ids,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Risultato:" in decoded:
                simplified = decoded.split("Risultato:")[1].strip()
            else:
                simplified = decoded.strip()

            result_line = f"{entry['id']}\t{simplified}"
            f_out.write(result_line + "\n")
            f_out.flush()
            print(f"Semplificata frase {entry['id']}")

        except Exception as e:
            print(f"Errore nella frase {entry['id']}: {e}")
            break

input_path = "data/output_preprocessing/dueparole/frasi_test_set_due_parole.txt"
output_path = "data/output_LLM/original_outputs/frasi_segmentate_test_due_parole_2p.txt"

simplified_results = []

frasi = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            print("Riga non valida:", line)
            continue
        id_, frase = parts[0], parts[1]
        frasi.append({"id": id_, "frase": frase})

processed_ids = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f_out:
        for line in f_out:
            parts = line.strip().split("\t", maxsplit=1)
            if len(parts) >= 1:
                processed_ids.add(parts[0])

with open(output_path, "a", encoding="utf-8") as f_out:
    for entry in frasi:
        if entry["id"] in processed_ids:
            print(f"Già processata: {entry['id']}, salto.")
            continue

        prompt = f"""Dividi la seguente frase in segmenti separati, che rispettino i confini grammaticali naturali.
Ogni segmento di testo dovrebbe contenere tra le 5 e le 15 parole.
Il contenuto della frase originale deve essere mantenuto rigorosamente; pertanto non deve essere aggiunta nuova informazione di alcun tipo.
Scrivi ogni segmento su una nuova riga, senza numerazione o simboli all'inizio.
Non generare altro testo ad eccezione del testo originale segmentato.

Testo: {entry['frase']}

Risultato:"""

        try:
            input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

            outputs = model.generate(
                **input_ids,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Risultato:" in decoded:
                simplified = decoded.split("Risultato:")[1].strip()
            else:
                simplified = decoded.strip()

            result_line = f"{entry['id']}\t{simplified}"
            f_out.write(result_line + "\n")
            f_out.flush()
            print(f"Semplificata frase {entry['id']}")

        except Exception as e:
            print(f"Errore nella frase {entry['id']}: {e}")
            break
