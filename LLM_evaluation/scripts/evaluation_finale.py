
# This script processes a multiline text where some lines start with a sentence index (number + tab),
# followed by text, and other lines are continuations of the previous sentence.
# It uses spaCy's Italian model to tokenize each line's text.
# For each token, it associates the current sentence index and determines a boundary flag:
# - The boundary is set to 1 if the token is the last token of the line,
#   the token does NOT end with strong punctuation (., !, ?, …),
#   and the next line does NOT start a new sentence index.
# Otherwise, the boundary is 0.
# The function returns a list of tuples: (sentence_index, token_text, boundary_flag).
# If debug=True, it prints each token with its sentence index and boundary flag for inspection.


import re
import spacy

!python -m spacy download it_core_news_sm

nlp = spacy.load("it_core_news_sm")

def extract_tokens_with_sentence_index(text, debug=False):
    righe = text.strip().split("\n")
    results = []

    current_index = None  # Current sentence index

    for idx, riga in enumerate(righe):
        riga = riga.strip()
        if not riga:
            continue

        # Check if the line starts with a sentence index (number + tab)
        match = re.match(r"^(\d+)\t(.*)", riga)
        if match:
            current_index = int(match.group(1))
            testo = match.group(2)
        else:
            # Continuation line, use the same current sentence index
            testo = riga

        doc = nlp(testo)
        num_tokens = len(doc)

        for i, token in enumerate(doc):
            boundary = 0
            is_last_token = (i == num_tokens - 1)

# For boundary we use the same logic as before:
            # if last token of the line, and it does NOT end with strong punctuation,
            # and the next line is NOT a new sentence index -> boundary=1

            if is_last_token:
                next_is_new_sentence = False
                if idx + 1 < len(righe):
                    next_line = righe[idx + 1].strip()
                    next_is_new_sentence = bool(re.match(r"^\d+\t", next_line))

                ends_with_punct = token.text in {".", "!", "?", "…"}
                if not ends_with_punct and not next_is_new_sentence:
                    boundary = 1

            results.append((current_index, token.text, boundary))

            if debug:
                print(f"frase {current_index}] '{token.text}' -> {boundary}")

    return results

#file output of the II prompt

with open("/content/frasi_semplificate_gemma_2_9b_it_anfass2_tagliato.txt", "r", encoding="utf-8") as f:
    testo = f.read()

risultati_llm = extract_tokens_with_sentence_index(testo, debug=True)

#reference file

with open("/content/frasi_segmentate_per_evaluation_tagliato.txt", "r", encoding="utf-8") as f:
    testo = f.read()

risultati_reference= extract_tokens_with_sentence_index(testo, debug=True)

#file output of the first prompt

with open("/content/frasi_semplificate_gemma_2_9b_it_anfass1_tagliato.txt", "r", encoding="utf-8") as f:
    testo = f.read()

risultati_llm_3 = extract_tokens_with_sentence_index(testo, debug=True)

from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support

def evaluate_by_sentence_2(predicted, reference):
    """
    predicted and reference are lists of tuples (sentence_index, token, boundary)
    It is assumed that predicted and reference contain the same tokens and sentence indices in the same order
    """

    if len(predicted) != len(reference):
        print(f" Different lenghts: predicted = {len(predicted)}, reference = {len(reference)}")

        # Confronto (idx, token) ignorando boundary
        pred_pairs = [(idx, tok) for idx, tok, _ in predicted]
        ref_pairs  = [(idx, tok) for idx, tok, _ in reference]

        pred_set = set(pred_pairs)
        ref_set  = set(ref_pairs)

        missing_in_pred = ref_set - pred_set
        extra_in_pred   = pred_set - ref_set

        print(f"\nTokens (idx, token) in reference but NOT in predicted: {len(missing_in_pred)}")
        for item in sorted(missing_in_pred)[:10]:
            print("  Manca:", item)

        print(f"Tokens (idx, token) in predicted but NOT in reference:: {len(extra_in_pred)}")
        for item in sorted(extra_in_pred)[:10]:
            print("  In più:", item)

        #Find the first point of divergence
        min_len = min(len(pred_pairs), len(ref_pairs))
        for i in range(min_len):
            if pred_pairs[i] != ref_pairs[i]:
                print(f"\nDivergence at position {i}:")
                print(f"   Pred: {pred_pairs[i]}")
                print(f"   Ref:  {ref_pairs[i]}")
                break

        raise ValueError("Predicted and reference lists have different lengths.")

    # Check sentence indices and tokens
    for i in range(len(predicted)):
        p_idx, p_tok, _ = predicted[i]
        r_idx, r_tok, _ = reference[i]

        if p_idx != r_idx:
            raise ValueError(f"Sentence index mismatch at position {i}: pred={p_idx}, ref={r_idx}")
        if p_tok != r_tok:
            raise ValueError(f"Token mismatch at position {i}: pred='{p_tok}', ref='{r_tok}'")

    # Group boundaries by sentence
    pred_by_sent = defaultdict(list)
    ref_by_sent = defaultdict(list)

    for (idx_pred, _, b_pred), (_, _, b_ref) in zip(predicted, reference):
        pred_by_sent[idx_pred].append(b_pred)
        ref_by_sent[idx_pred].append(b_ref)

    ## For each sentence, compute precision, recall, f1
    precision_list = []
    recall_list = []
    f1_list = []

    for sent_id in sorted(ref_by_sent.keys()):
        y_true = ref_by_sent[sent_id]
        y_pred = pred_by_sent.get(sent_id, [0]*len(y_true))

        if all(v == 0 for v in y_true) and all(v == 0 for v in y_pred):
            #print(f"Frase {sent_id}: Correct")

    # debug: double-check if they are really all zeros
            if any(v != 0 for v in y_true):
                print("y_true contains non-zero values!")
            if any(v != 0 for v in y_pred):
                print("y_pred contains non-zero values!")

            #print("→ y_true:", y_true)
            #print("→ y_pred:", y_pred)

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)

        #print(f"Precision: {p}")
        #print(f"Recall:    {r}")
        #print(f"F1-score:  {f1}")


    # Aggregate average
    precision_avg = sum(precision_list) / len(precision_list)
    recall_avg = sum(recall_list) / len(recall_list)
    f1_avg = sum(f1_list) / len(f1_list)


    print(f"\nAverage over all sentences (class 0): Precision={precision_avg[0]:.3f}, Recall={recall_avg[0]:.3f}, F1={f1_avg[0]:.3f}")
    print(f"Average over all sentences (class 1): Precision={precision_avg[1]:.3f}, Recall={recall_avg[1]:.3f}, F1={f1_avg[1]:.3f}")

    return precision_list, recall_list, f1_list

evaluate_by_sentence_2(risultati_llm, risultati_reference)

evaluate_by_sentence_2(risultati_llm_3, risultati_reference)
