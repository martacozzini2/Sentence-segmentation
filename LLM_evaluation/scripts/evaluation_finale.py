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
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

nlp = spacy.load("it_core_news_sm")

def extract_tokens_with_sentence_index(text, debug=False):
    righe = text.strip().split("\n")
    results = []

    current_index = None

    for idx, riga in enumerate(righe):
        riga = riga.strip()
        if not riga:
            continue

        match = re.match(r"^(\d+)\t(.*)", riga)
        if match:
            current_index = int(match.group(1))
            testo = match.group(2)
        else:
            testo = riga

        doc = nlp(testo)
        num_tokens = len(doc)

        for i, token in enumerate(doc):
            boundary = 0
            is_last_token = (i == num_tokens - 1)

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


def evaluate_by_sentence_2(predicted, reference):
    if len(predicted) != len(reference):
        print(f"Different lengths: predicted = {len(predicted)}, reference = {len(reference)}")

        pred_pairs = [(idx, tok) for idx, tok, _ in predicted]
        ref_pairs = [(idx, tok) for idx, tok, _ in reference]

        missing_in_pred = set(ref_pairs) - set(pred_pairs)
        extra_in_pred = set(pred_pairs) - set(ref_pairs)

        print(f"\nTokens (idx, token) in reference but NOT in predicted: {len(missing_in_pred)}")
        for item in sorted(missing_in_pred)[:10]:
            print("  Manca:", item)

        print(f"Tokens (idx, token) in predicted but NOT in reference: {len(extra_in_pred)}")
        for item in sorted(extra_in_pred)[:10]:
            print("  In più:", item)

        for i in range(min(len(pred_pairs), len(ref_pairs))):
            if pred_pairs[i] != ref_pairs[i]:
                print(f"\nDivergence at position {i}:")
                print(f"   Pred: {pred_pairs[i]}")
                print(f"   Ref:  {ref_pairs[i]}")
                break

        raise ValueError("Predicted and reference lists have different lengths.")

    for i in range(len(predicted)):
        p_idx, p_tok, _ = predicted[i]
        r_idx, r_tok, _ = reference[i]

        if p_idx != r_idx:
            raise ValueError(f"Sentence index mismatch at position {i}: pred={p_idx}, ref={r_idx}")
        if p_tok != r_tok:
            raise ValueError(f"Token mismatch at position {i}: pred='{p_tok}', ref='{r_tok}'")

    pred_by_sent = defaultdict(list)
    ref_by_sent = defaultdict(list)

    for (idx_pred, _, b_pred), (_, _, b_ref) in zip(predicted, reference):
        pred_by_sent[idx_pred].append(b_pred)
        ref_by_sent[idx_pred].append(b_ref)

    precision_list = []
    recall_list = []
    f1_list = []

    for sent_id in sorted(ref_by_sent.keys()):
        y_true = ref_by_sent[sent_id]
        y_pred = pred_by_sent.get(sent_id, [0] * len(y_true))

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)

    precision_avg = sum(precision_list) / len(precision_list)
    recall_avg = sum(recall_list) / len(recall_list)
    f1_avg = sum(f1_list) / len(f1_list)

    print(f"\nAverage over all sentences (class 0): Precision={precision_avg[0]:.3f}, Recall={recall_avg[0]:.3f}, F1={f1_avg[0]:.3f}")
    print(f"Average over all sentences (class 1): Precision={precision_avg[1]:.3f}, Recall={recall_avg[1]:.3f}, F1={f1_avg[1]:.3f}")

    return precision_list, recall_list, f1_list


if __name__ == "__main__":

    #first prompt output
    file_llm_1 = "frasi_semplificate_gemma_2_9b_it_anfass1_tagliato.txt"
    with open(file_llm_1, "r", encoding="utf-8") as f:
        testo = f.read()
    risultati_llm_3 = extract_tokens_with_sentence_index(testo, debug=False)

    #second prompt output
    file_llm_2 = "frasi_semplificate_gemma_2_9b_it_anfass2_tagliato.txt"
    with open(file_llm_2, "r", encoding="utf-8") as f:
        testo = f.read()
    risultati_llm = extract_tokens_with_sentence_index(testo, debug=False)

    #reference file
    file_reference = "frasi_segmentate_per_evaluation_tagliato.txt"
    with open(file_reference, "r", encoding="utf-8") as f:
        testo = f.read()
    risultati_reference = extract_tokens_with_sentence_index(testo, debug=False)

    #Evaluation
    print("Second prompt evaluation")
    evaluate_by_sentence_2(risultati_llm, risultati_reference)

    print("First prompt evaluation")
    evaluate_by_sentence_2(risultati_llm_3, risultati_reference)
