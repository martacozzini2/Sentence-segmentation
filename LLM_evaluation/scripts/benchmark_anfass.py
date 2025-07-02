# This script takes a file where segments are marked with "<seg>" indicating where line breaks should be.
# It reconstructs the sentences using these original line breaks from the corpus.
# The output file serves as a benchmark to compare the model's segmentation with the original corpus segmentation.


input_file = "data/output_preprocessing/anfass/corpus_in_frasi_anfass.txt"

output_file = "data/output_LLM/evaluation/frasi_segmentate_per_evaluation.txt"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        if not line.strip():
            continue

        parts = line.strip().split("\t", 1)
        if len(parts) != 2:
            continue

        numero, testo = parts
        segmenti = [seg.strip() for seg in testo.split("<seg>") if seg.strip()]

        if segmenti:
            f_out.write(f"{numero}\t{segmenti[0]}\n")
            for seg in segmenti[1:]:
                f_out.write(f"{seg}\n")
