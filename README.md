# Usage with Google Drive and Colab

This script is designed to run on Google Colab with Google Drive mounted.

1. Mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

i notebooks sono pronti per essere runnati su google colab vs gli scripts in teoria potranno essere eseguiti ovunque

## Manual Files

The following files were created manually and are not generated by any script in this repository:

-output_corpus_no_ricette- from the file output_corpus_in_frasi that contains le il corpus dueparole diviso in frasi con i seg- da cui poi creeremo il dataframe per allenare il decision tree rimuoviamo manualmente le ricette, perché contenevano troppi elenchi puntati e cose del genere
non so se dire manulamente o con tecniche comunque senza codice
questo file viene usato per creare il dataframe per il decision tree

-corpus_in_frasi_anfass_tagliato_perdecision.txt- questo file è stato creato manualmente a partire dal file corpus in frasi anfass- a questo corpus che conteneva i seg sono state tolte le frasi che il modello LLM ha modificato e che quindi non abbiamo potuto considerare nell'evaluation del modello. Per rendere i nostri modelli comparabili abbiamo eliminato le frasi anche nell'evaluation del decision tree. Questo file quindi serve per creare un dataframe su cui sarà allenato un nuovo decision tree che non terrà in considerazione le frasi che LLM non considera nella sua evaluation
si trova nel file dataframe_comparable_results

-i file usati in evaluation_finale pure- abbiamo eliminato usando metodi mezzo automatici mezzo manuali e revisioni manuali tutte le frasi per cui il modello modificava alcuni token. Usiamo questi 3 per fare evaluation del nostro modello frasi_semplificate_gemma_2_9b_it_anfass2_tagliato.txt, /content/frasi_semplificate_gemma_2_9b_it_anfass1_tagliato.txt, frasi_segmentate_per_evaluation_tagliato.txt
