# Tracciamento delle Tecnologie Emergenti con Neo4j e BERT utilizzando i dati di arXiv

Questo progetto esplora l'uso di Knowledge Graph dinamici per tracciare l'evoluzione delle tecnologie emergenti, utilizzando il database a grafo Neo4j e l'API di arXiv per accedere ai dati dei paper scientifici. L'obiettivo è identificare trend tecnologici e collaborazioni tra ricercatori. Il progetto include l'integrazione di modelli di deep learning come BERT per migliorare la similarità semantica tra i documenti, superando le limitazioni dei metodi tradizionali basati su WordNet e BabelNet.

## Caratteristiche principali

- **Neo4j Database:** Creazione e gestione di un Knowledge Graph dinamico basato su dati scientifici.
- **API di arXiv:** Raccolta e preparazione dei dati dai paper pubblicati su arXiv.
- **WordNet Integration:** Miglioramento della correlazione semantica tra i documenti utilizzando WordNet.
- **BERT Integration:** Applicazione di modelli di deep learning per un'analisi semantica avanzata e precisa.
- **Analisi Avanzate:** Esempi di query e analisi pratiche per trovare paper simili, collaborazioni tra autori e identificare comunità di ricerca.

## Requisiti

- Neo4j 4.x o superiore
- Python 3.x
- Librerie Python: `requests`, `xml.etree.ElementTree`, `nltk`, `transformers`, `torch`, `pandas`, `scikit-learn`, `matplotlib`
- Dataset di arXiv accessibile tramite l'API di arXiv

## Installazione

1. Clona il repository:
   ```bash
   git clone https://github.com/tuo_username/tuo_repository.git
   cd tuo_repository
   ```

2. Installa le dipendenze Python:
   ```bash
   pip install -r requirements.txt
   ```

3. Configura Neo4j e importa i dati da arXiv utilizzando gli script forniti.

## Esecuzione

- Esegui gli script Python per scaricare e preprocessare i dati da arXiv.
- Carica i dati in Neo4j e utilizza le query Cypher per esplorare il Knowledge Graph.
- Esegui le analisi semantiche utilizzando WordNet e BERT.

## Contributi

Le richieste di pull sono benvenute. Per modifiche importanti, apri prima un problema per discutere ciò che desideri modificare.
