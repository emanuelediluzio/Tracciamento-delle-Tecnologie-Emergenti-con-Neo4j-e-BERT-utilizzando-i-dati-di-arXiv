import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Caricamento del file CSV
file_path = 'cleaned_arxiv_data.csv'
data = pd.read_csv(file_path)

# Estrazione degli abstract
abstracts = data['abstract'].fillna('').tolist()

# Caricamento del modello e del tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Funzione per ottenere l'embedding BERT di un testo
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Calcolo degli embedding per tutti gli abstract
embeddings = []
for abstract in tqdm(abstracts, desc="Calcolo degli embedding"):
    embeddings.append(get_bert_embedding(abstract).numpy())

# Calcolo della similarità del coseno tra gli embedding
cosine_similarities = cosine_similarity(embeddings)

# Creazione di un DataFrame per mostrare le similarità
similarity_df = pd.DataFrame(cosine_similarities, index=data['id'], columns=data['id'])

# Funzione per creare relazioni di similarità basate su una soglia
def create_similarity_edges(similarity_df, threshold=0.6):
    edges = []
    for i, row in similarity_df.iterrows():
        for j, similarity in row.items():
            if i != j and similarity >= threshold:
                edges.append((i, j, similarity))
    return edges

# Creazione delle relazioni di similarità
edges = create_similarity_edges(similarity_df, threshold=0.6)

# Creazione di un file CSV per le relazioni di similarità
edges_df = pd.DataFrame(edges, columns=['Paper1', 'Paper2', 'Similarity'])
edges_df.to_csv('similarity_edges.csv', index=False)

print("Relazioni di similarità create e salvate in 'similarity_edges.csv'.")