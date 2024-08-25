import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import matplotlib.pyplot as plt

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
bert_embeddings = []
for abstract in tqdm(abstracts, desc="Calcolo degli embedding BERT"):
    bert_embeddings.append(get_bert_embedding(abstract).numpy())

# Calcolo della similarità del coseno tra gli embedding BERT
bert_cosine_similarities = cosine_similarity(bert_embeddings)

# Creazione di un DataFrame per mostrare le similarità BERT
bert_similarity_df = pd.DataFrame(bert_cosine_similarities, index=data['id'], columns=data['id'])

# Funzione per creare relazioni di similarità basate su una soglia
def create_similarity_edges(similarity_df, threshold=0.6):
    edges = []
    for i, row in similarity_df.iterrows():
        for j, similarity in row.items():
            if i != j and similarity >= threshold:
                edges.append((i, j, similarity))
    return edges

# Creazione delle relazioni di similarità per BERT
bert_edges = create_similarity_edges(bert_similarity_df, threshold=0.6)

# Creazione di un file CSV per le relazioni di similarità BERT
bert_edges_df = pd.DataFrame(bert_edges, columns=['Paper1', 'Paper2', 'Similarity'])
bert_edges_df.to_csv('bert_similarity_edges.csv', index=False)

print("Relazioni di similarità BERT create e salvate in 'bert_similarity_edges.csv'.")

# Calcolo della similarità TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(abstracts)
tfidf_cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Creazione di un DataFrame per mostrare le similarità TF-IDF
tfidf_similarity_df = pd.DataFrame(tfidf_cosine_similarities, index=data['id'], columns=data['id'])

# Creazione delle relazioni di similarità per TF-IDF
tfidf_edges = create_similarity_edges(tfidf_similarity_df, threshold=0.6)

# Creazione di un file CSV per le relazioni di similarità TF-IDF
tfidf_edges_df = pd.DataFrame(tfidf_edges, columns=['Paper1', 'Paper2', 'Similarity'])
tfidf_edges_df.to_csv('tfidf_similarity_edges.csv', index=False)

print("Relazioni di similarità TF-IDF create e salvate in 'tfidf_similarity_edges.csv'.")

# Plot della distribuzione delle similarità BERT
plt.figure(figsize=(10, 5))
plt.hist(bert_cosine_similarities.flatten(), bins=50, alpha=0.5, label='BERT Similarity')
plt.hist(tfidf_cosine_similarities.flatten(), bins=50, alpha=0.5, label='TF-IDF Similarity')
plt.title('Distribuzione delle Similarità')
plt.xlabel('Similarità')
plt.ylabel('Frequenza')
plt.legend(loc='upper right')
plt.show()