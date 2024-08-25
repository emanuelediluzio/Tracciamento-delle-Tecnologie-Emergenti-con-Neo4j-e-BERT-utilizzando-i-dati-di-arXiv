import pandas as pd
import matplotlib.pyplot as plt

# Carica il file CSV
file_path = 'similarity_edges.csv'
df = pd.read_csv(file_path, header=None, names=["paper1", "paper2", "similarity"])

# Assicurati che i dati siano correttamente caricati e che non ci siano problemi con i valori
df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce')
df.dropna(subset=['similarity'], inplace=True)

# Visualizza la distribuzione delle similarità
plt.figure(figsize=(10, 6))
plt.hist(df['similarity'], bins=50, edgecolor='k', alpha=0.7)
plt.title('Distribuzione delle Similarità tra i Paper')
plt.xlabel('Similarità')
plt.ylabel('Frequenza')
plt.grid(True)
plt.show()

# Calcola i quartili e l'IQR
Q1 = df['similarity'].quantile(0.25)
Q3 = df['similarity'].quantile(0.75)
IQR = Q3 - Q1

# Definisce i limiti degli outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identifica gli outlier
outliers = df[(df['similarity'] < lower_bound) | (df['similarity'] > upper_bound)]

# Stampa un riepilogo degli outlier
print("Number of outliers:", outliers.shape[0])
print(outliers)