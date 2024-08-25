import csv
import re

def clean_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            cleaned_row = []
            for field in row:
                # Rimuove le virgolette doppie e chiude correttamente le virgolette aperte
                field = re.sub(r'""', '"', field)
                if field.count('"') % 2 != 0:
                    field += '"'
                cleaned_row.append(field)
            writer.writerow(cleaned_row)

input_file = 'arxiv_data.csv'
output_file = 'cleaned_arxiv_data.csv'

clean_csv(input_file, output_file)