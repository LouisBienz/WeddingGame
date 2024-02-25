import csv

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)  # Read the headers
        for row in csv_reader:
            data.append(row)
    return headers, data

def import_csv():
    file_path = "/Users/lou/workarea/WeddingGame/antworten_vektoren.csv" 
    headers, data = read_csv(file_path)
    return headers, data
