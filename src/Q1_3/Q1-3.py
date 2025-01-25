import csv

def calculate_event_influence(input_csv, output_csv):
    country_totals = {}
    rows = []
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            noc = row['NOC']
            medalcount = int(row['medalcount'])
            country_totals[noc] = country_totals.get(noc, 0) + medalcount
            rows.append(row)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        fieldnames = ['NOC', 'Event', 'Influence']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            noc = row['NOC']
            event = row['Event']
            influence = int(row['medalcount']) / country_totals[noc]
            writer.writerow({'NOC': noc, 'Event': event, 'Influence': influence})

calculate_event_influence('D:/2025-MCM-self/Q1-3_data.csv', 'event_influence.csv')