import csv
import openpyxl

def calculate_event_influence(input_xlsx, output_csv):
    country_totals = {}
    rows = []
    
    wb = openpyxl.load_workbook(input_xlsx)
    sheet = wb.active
    
    header = [cell.value for cell in sheet[1]]
    for row in sheet.iter_rows(min_row=2, values_only=True):
        row_dict = dict(zip(header, row))
        noc = row_dict['NOC']
        medalcount = int(row_dict['medalcount'])
        country_totals[noc] = country_totals.get(noc, 0) + medalcount
        rows.append(row_dict)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        fieldnames = ['NOC', 'Event', 'Influence']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            noc = row['NOC']
            event = row['Event']
            total_medals = country_totals[noc]
            influence = int(row['medalcount']) / total_medals if total_medals != 0 else 0
            writer.writerow({'NOC': noc, 'Event': event, 'Influence': influence})

calculate_event_influence(r'src\\Q1_3\\Q1-3_data.xlsx', 'event_influence.csv')