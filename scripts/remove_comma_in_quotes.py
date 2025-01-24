import re

def remove_commas_in_quotes(text):
    # Use regex to find all quoted strings and replace commas within them
    return re.sub(r'\"(.*?)\"', lambda m: m.group(0).replace(',', ''), text)

def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()
    
    updated_content = remove_commas_in_quotes(content)
    
    with open(output_file, 'w') as file:
        file.write(updated_content)

if __name__ == "__main__":
    input_file = 'data/generated_training_data/Events.csv'
    output_file = 'data/generated_training_data/Events_no_comma.csv'
    process_file(input_file, output_file)
