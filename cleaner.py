import csv
import sys

def clean_person_column(input_path, output_path):
    # Set of entities to remove, including organization names, locations, dates, and objects
    entities_to_remove = {
        # Organizations/Agencies
        "USBP", "JOINT HARBOR OPERATION CENTER", "PORTLAND HARBOR UNIT", "RCMP", "CSIS", "CBSA",
        "HOMELAND SECURITY INVESTIGATIONS", "INA", "PBSO", "HSI", "AMO", "CBP MARINE INTERDICTION AGENTS",
        "U.S. BORDER PATROL (USBP) AGENTS", "HOMELAND SECURITY INVESTIGATIONS (HSI)", "MARINE TASK FORCE (MTF)",
        "AIR AND MARINE OPERATIONS (AMO)", "JOINT HARBOR OPERATIONS CENTER (JHOC)", "UNITED STATES COAST GUARD (USCG)",
        "SHERIFF'S DEPUTIES", "CALIFORNIA NATIONAL GUARD TROOPS", "CBP AIRCRAFT", "HSI ST. THOMAS", "CUSTOMS AGENTS",

        # Locations/Routes
        "NUEVO LAREDO", "HIGHWAY 359", "EXIT 101",

        # Dates
        "MAY 26, 2006",

        # Objects/Vehicles
        "1996 MERCURY COUGAR"
    }

    try:
        with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
             open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Read and write header
            header = next(reader)
            writer.writerow(header)

            # Process each row
            for row in reader:
                if len(row) > 4:
                    person_column_index = 4  # 'Person' is the 5th column
                    
                    # Handle potential empty values
                    if row[person_column_index]:
                        person_entities = row[person_column_index].split(',')
                        
                        # Clean entities: strip whitespace and filter
                        cleaned_entities = [
                            entity.strip() for entity in person_entities 
                            if entity.strip() and entity.strip() not in entities_to_remove
                        ]
                        
                        row[person_column_index] = ",".join(cleaned_entities)
                
                writer.writerow(row)
                
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    input_csv = '/home/eli/Downloads/gmu work/dataset1.csv'
    output_csv = '/home/eli/Downloads/gmu work/dataset2.csv'
    clean_person_column(input_csv, output_csv)
    print(f"Successfully cleaned '{input_csv}' and saved to '{output_csv}'")
