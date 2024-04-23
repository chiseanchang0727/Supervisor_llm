import json



def read_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except json.JSONDecodeError:
        print("Invalid JSON file.")
    except Exception as e:
        print(f"An error occurred: {e}")