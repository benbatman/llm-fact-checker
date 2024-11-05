import json


def remove_duplicates_by_url(file_path):
    # Load existing data from the JSON file
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        print("File not found or invalid JSON format.")
        return

    # Dictionary to track unique entries by 'url'
    unique_data = {}

    # Iterate through the data and filter out duplicates by 'url'
    for entry in data:
        url = entry.get("url")  # Assuming each entry has a 'url' property
        if url and url not in unique_data:
            unique_data[url] = entry  # Store the entry using its 'url' as a key

    # Convert the dictionary back to a list of unique entries
    cleaned_data = list(unique_data.values())

    # Save the cleaned data back to the JSON file
    with open(file_path, "w") as f:
        json.dump(cleaned_data, f, indent=4)

    print(f"Removed duplicates. Cleaned data saved to {file_path}.")


# Example usage
file_path = "snopes_results.json"
remove_duplicates_by_url(file_path)
