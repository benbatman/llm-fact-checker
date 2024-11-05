import json

with open("pbs_results_cleaned.json", "r") as f:
    data = json.load(f)

# Get all url values and put in list
urls = []
for item in data:
    urls.append(item["url"])
