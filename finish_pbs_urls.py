import json

with open("pbs_results_cleaned.json", "r") as f:
    data = json.load(f)

json_links = [item["url"] for item in data]

with open("all_links_final.txt", "r") as f:
    links = f.read().splitlines()

uncompleted_links = [link for link in links if link not in json_links]

with open("uncompleted_links.txt", "w") as f:
    for link in uncompleted_links:
        f.write(link + "\n")
