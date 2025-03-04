from serpapi import GoogleSearch
import json

params = {
  "q": "fresh apples vendors",
  "location": "Ahmedabad, Gujarat, India",
  "hl": "en",
  "gl": "us",
  "google_domain": "google.com",
  "as_epq": "email",
  "api_key": "45918ac244a7df014a4a5fa1f90f07fb21aa194e0d01ff91fc9addcf54a5b0b5"
}

search = GoogleSearch(params)
results = search.get_dict()
print(results)

with open('formatted_output.json', 'w') as f:
    f.truncate(0)
    json.dump(results, f, indent=2)