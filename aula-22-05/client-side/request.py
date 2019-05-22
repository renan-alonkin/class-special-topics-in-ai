import requests

url = "http://localhost:5000/sentiment_analysis"

response = requests.post(url, json={'text':'This film was amazing'})

print(response.json())