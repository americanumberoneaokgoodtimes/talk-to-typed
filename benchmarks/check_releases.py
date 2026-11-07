import urllib.request
import json

url = 'https://api.github.com/repos/k2-fsa/sherpa-onnx/releases?per_page=10'
req = urllib.request.Request(url, headers={'User-Agent': 'stt-download'})
response = urllib.request.urlopen(req)
data = json.load(response)

for release in data:
    print('Release:', release['tag_name'], '-', release['name'])
    print('  Published:', release['published_at'])
    print()