import urllib.request
import json

url = 'https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/v2.0.1'
req = urllib.request.Request(url, headers={'User-Agent': 'stt-download'})
response = urllib.request.urlopen(req)
data = json.load(response)

# Get assets URL
assets_url = data['assets_url'] + '?per_page=300'
req2 = urllib.request.Request(assets_url, headers={'User-Agent': 'stt-download'})
response2 = urllib.request.urlopen(req2)
assets = json.load(response2)

print('Total assets:', len(assets))

for asset in assets:
    name = asset['name']
    name_lower = name.lower()
    if any(kw in name_lower for kw in ['multilingual', 'parakeet', '1.1b', 'dolphin', 'moonshine', 'cohere', 'nllb', 'translation', 'spanish', 'portuguese', 'es-', 'pt-']):
        print(f'  {name}: {asset["browser_download_url"]}')