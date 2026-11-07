import urllib.request
import json

url = 'https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/130628817/assets?per_page=100'
req = urllib.request.Request(url, headers={'User-Agent': 'stt-download'})
response = urllib.request.urlopen(req)
data = json.load(response)

print(f'Total assets: {len(data)}')
print()

multilingual_models = []
for asset in data:
    name = asset['name']
    name_lower = name.lower()
    if any(kw in name_lower for kw in ['multilingual', 'parakeet', '14-lang', 'cohere', 'nemo', 'es-', 'pt-', 'spanish', 'portuguese']):
        multilingual_models.append((name, asset['browser_download_url']))

for name, url in multilingual_models:
    print(f'{name}: {url}')

print()
print('All parakeet models:')
for asset in data:
    if 'parakeet' in asset['name'].lower():
        print(f'  {asset["name"]}')