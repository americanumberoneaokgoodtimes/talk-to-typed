import urllib.request
import json

url = 'https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/130628817/assets?per_page=200'
req = urllib.request.Request(url, headers={'User-Agent': 'stt-download'})
response = urllib.request.urlopen(req)
data = json.load(response)

print(f'Total assets: {len(data)}')
print()

# Look for parakeet 1.1b multilingual
for asset in data:
    name = asset['name']
    if '1.1b' in name and 'multilingual' in name.lower():
        print(f'FOUND: {name}: {asset["browser_download_url"]}')
    elif 'parakeet' in name.lower() and '1.1' in name:
        print(f'PARAKEET: {name}: {asset["browser_download_url"]}')
    elif 'nllb' in name.lower():
        print(f'NLLB: {name}: {asset["browser_download_url"]}')

# Print all assets for manual review
print("\n--- ALL ASSETS ---")
for asset in data:
    print(asset['name'])