import urllib.request
import json

url = 'https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/v2.0.1'
req = urllib.request.Request(url, headers={'User-Agent': 'stt-download'})
response = urllib.request.urlopen(req)
data = json.load(response)

print('Release:', data['tag_name'])
print('Assets:', len(data.get('assets', [])))

for asset in data.get('assets', []):
    name = asset['name']
    name_lower = name.lower()
    if any(kw in name_lower for kw in ['multilingual', 'parakeet', '1.1b', 'dolphin', 'moonshine', 'cohere', 'nllb', 'translation']):
        print(f'  {name}: {asset["browser_download_url"]}')