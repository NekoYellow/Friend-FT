import json

ALL_DATA_FILE = "chatdata_all.json"
TEXT_DATA_FILE = "chatdata_text.json"

with open(ALL_DATA_FILE, "r", encoding="utf-8") as f:
    data_all = json.load(f)

data_text = []
for entry in data_all:
    if entry["type"] == 1:
        data_text.append({
            "time": entry["time"],
            "is_self": entry["isSelf"],
            "content": entry["content"],
        })

print(len(data_all), len(data_text))

with open(TEXT_DATA_FILE, "w", encoding="utf-8") as f:
    json.dump(data_text, f, ensure_ascii=False, indent=2)
