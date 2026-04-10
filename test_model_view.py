"""Test model_view with augmentation."""
import requests
import json

r = requests.get("http://localhost:5050/api/model_view/001_576_440.mp4", params={
    "frame": 100, "x": 200, "y": 150,
    "dir_x": 0.7, "dir_y": -0.3, "seed": 42,
    "augment": "true",
})
d = r.json()

for k, v in d.items():
    if k in ("crop_image", "augmented_image"):
        if v:
            print(f"{k}: {len(v)} chars base64")
        else:
            print(f"{k}: None")
    else:
        print(f"{k}: {json.dumps(v, indent=2)}")
