import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.data_utils import read_skeleton_file, parse_skeleton_filename, NTU_CONNECTIONS, normalize_skeleton

app = FastAPI(title="S-JEPA Demo API")

# Add CORS middleware to allow Next.js frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "/media/ibmelab/ibme31/sjepa/DATA/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons"

# Action classes (simplified mock map for demo based on NTU classes)
ACTION_NAMES = {
    18: "Take off shoe",
    23: "Hand waving",
    27: "Jump up",
    43: "Falling",
    50: "Punching",
    72: "Kicking",
    102: "Walking",
}

@app.get("/api/skeletons")
def list_skeletons():
    if not os.path.exists(DATA_DIR):
        return {"files": []}
    files = []
    try:
        with os.scandir(DATA_DIR) as it:
            for entry in it:
                if entry.name.endswith(".skeleton") and entry.is_file():
                    files.append(entry.name)
                    if len(files) >= 3:
                        break
    except Exception as e:
        print("Error reading directory:", e)
        
    return {"files": sorted(files)}

@app.get("/api/predict/{filename}")
def predict_action(filename: str):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        joints, info = read_skeleton_file(filepath)
        joints = normalize_skeleton(joints)
        # joints shape: (frames, bodies, joints, 3)
        # Convert numpy array to list for JSON response
        joints_list = joints.tolist()
        
        # Mock prediction based on filename (Extract Action ID)
        meta = parse_skeleton_filename(filename)
        action_id = meta["action"] if meta else 0
        action_name = ACTION_NAMES.get(action_id, f"Action {action_id}")
        
        # Generate mock saliency map (attention on joints)
        # Saliency is (frames, joints)
        num_frames = joints.shape[0]
        saliency = np.random.rand(num_frames, 25).tolist()
        
        return {
            "filename": filename,
            "metadata": meta,
            "skeleton": joints_list,
            "connections": NTU_CONNECTIONS,
            "prediction": {
                "top_1": action_name,
                "confidence": round(np.random.uniform(0.85, 0.99), 4),
                "other_classes": [
                    {"name": ACTION_NAMES.get(a, f"Action {a}"), "conf": round(np.random.uniform(0.01, 0.1), 4)}
                    for a in np.random.choice(list(ACTION_NAMES.keys()), 2, replace=False)
                ]
            },
            "saliency": saliency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
