import json 
from datetime import datetime
import os 

def log_interaction(question, answer, corpus):
    timestamp = datetime.now().isoformat()  
    os.makedirs("logs", exist_ok=True)

    log_entry = {
        "timestamp": timestamp,
        "corpus": corpus,
        "question": question,
        "answer": answer
    }

    with open(f"logs/{corpus}_chatlog.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
