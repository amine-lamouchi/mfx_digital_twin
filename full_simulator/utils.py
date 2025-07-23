import os
from datetime import datetime

def make_experiment_dir(task_name, optimizer_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join("experiments", task_name, f"{optimizer_name}_{timestamp}")
    os.makedirs(base, exist_ok=True)
    return base
