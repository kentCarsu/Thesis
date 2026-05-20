from collections import Counter
from pathlib import Path

# CHANGE THIS PATH if needed
LABEL_DIR = Path(
    r"C:/Users/Lubi/OneDrive/Documents/THESIS_PPE/TESTBARE/ALL DATA/newBESTDATA/train/labels"
)

counter = Counter()

for label_file in LABEL_DIR.glob("*.txt"):
    with open(label_file, "r") as f:
        for line in f:
            cls = int(line.split()[0])
            counter[cls] += 1

print("Class distribution (train set):")
for cls, count in sorted(counter.items()):
    print(f"Class {cls}: {count}")              