import os
import shutil


# CONFIG

LABEL_DIRS = [
    "train/labels",
    "valid/labels",
    "test/labels"
]
BACKUP_SUFFIX = "_backup"


# PROCESS

for label_dir in LABEL_DIRS:
    if not os.path.exists(label_dir):
        print(f"Skipping (not found): {label_dir}")
        continue

    print(f"\nProcessing: {label_dir}")
    print("Files found:", len(os.listdir(label_dir)))

    # Backup
    backup_dir = label_dir + BACKUP_SUFFIX
    if not os.path.exists(backup_dir):
        shutil.copytree(label_dir, backup_dir)
        print(f"Backup created: {backup_dir}")

    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, fname)

        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            cls = int(parts[0])

            # remove person class
            if cls == 0:
                continue

            # shift class index down by 1
            parts[0] = str(cls - 1)
            new_lines.append(" ".join(parts))

        with open(label_path, "w") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    print(f"Finished: {label_dir}")

print("\n Class removal complete.")
