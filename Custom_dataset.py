import os
import shutil


# CONFIG

SOURCE_DATASET = r"C:\Users\Lubi\OneDrive\Documents\THESIS_PPE\TESTBARE\BESTDATA\Gloves2"
TARGET_DATASET = "."  # BESTDATA root

SOURCE_GLOVE_ID = 0    
TARGET_GLOVE_ID = 2    

PREFIX = "Gloves2"

SPLITS = ["train", "valid", "test"]


# MERGE

for split in SPLITS:
    src_img = os.path.join(SOURCE_DATASET, split, "images")
    src_lbl = os.path.join(SOURCE_DATASET, split, "labels")

    dst_img = os.path.join(TARGET_DATASET, split, "images")
    dst_lbl = os.path.join(TARGET_DATASET, split, "labels")

    if not os.path.exists(src_lbl):
        print(f"Skipping {split} (no labels)")
        continue

    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    print(f"\nProcessing {split}...")

    for lbl_file in os.listdir(src_lbl):
        if not lbl_file.endswith(".txt"):
            continue

        lbl_path = os.path.join(src_lbl, lbl_file)

        with open(lbl_path) as f:
            lines = f.readlines()

        helmet_lines = []
        for line in lines:
            cls, *rest = line.strip().split()
            if int(cls) == SOURCE_GLOVE_ID:
                helmet_lines.append(
                    f"{TARGET_GLOVE_ID} " + " ".join(rest)
                )

        if not helmet_lines:
            continue  # no helmet → skip image

        img_name = lbl_file.replace(".txt", ".jpg")
        img_path = os.path.join(src_img, img_name)

        if not os.path.exists(img_path):
            continue

        new_img = f"{PREFIX}_{img_name}"
        new_lbl = f"{PREFIX}_{lbl_file}"

        shutil.copy(img_path, os.path.join(dst_img, new_img))

        with open(os.path.join(dst_lbl, new_lbl), "w") as f:
            f.write("\n".join(helmet_lines) + "\n")

    print(f"{split} done")

print("\ngloves → BESTDATA merge COMPLETE")