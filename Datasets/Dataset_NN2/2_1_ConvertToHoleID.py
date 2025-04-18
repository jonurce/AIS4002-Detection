import os

def convert_to_hole_id(input_label_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    for label_file in os.listdir(input_label_dir):
        if label_file.endswith(".txt"):
            new_lines = []
            with open(os.path.join(input_label_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 6:
                        print(f"Warning: Invalid line in {label_file}: {line.strip()}")
                        continue
                    hole_id, x, y, w, h, empty_full = map(float, parts)
                    new_lines.append(f"{int(2*hole_id+empty_full)} {x} {y} {w} {h}")
            with open(os.path.join(output_label_dir, label_file), 'w') as f:
                f.write("\n".join(new_lines))
            print(f"Converted {label_file} to {output_label_dir}")

# Convert for each split
for split in ["train", "val", "test"]:
    input_label_dir = "Annotations_NN2_complete"
    output_label_dir = "Annotations_NN2_hole_id"
    if os.path.exists(input_label_dir):
        convert_to_hole_id(input_label_dir, output_label_dir)
    else:
        print(f"Warning: {input_label_dir} not found, skipping {split}")