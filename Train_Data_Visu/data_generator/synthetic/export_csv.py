import csv
from rc import generate as rc_gen
from cascaded import generate as cascaded_gen
from feedback import generate as feedback_gen
from bjt_amp import generate as bjt_gen
from mos_amp import generate as mos_gen
import random

rng = random.Random(42)

dataset = []
dataset += rc_gen(5000)
dataset += cascaded_gen(2000)
dataset += feedback_gen(2000)
dataset += bjt_gen(2000)
dataset += mos_gen(2000)

rng.shuffle(dataset)

with open("results_augmented.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["input_text", "output_text"])
    for nl, spice in dataset:
        writer.writerow([nl, spice])

print(f"Generated {len(dataset)} samples -> results_augmented.csv")
