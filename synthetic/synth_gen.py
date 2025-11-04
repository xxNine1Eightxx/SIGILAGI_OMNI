#!/usr/bin/env python3
"""
Synthetic Task Generator: 360 templates → 10k ARC pairs.
Usage: python synth_gen.py --archetype COLOR --num=1000
"""

import argparse
import json
import random
import numpy as np
from core.omni_prims import execute, glyph_encode

ARCHETYPES = {
    "COLOR": [("RECOLOR", (random.randint(0,9), random.randint(0,9)))],
    # ... (GEOMETRIC: ROT90, etc.; 360 total)
}

def generate_synthetic(num=1000, archetype="COLOR"):
    tasks = []
    for _ in range(num):
        base = np.random.randint(0, 10, (5,5)).tolist()  # Random base
        prog = ARCHETYPES.get(archetype, [("IDENTITY", ())])
        out = execute(prog, base)
        glyph = glyph_encode(prog)
        task = {
            "train": [{"input": base, "output": out}],
            "test": [{"input": np.random.randint(0, 10, (5,5)).tolist()}],
            "sigil": glyph
        }
        tasks.append(task)
    return tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archetype", default="COLOR")
    parser.add_argument("--num", type=int, default=1000)
    args = parser.parse_args()
    synthetic = generate_synthetic(args.num, args.archetype)
    with open("../synthetic_arc_tasks.json", "w") as f:
        json.dump({"tasks": synthetic}, f)
    print(f"Generated {args.num} tasks: {args.archetype}")
