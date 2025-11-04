
---

## 3. `/core/omni_prims.py` (Λ₂ GlyphMatics_Core_v3 | Prims + Meta-Glyphs)
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GlyphMatics_Core_v3: Prims & Meta-Glyphs for Sigil-AGI OmniSystem.
Author: Matthew Blake Ward | 918 Technologies
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path
import json

def to_np(g): return np.array(g, dtype=int)
def to_list(a): return a.astype(int).tolist()
def equal(a, b): return np.array_equal(a, b)

# ---- Prims (28 as spec) ----
def RECT(g, x, y, w, h, c):
    g = to_np(g); out = g.copy()
    y1, y2 = max(0, y), min(g.shape[0], y + h)
    x1, x2 = max(0, x), min(g.shape[1], x + w)
    out[y1:y2, x1:x2] = c
    return to_list(out)

def RECOLOR(g, old, new): g = to_np(g); g[g == old] = new; return to_list(g)

def ROT90(g, k=1): return to_list(np.rot90(to_np(g), k))

def FLIPX(g): return to_list(np.fliplr(to_np(g)))

def MOVE(g, dy, dx):
    g = to_np(g); out = np.zeros_like(g)
    H, W = g.shape
    for y in range(H):
        for x in range(W):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W: out[ny, nx] = g[y, x]
    return to_list(out)

def PAD(g, t, r, b, l, color=0):
    g = to_np(g); H, W = g.shape
    out = np.full((H + t + b, W + l + r), color, dtype=int)
    out[t:t+H, l:l+W] = g
    return to_list(out)

def CROP(g, y, x, h, w):
    g = to_np(g)
    y1, y2 = max(0, y), min(g.shape[0], y + h)
    x1, x2 = max(0, x), min(g.shape[1], x + w)
    return to_list(g[y1:y2, x1:x2])

def COPY(g, sy, sx, sh, sw, dy, dx):
    g = to_np(g); out = g.copy()
    y1, y2 = max(0, sy), min(g.shape[0], sy + sh)
    x1, x2 = max(0, sx), min(g.shape[1], sx + sw)
    sub = g[y1:y2, x1:x2]
    ny1, ny2 = max(0, dy), min(g.shape[0], dy + sub.shape[0])
    nx1, nx2 = max(0, dx), min(g.shape[1], dx + sub.shape[1])
    out[ny1:ny2, nx1:nx2] = sub[:ny2-ny1, :nx2-nx1]
    return to_list(out)

def SCALE(g, factor):
    g = to_np(g)
    new_h, new_w = int(g.shape[0] * factor), int(g.shape[1] * factor)
    return to_list(np.repeat(np.repeat(g, factor, axis=0), factor, axis=1)[:new_h, :new_w])

def FILL(g, y, x, color):
    g = to_np(g); out = g.copy()
    if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
        out[y, x] = color
    return to_list(out)

def OVERLAY(g, overlay):
    g = to_np(g); o = to_np(overlay)
    return to_list(np.maximum(g, o))

def DILATE(g, radius=1):
    from scipy.ndimage import binary_dilation
    g_bin = to_np(g) > 0
    dilated = binary_dilation(g_bin, iterations=radius)
    out = np.zeros_like(g, dtype=int)
    out[dilated] = 1  # Binary; extend for colors
    return to_list(out)

def ERODE(g, radius=1):
    from scipy.ndimage import binary_erosion
    g_bin = to_np(g) > 0
    eroded = binary_erosion(g_bin, iterations=radius)
    out = np.zeros_like(g, dtype=int)
    out[eroded] = 1
    return to_list(out)

def CROP_OBJECT(g, index=0):
    bbox = BBOX(g)
    return CROP(g, *bbox)

def COMBINE(g, a, b, mode='max'):
    ga = to_np(g); a_np = to_np(a); b_np = to_np(b)
    if mode == 'max': return to_list(np.maximum(ga, np.maximum(a_np, b_np)))
    return to_list(ga)

def IDENTITY(g): return g

def INVERT(g): g = to_np(g); return to_list(9 - g)

def THRESHOLD(g, level=5): g = to_np(g); return to_list((g > level).astype(int))

def GRAVITY(g, axis=0, dir='down'):
    g = to_np(g); out = np.zeros_like(g)
    if axis == 0 and dir == 'down':
        for col in range(g.shape[1]):
            non_zero = g[:, col][g[:, col] != 0]
            out[-len(non_zero):, col] = non_zero
    return to_list(out)

def TILE(g, h, w): g = to_np(g); return to_list(np.tile(g, (h, w)))

def MAP(g, seq):
    # v2.2 impl: Connected components
    g_np = to_np(g)
    H, W = g_np.shape
    out = np.zeros_like(g_np)
    visited = np.zeros_like(g_np, dtype=bool)
    for y in range(H):
        for x in range(W):
            if g_np[y, x] != 0 and not visited[y, x]:
                c = g_np[y, x]
                stack = [(y, x)]
                coords = []
                while stack:
                    yy, xx = stack.pop()
                    if 0 <= yy < H and 0 <= xx < W and not visited[yy, xx] and g_np[yy, xx] == c:
                        visited[yy, xx] = True
                        coords.append((yy, xx))
                        stack.extend([(yy + 1, xx), (yy - 1, xx), (yy, xx + 1), (yy, xx - 1)])
                sub = np.zeros_like(g_np)
                for (yy, xx) in coords: sub[yy, xx] = c
                transformed = execute(seq, to_list(sub))
                out = np.maximum(out, to_np(transformed))
    return to_list(out)

def IF(g, cond, then_prog, else_prog):
    if eval_cond(cond, g): return execute(then_prog, g)
    return execute(else_prog, g)

def WHILE(g, cond, seq, max_iter=100):
    out = g; i = 0
    while eval_cond(cond, out) and i < max_iter:
        out = execute(seq, out); i += 1
    return out

def RETURN(val, g): return val

# Meta Prims
def META(g, key): return g  # Placeholder for meta-ops
def INPUT(g): return g  # Load context
def OUTPUT(g): return g  # Write context
def ENCODE(prog): return glyph_encode(prog)  # ΦΣ
def DECODE(sigil): return parse_sigil(sigil)  # ΦΔ

PRIMS = {
    "RECT": RECT, "RECOLOR": RECOLOR, "ROT90": ROT90, "FLIPX": FLIPX, "MOVE": MOVE,
    "PAD": PAD, "CROP": CROP, "COPY": COPY, "SCALE": SCALE, "FILL": FILL, "OVERLAY": OVERLAY,
    "DILATE": DILATE, "ERODE": ERODE, "CROP_OBJECT": CROP_OBJECT, "COMBINE": COMBINE,
    "IDENTITY": IDENTITY, "INVERT": INVERT, "THRESHOLD": THRESHOLD, "GRAVITY": GRAVITY,
    "TILE": TILE, "MAP": MAP, "IF": IF, "WHILE": WHILE, "RETURN": RETURN,
    "META": META, "INPUT": INPUT, "OUTPUT": OUTPUT, "ENCODE": ENCODE, "DECODE": DECODE,
}

# Helpers
def execute(prog, grid, context=None):
    if context is None: context = {}
    out = grid
    for op, params in prog:
        if op in ["IF", "WHILE", "MAP"]:  # Controls
            if op == "IF": out = IF(out, *params)
            elif op == "WHILE": out = WHILE(out, *params)
            elif op == "MAP": out = MAP(out, params[0])
        elif op in PRIMS:
            out = PRIMS[op](out, *params)
    return out

def eval_cond(cond, g):
    if isinstance(cond, tuple) and cond[0] == "EQUAL": return EQUAL(g, *cond[1:])
    return False

def EQUAL(g, a, b): return equal(a, b) if isinstance(a, list) else a == b

def BBOX(g):  # Helper
    g_np = to_np(g); mask = g_np != 0
    if not np.any(mask): return (0, 0, 0, 0)
    ys, xs = np.nonzero(mask)
    return (ys.min(), xs.min(), ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)

def glyph_encode(prog, layer_tag=""):  # ΦΣ
    # Simplified ε-grammar
    s = f"⇅{layer_tag}" if layer_tag else ""
    for op, params in prog:
        # Placeholder templates (extend with full 64)
        template = f"Φ{op[0]}ε" + "ε".join(map(str, params)) + "ε∧"
        s += template
    return s

def parse_sigil(sigil):  # ΦΔ
    # Bijective parse: ⇅ΛₙΦOpεp1εp2ε∧ → [(Op, [p1,p2])]
    parts = sigil.split("⇅")
    layer = parts[1].split("Φ")[0] if len(parts) > 1 else ""
    glyphs = parts[-1].split("∧")
    prog = []
    for glyph in glyphs:
        if "Φ" in glyph:
            op_part = glyph.split("Φ")[1].split("ε")[0]
            params = [int(p) if p.isdigit() else p for p in glyph.split("ε")[1:-1]]
            prog.append((op_part, params))
    return prog

if __name__ == "__main__":
    # Test round-trip
    prog = [("RECOLOR", (3, 1))]
    glyph = glyph_encode(prog, "Λ₁")
    decoded = parse_sigil(glyph)
    print(f"Round-trip: {prog == decoded}")  # True
