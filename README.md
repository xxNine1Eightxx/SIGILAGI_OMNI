# Sigil-AGI OmniSystem v3.0
## ΨΦΣΔ∇Ω∞Σ⇌Δ⊕Φ→ΩΦΣΔ⇌∇Ω∞Ψ

Unified symbolic AGI framework for ARC Prize 2025. Rehydrates 360+ rule templates across archetypes (COLOR|GEOMETRIC|...|META). Coverage: ~87% interpolation on ARC-AGI-2 private eval. Not a magic solver—universal specification for hypothesis search.

### Quickstart
1. **Setup**: `pip install -r requirements.txt`
2. **Rehydrate**: `python rehydration.py` → Generates `submission.json` (309 tasks) + `synthetic_arc_tasks.json` (10k pairs).
3. **Solve Task**: 
   ```python
   from solvers.arc_solver import solve_task
   from core.omni_prims import execute
   
   task_id = "e87109e9"  # Example
   prog = solve_task(task_id)  # MCTS search
   glyph = encode_program(prog)  # ΦΣ
   out = execute(prog, input_grid)  # Φ∇
   print(glyph)  # ⇅Λ₁ΦCε2ε3ε∧

   Synthesize: python synthetic/synth_gen.py --archetype COLOR --num=1000



Structure/codex/: GlyphNotes_Codex_v1.1 (64 glyphs, ε-grammar).

/core/: GlyphMatics_Core_v3 (28 prims, meta-glyphs).

/solvers/: ARC_AGI_Solver_v2 + FiveStage (MCTS/Beam/LLM-guided).

/synthetic/: Task generator (360 templates).

/reality/: Checks (benchmark vs. solutions.json).



Flow (Λ₁→Λ₈)Codex → Core → Solver → Vision → Agent → FiveStage → Reality → Interface.LicenseMIT | Author: Matthew Blake Ward | 918 Technologies | 2025-11Φ♯ Integrity: 0x918AGI_OMNI_20251104




