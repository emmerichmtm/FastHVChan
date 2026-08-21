# Vendored benchmark contenders

- `yildiz_suri_anchor.py` — Yıldız–Suri-style anchored hypervolume
  implementation, vendored unmodified (apart from the file name, made
  importable) from
  https://github.com/emmerichmtm/ImplementationOfYilidizAndSuriSubquadratic4DHypervolume
  (author: Michael T. M. Emmerich, with ChatGPT; 2026). It computes the
  volume of a union of boxes anchored at the origin `[0, p]`; the
  benchmark adapter converts a minimisation instance with reference point
  `r` via the transform `q = r − y`. Used purely as a benchmark contender —
  see Yıldız & Suri, "On Klee's measure problem for grounded boxes",
  SoCG 2012, for the underlying algorithm.
