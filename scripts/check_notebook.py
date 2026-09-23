#!/usr/bin/env python3
"""Syntax-check notebook code cells without executing them.

The subtlety: a naive checker that deletes any line starting with `!` or `%`
also eats format-operator continuations like

    print("loss %.4f"
          % (x,))                     <- not a magic

and shell-command continuations after a trailing backslash. Both produce
phantom "'(' was never closed" errors. So magics are replaced with `pass`
(preserving indentation) rather than deleted, and `%` only counts as a magic
when followed by a letter.
"""
import ast, json, re, sys


def is_magic(line):
    t = line.lstrip()
    return t.startswith("!") or bool(re.match(r"%[a-zA-Z]", t))


def to_python(src):
    out, skipping = [], False
    for line in src.split("\n"):
        indent = line[:len(line) - len(line.lstrip())]
        if skipping:
            # Continuation of a shell command. Emit a BLANK line, not an
            # indented `pass` — the latter is itself an indent error, which is
            # the bug this comment exists to stop someone reintroducing.
            skipping = line.rstrip().endswith("\\")
            out.append("")
            continue
        if is_magic(line):
            skipping = line.rstrip().endswith("\\")
            out.append(indent + "pass")
        else:
            out.append(line)
    return "\n".join(out)


def check(path):
    nb = json.load(open(path))
    bad = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        try:
            ast.parse(to_python("".join(c["source"])))
        except SyntaxError as e:
            bad += 1
            print("  cell %d: %s (line %s)" % (i, e.msg, e.lineno))
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print("%s: %d cells (%d code) — %s"
          % (path, len(nb["cells"]), n_code, "clean" if not bad else "%d error(s)" % bad))
    return bad


if __name__ == "__main__":
    sys.exit(min(1, sum(check(p) for p in (sys.argv[1:] or ["notebooks/train_multimodal.ipynb"]))))
