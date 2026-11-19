#!/usr/bin/env python3
"""The release transforms of the paper (called by sync_overleaf.sh; run with python3 -I).

  release.py transform PAPER_DIR      the release build of a freshly flattened Overleaf export (in place)
  release.py prune PAPER_DIR FLS      remove every file under fig/ and latex/ that the build (its .fls) did not read
  release.py arxiv SRC_DIR OUT_DIR    the arXiv source tree: SRC_DIR's build inputs, full-line comments stripped

Every required anchor that is missing is an error (exit 1): the tool never writes a review build silently.
"""
import os
import re
import sys
import tomllib

TOOLS = os.path.dirname(os.path.abspath(__file__))


class Fail(Exception):
    pass


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


def tex_files(root):
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ("tools", "build")]
        for f in fs:
            if f.endswith(".tex"):
                yield os.path.join(d, f)


def sub_required(text, pattern, repl, what, flags=0, count=1):
    new, n = re.subn(pattern, repl, text, count=count, flags=flags)
    if n == 0:
        raise Fail(f"main.tex: anchor not found: {what} (/{pattern}/)")
    return new


def sub_optional(text, pattern, repl, flags=0):
    return re.sub(pattern, repl, text, flags=flags)


# ---------------------------------------------------------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------------------------------------------------------
# Every .tex: the flattened layout (src/{fig,latex,sec} -> {fig,latex,sec}) and the provenance scrubs of the public repository
# (the experiments' directory name; no private commit hashes of the Overleaf repository in comments).
FLATTEN = re.compile(r"(?<![\w./-])src/(?=(?:fig|latex|sec)\b)")
SCRUBS = [
    (re.compile(r"\bcvpr" + "2027/"), "experiments/"),   # the experiments' former directory name (split: the venue grep)
    (re.compile(r" ?\(Blens [0-9a-f]{7,40}\)"), ""),
]


def transform_main(s):
    # the template's banner
    s = sub_required(s, r"\A% CVPR \d{4} Paper Template;[^\n]*",
                     "% The paper (release build); the LaTeX template is the CVF author kit, https://github.com/cvpr-org/author-kit",
                     "the author kit's first line")
    # the paper type: [review] -> [pagenumbers]; the block of commented alternatives goes with it
    s = sub_required(s, r"%{5,} PAPER TYPE[^\n]*\n(?:[ \t]*%[^\n]*\n|[ \t]*\\usepackage\[review\]\{cvpr\}[^\n]*\n)+",
                     "%%%%%%%%% PAPER TYPE: the release (arXiv) build, with page numbers\n\\\\usepackage[pagenumbers]{cvpr}\n",
                     "the PAPER TYPE block with \\usepackage[review]{cvpr}")
    if "[review]{cvpr}" in s:
        raise Fail("main.tex: \\usepackage[review]{cvpr} outside the PAPER TYPE block")
    # the paper ID block
    s = sub_required(s, r"(?:%{5,} PAPER ID[^\n]*\n)?\\def\\paperID\{[^}]*\}[^\n]*\n\\def\\confName\{[^}]*\}[^\n]*\n\\def\\confYear\{[^}]*\}[^\n]*\n\n?",
                     "", "\\def\\paperID / \\confName / \\confYear")
    # the author block: the template's placeholder (or any \author{...}) -> tools/authors.tex
    s = sub_optional(s, r"%{5,} AUTHORS[^\n]*\n", "")
    i = s.find("\\author{")
    if i < 0:
        raise Fail("main.tex: anchor not found: \\author{")
    depth, j = 0, i + len("\\author")
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "%":                              # a comment inside the block: to its end of line
            j = s.find("\n", j)
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    else:
        raise Fail("main.tex: the \\author{ block does not close")
    end = j + 1
    if s[end:end + 1] == "\n":
        end += 1
    s = s[:i] + read(os.path.join(TOOLS, "authors.tex")) + s[end:]
    # cosmetic: the template's review/instruction comments (optional anchors)
    s = sub_optional(s, r"% It is strongly recommended to use hyperref[^\n]*\n(?:%[^\n]*\n)*", "")
    s = sub_optional(s, r"%{5,} TITLE[^\n]*\n", "")
    s = sub_optional(s, r"\n\n(?:% \\input\{sec/[123]_(?:intro|formatting|finalcopy)\}\n)+", "\n")
    s = sub_optional(s, r"% WARNING: do not forget to delete the supplementary pages[^\n]*\n",
                     "% The supplementary material, in the same document so that its cross-references to the paper resolve\n")
    return s


def apply_edits(paper):
    """tools/release_edits.toml: anchored text edits of the release (each applied when its `old` occurs exactly once; reported
    as not applicable when it does not occur — the upstream text changed, so the edit is no longer needed or must be redone)."""
    cfg = tomllib.loads(read(os.path.join(TOOLS, "release_edits.toml")))
    for e in cfg.get("edit", []):
        p = os.path.join(paper, e["file"])
        if not os.path.isfile(p):
            print(f"  release edit [{e['id']}]: {e['file']} not in the export — not applicable")
            continue
        s = read(p)
        n = s.count(e["old"])
        if n == 1:
            write(p, s.replace(e["old"], e["new"]))
            print(f"  release edit [{e['id']}]: applied")
        elif n == 0:
            print(f"  release edit [{e['id']}]: anchor absent — not applicable (resolved upstream? review it)")
        else:
            raise Fail(f"release edit [{e['id']}]: its anchor occurs {n} times in {e['file']}")


TODO_USE = re.compile(r"\\(?:todo|TODO)\{")


def check_release(paper):
    errs = []
    main = read(os.path.join(paper, "main.tex"))
    if main.count("\\usepackage[pagenumbers]{cvpr}") != 1:
        errs.append("main.tex: not exactly one \\usepackage[pagenumbers]{cvpr}")
    if read(os.path.join(TOOLS, "authors.tex")) not in main:
        errs.append("main.tex: the author block is not tools/authors.tex")
    for p in tex_files(paper):
        rel = os.path.relpath(p, paper)
        for k, line in enumerate(read(p).splitlines(), 1):
            code = re.sub(r"(?<!\\)%.*", "", line)
            if re.search(r"\\(?:paperID|confName|confYear)\b", code) or "[review]{cvpr}" in code:
                errs.append(f"{rel}:{k}: review-build macro: {line.strip()}")
            if rel != "preamble.tex" and TODO_USE.search(code):
                errs.append(f"{rel}:{k}: an open \\todo of the Overleaf text (resolve it on Overleaf; "
                            f"the release never decides the authors' text): {line.strip()[:160]}")
    for x in ("rebuttal.tex", "sec_full", "README.md", ".github"):
        if os.path.exists(os.path.join(paper, x)):
            errs.append(f"{x}: review/author-kit material in the release tree")
    if errs:
        raise Fail("release checks failed:\n  " + "\n  ".join(errs))


def transform(paper):
    for p in tex_files(paper):
        s = read(p)
        t = FLATTEN.sub("", s)
        for rx, r in SCRUBS:
            t = rx.sub(r, t)
        if t != s:
            write(p, t)
    mp = os.path.join(paper, "main.tex")
    write(mp, transform_main(read(mp)))
    apply_edits(paper)
    check_release(paper)


# ---------------------------------------------------------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------------------------------------------------------
def fls_inputs(paper, fls):
    pwd, read_ = paper, set()
    for line in read(fls).splitlines():
        if line.startswith("PWD "):
            pwd = line[4:]
        elif line.startswith("INPUT "):
            p = line[6:]
            p = os.path.normpath(p if os.path.isabs(p) else os.path.join(pwd, p))
            read_.add(p)
    return read_


def prune(paper, fls):
    paper = os.path.abspath(paper)
    used = fls_inputs(paper, fls)
    removed = []
    for top in ("fig", "latex"):
        for d, dirs, fs in os.walk(os.path.join(paper, top), topdown=False):
            for f in fs:
                p = os.path.join(d, f)
                if os.path.normpath(p) not in used:
                    os.remove(p)
                    removed.append(os.path.relpath(p, paper))
            if not os.listdir(d):
                os.rmdir(d)
    for f in sorted(removed):
        print(f"  pruned (not read by the build): {f}")
    unread = [os.path.relpath(p, paper) for p in tex_files(os.path.join(paper, "sec")) if os.path.normpath(p) not in used]
    for f in unread:
        print(f"  WARNING: {f} is not read by the build (kept: only fig/ and latex/ are pruned)")


# ---------------------------------------------------------------------------------------------------------------------------
# arxiv: the build inputs with full-line comments removed (inline comments and a line's trailing % are kept, as the package of
# 2026-11-18 has them); a .tex left with no content gets \relax; main.bib/.bst are replaced by build/main.bbl (copied by the caller)
# ---------------------------------------------------------------------------------------------------------------------------
ARXIV_SKIP = {"Makefile", ".gitignore", "main.bib", "ieeenat_fullname.bst"}


def strip_comments(s):
    t = "".join(l for l in s.splitlines(keepends=True) if not re.match(r"[ \t]*%", l))
    return t if t else "\\relax\n"     # a file of comments only: \relax (an empty input file is legal, but kept explicit)


def arxiv(src, out):
    for d, dirs, fs in os.walk(src):
        dirs[:] = [x for x in dirs if x not in ("tools", "build") and not x.startswith(".")]
        for f in fs:
            rel = os.path.relpath(os.path.join(d, f), src)
            if rel in ARXIV_SKIP or f.startswith("."):
                continue
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if f.endswith(".tex"):
                write(dst, strip_comments(read(os.path.join(d, f))))
            else:
                with open(os.path.join(d, f), "rb") as a, open(dst, "wb") as b:
                    b.write(a.read())
    m = read(os.path.join(out, "main.tex"))
    if re.search(r"\\(?:paperID|confName|confYear)\b|\[review\]\{cvpr\}", m):
        raise Fail("arxiv: main.tex carries a review-build macro")


def main(argv):
    try:
        if argv[:1] == ["transform"] and len(argv) == 2:
            transform(argv[1])
        elif argv[:1] == ["prune"] and len(argv) == 3:
            prune(argv[1], argv[2])
        elif argv[:1] == ["arxiv"] and len(argv) == 3:
            arxiv(argv[1], argv[2])
        else:
            print(__doc__, file=sys.stderr)
            return 2
    except Fail as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
