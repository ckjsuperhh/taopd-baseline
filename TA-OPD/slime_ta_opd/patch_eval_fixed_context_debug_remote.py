#!/usr/bin/env python3
from pathlib import Path

path = Path("/path/to/slime-main/tools/eval_fixed_context_bank.py")
text = path.read_text()

repls = [
    (
        '    tok_s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=args.trust_remote_code)\n'
        '    tok_t = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=args.trust_remote_code)\n',
        '    print(f"loading_tokenizers student={args.student} teacher={args.teacher}", flush=True)\n'
        '    tok_s = AutoTokenizer.from_pretrained(args.student, trust_remote_code=args.trust_remote_code)\n'
        '    tok_t = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=args.trust_remote_code)\n'
        '    print("tokenizers_loaded", flush=True)\n',
    ),
    (
        '    contexts = _read_table(Path(args.context_bank))\n'
        '    if args.max_samples is not None:\n'
        '        contexts = contexts.head(args.max_samples)\n'
        '\n'
        '    student = _load_model(args.student, args.student_device, args.dtype, args.trust_remote_code)\n'
        '    teacher = _load_model(args.teacher, args.teacher_device, args.dtype, args.trust_remote_code)\n'
        '\n'
        '    rows = []\n',
        '    print(f"loading_context_bank path={args.context_bank}", flush=True)\n'
        '    contexts = _read_table(Path(args.context_bank))\n'
        '    if args.max_samples is not None:\n'
        '        contexts = contexts.head(args.max_samples)\n'
        '    print(f"context_bank_loaded samples={len(contexts)}", flush=True)\n'
        '\n'
        '    print(f"loading_student_model device={args.student_device} dtype={args.dtype}", flush=True)\n'
        '    student = _load_model(args.student, args.student_device, args.dtype, args.trust_remote_code)\n'
        '    print("student_model_loaded", flush=True)\n'
        '    print(f"loading_teacher_model device={args.teacher_device} dtype={args.dtype}", flush=True)\n'
        '    teacher = _load_model(args.teacher, args.teacher_device, args.dtype, args.trust_remote_code)\n'
        '    print("teacher_model_loaded", flush=True)\n'
        '\n'
        '    print("scoring_start", flush=True)\n'
        '    rows = []\n',
    ),
    (
        '            print(f"scored_samples={ordinal + 1} rows={len(rows)}")\n',
        '            print(f"scored_samples={ordinal + 1} rows={len(rows)}", flush=True)\n',
    ),
    (
        '    print(f"samples={len(contexts)}")\n'
        '    print(f"tokens={len(metrics)}")\n'
        '    print(f"saved={args.output}")\n',
        '    print(f"samples={len(contexts)}", flush=True)\n'
        '    print(f"tokens={len(metrics)}", flush=True)\n'
        '    print(f"saved={args.output}", flush=True)\n',
    ),
    (
        '        print(f"mean_G_KLf={float(finite.mean()) if len(finite) else float(\'nan\'):.6f}")\n',
        '        print(f"mean_G_KLf={float(finite.mean()) if len(finite) else float(\'nan\'):.6f}", flush=True)\n',
    ),
]

for old, new in repls:
    if new in text:
        continue
    if old not in text:
        raise SystemExit(f"pattern not found for replacement starting: {old[:80]!r}")
    text = text.replace(old, new)

path.write_text(text)
print(f"patched {path}")
