from __future__ import annotations

import importlib
import inspect
from pathlib import Path


MODULES = [
    "src.phase4.prepare_phase4_data",
    "src.phase4.phase4_dataset",
    "src.phase4.phase4_models",
    "src.phase4.phase4_train_mlp",
    "src.phase4.phase4_train_gnn",
    "src.phase4.phase4_infer",
    "src.phase4.phase4_evaluate",
    "src.phase4.run_phase4",
]


def short_sig(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "(signature unavailable)"


def main() -> None:
    for mod_name in MODULES:
        print(f"\n===== {mod_name} =====")
        mod = importlib.import_module(mod_name)
        print(f"file: {inspect.getsourcefile(mod)}")

        public_names = [n for n in dir(mod) if not n.startswith("_")]
        for name in public_names:
            obj = getattr(mod, name)

            if inspect.isfunction(obj):
                print(f"[fn] {name}{short_sig(obj)}")
            elif inspect.isclass(obj):
                print(f"[class] {name}{short_sig(obj)}")
            else:
                t = type(obj).__name__
                if name.isupper():
                    print(f"[const] {name}: {t}")

    print("\n===== phase4 artifacts tree =====")
    for root in [
        Path("artifacts/phase4"),
        Path("reports/phase4"),
    ]:
        print(f"\n-- {root} --")
        if not root.exists():
            print("missing")
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                print(path)


if __name__ == "__main__":
    main()