from __future__ import annotations

from src.phase4.prepare_phase4_data import main as prepare_phase4_data
from src.phase4.phase4_train_gnn import train_gnn
from src.phase4.phase4_train_mlp import train_mlp


def main() -> None:
    prepare_phase4_data()
    train_mlp()
    train_gnn()


if __name__ == "__main__":
    main()