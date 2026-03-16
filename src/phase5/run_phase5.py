from src.phase5.build_gnn_score_only_features import main as build_features
from src.phase5.train_phase5_hybrid_lgbm import main as train_hybrid


def main() -> None:
    build_features()
    train_hybrid()


if __name__ == "__main__":
    main()