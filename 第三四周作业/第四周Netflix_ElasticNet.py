"""第四周作业：用 Elastic Net 预测 Netflix Prize 用户评分。

首次运行会在 data/ 中缓存 Kaggle 数据卡所指向的原始压缩包（约 698 MB），
从中按固定随机种子抽取 150 部电影、每部最多 100 条评分。缓存目录已加入 .gitignore。
"""

from __future__ import annotations

import argparse
import random
import re
import tarfile
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARCHIVE_PATH = DATA_DIR / "nf_prize_dataset.tar.gz"
N_MOVIES = 150
RATINGS_PER_MOVIE = 100
SEED = 2026
SAMPLE_PATH = DATA_DIR / f"netflix_sample_m{N_MOVIES}_r{RATINGS_PER_MOVIE}_s{SEED}.csv"
RESULT_PATH = BASE_DIR / "第四周Netflix模型结果.csv"
COEFFICIENT_PATH = BASE_DIR / "第四周Netflix非零系数.csv"
PLOT_PATH = BASE_DIR / "第四周NetflixElasticNet评估.png"
ARCHIVE_URL = "https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz"
EXPECTED_ARCHIVE_BYTES = 697_552_028
RATING_COLUMNS = ["user_id", "movie_id", "rating", "rating_date"]
ALPHAS = np.logspace(-5, 1, 25)
L1_RATIOS = [0.01, 0.05, 0.1, 0.5, 0.9, 1.0]


def ensure_archive() -> None:
    """按 Kaggle 数据卡提供的原始来源下载压缩包，只保存在本地缓存。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists():
        if ARCHIVE_PATH.stat().st_size < EXPECTED_ARCHIVE_BYTES:
            raise RuntimeError(
                f"本地压缩包大小异常：{ARCHIVE_PATH.stat().st_size} bytes。"
                "请检查下载是否完整后再运行。"
            )
        return

    partial_path = ARCHIVE_PATH.with_suffix(ARCHIVE_PATH.suffix + ".partial")
    if partial_path.exists():
        raise FileExistsError(
            f"检测到未完成下载 {partial_path}。请先检查该文件，再手动恢复或移走。"
        )
    print("正在下载 Netflix Prize 数据包（约 698 MB）……")
    downloaded = 0
    try:
        with urlopen(ARCHIVE_URL, timeout=120) as response, partial_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded % (50 * 1024 * 1024) < 1024 * 1024:
                    print(f"已下载 {downloaded / (1024 * 1024):.0f} MB")
    except Exception:
        print(f"下载未完成，临时文件保留在：{partial_path}")
        raise
    if downloaded < EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(
            f"下载文件不完整：{downloaded} bytes，预期至少 {EXPECTED_ARCHIVE_BYTES} bytes。"
        )
    partial_path.replace(ARCHIVE_PATH)
    print(f"数据包已缓存：{ARCHIVE_PATH}")


def sample_one_movie(
    inner_tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    movie_id: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    """从一个电影的评分流中进行有上限的蓄水池抽样。"""
    source = inner_tar.extractfile(member)
    if source is None:
        return []
    reservoir: list[tuple[str, int, str]] = []
    count = 0
    with source:
        header = source.readline().decode("ascii").strip()
        if header != f"{movie_id}:":
            raise ValueError(f"评分文件标题与文件名不一致：{member.name} / {header}")
        for raw_line in source:
            user_id, rating, rating_date = raw_line.decode("ascii").strip().split(",")
            count += 1
            row = (user_id, int(rating), rating_date)
            if count <= RATINGS_PER_MOVIE:
                reservoir.append(row)
            else:
                index = rng.randrange(count)
                if index < RATINGS_PER_MOVIE:
                    reservoir[index] = row
    return [
        {
            "user_id": user_id,
            "movie_id": str(movie_id),
            "rating": rating,
            "rating_date": rating_date,
        }
        for user_id, rating, rating_date in reservoir
    ]


def build_balanced_sample() -> pd.DataFrame:
    """流式读取内层 training_set.tar，抽取分布于全片库的电影评分。"""
    ensure_archive()
    rng = random.Random(SEED)
    movie_ids = sorted(rng.sample(range(1, 17_771), N_MOVIES))
    selected = set(movie_ids)
    found: set[int] = set()
    rows: list[dict[str, object]] = []
    movie_file = re.compile(r"mv_(\d+)\.txt$")

    with tarfile.open(ARCHIVE_PATH, mode="r|gz") as outer_tar:
        for outer_member in outer_tar:
            if not outer_member.name.endswith("training_set.tar"):
                continue
            inner_stream = outer_tar.extractfile(outer_member)
            if inner_stream is None:
                raise RuntimeError("无法读取压缩包中的 training_set.tar。")
            with tarfile.open(fileobj=inner_stream, mode="r|") as inner_tar:
                for member in inner_tar:
                    match = movie_file.search(member.name)
                    if not match or not member.isfile():
                        continue
                    movie_id = int(match.group(1))
                    if movie_id not in selected:
                        continue
                    rows.extend(sample_one_movie(inner_tar, member, movie_id, rng))
                    found.add(movie_id)
                    if len(found) % 25 == 0:
                        print(f"已采样 {len(found)}/{N_MOVIES} 部电影")
                    if len(found) == N_MOVIES:
                        break
            break

    if len(found) != N_MOVIES:
        missing = sorted(selected - found)
        raise RuntimeError(f"只找到 {len(found)}/{N_MOVIES} 部目标电影，缺少：{missing[:10]}")
    sample = pd.DataFrame(rows, columns=RATING_COLUMNS)
    sample["rating_year"] = sample["rating_date"].str.slice(0, 4).astype(int)
    sample["rating_month"] = sample["rating_date"].str.slice(5, 7).astype(int)
    sample.to_csv(SAMPLE_PATH, index=False, encoding="utf-8-sig")
    return sample


def load_sample(refresh: bool) -> pd.DataFrame:
    if refresh:
        sample = build_balanced_sample()
    elif SAMPLE_PATH.exists():
        sample = pd.read_csv(SAMPLE_PATH, dtype={"user_id": str, "movie_id": str})
    else:
        sample = build_balanced_sample()
    required = set(RATING_COLUMNS + ["rating_year", "rating_month"])
    missing = required - set(sample.columns)
    if missing:
        raise ValueError(f"本地样本缺少字段：{sorted(missing)}。请使用 --refresh-sample 重建。")
    if not sample["rating"].between(1, 5).all():
        raise ValueError("评分必须全部在 1 到 5 星之间。")
    return sample


def save_evaluation_plot(
    y_test: pd.Series, baseline: np.ndarray, prediction: np.ndarray
) -> None:
    clipped_prediction = np.clip(prediction, 1, 5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.arange(0.5, 5.6, 1)
    axes[0].hist(y_test, bins=bins, alpha=0.65, label="Actual", density=True)
    axes[0].hist(clipped_prediction, bins=bins, alpha=0.6, label="Elastic Net", density=True)
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xlabel("Rating (1–5)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Actual and predicted rating distributions")
    axes[0].legend()

    axes[1].scatter(y_test, clipped_prediction, s=12, alpha=0.25, label="Elastic Net")
    axes[1].scatter(y_test, baseline, s=12, alpha=0.2, label="Mean baseline")
    axes[1].plot([1, 5], [1, 5], linestyle="--", color="black", linewidth=1)
    axes[1].set_xlim(0.8, 5.2)
    axes[1].set_ylim(0.8, 5.2)
    axes[1].set_xticks(range(1, 6))
    axes[1].set_xlabel("Actual rating")
    axes[1].set_ylabel("Predicted rating")
    axes[1].set_title("Predictions on the held-out test set")
    axes[1].legend()
    fig.suptitle("Netflix Prize rating prediction with Elastic Net")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_analysis(sample: pd.DataFrame) -> pd.DataFrame:
    features = sample[["user_id", "movie_id", "rating_year", "rating_month"]]
    target = sample["rating"].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=SEED
    )
    preprocessor = ColumnTransformer(
        [
            (
                "ids",
                OneHotEncoder(handle_unknown="ignore"),
                ["user_id", "movie_id"],
            ),
            ("date", StandardScaler(), ["rating_year", "rating_month"]),
        ],
        sparse_threshold=1.0,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        estimator=Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    ElasticNet(
                        max_iter=5_000,
                        tol=1e-3,
                        precompute=False,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        param_grid={"model__alpha": ALPHAS, "model__l1_ratio": L1_RATIOS},
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=4,
        refit=True,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_.named_steps["model"]
    prediction = search.predict(x_test)
    clipped = np.clip(prediction, 1, 5)

    baseline_model = DummyRegressor(strategy="mean").fit(
        np.zeros((len(y_train), 1)), y_train
    )
    baseline = baseline_model.predict(np.zeros((len(y_test), 1)))
    metrics = pd.DataFrame(
        [
            {
                "model": "Training-mean baseline",
                "rows": len(sample),
                "movies": sample["movie_id"].nunique(),
                "users": sample["user_id"].nunique(),
                "train_rows": len(y_train),
                "test_rows": len(y_test),
                "alpha": np.nan,
                "l1_ratio": np.nan,
                "nonzero_features": 0,
                "test_rmse": np.sqrt(mean_squared_error(y_test, baseline)),
                "test_mae": mean_absolute_error(y_test, baseline),
                "test_r2": r2_score(y_test, baseline),
            },
            {
                "model": "ElasticNet GridSearchCV (clipped to 1–5)",
                "rows": len(sample),
                "movies": sample["movie_id"].nunique(),
                "users": sample["user_id"].nunique(),
                "train_rows": len(y_train),
                "test_rows": len(y_test),
                "alpha": float(model.alpha),
                "l1_ratio": float(model.l1_ratio),
                "nonzero_features": int(np.count_nonzero(np.abs(model.coef_) > 1e-6)),
                "cv_rmse": np.sqrt(-search.best_score_),
                "test_rmse": np.sqrt(mean_squared_error(y_test, clipped)),
                "test_mae": mean_absolute_error(y_test, clipped),
                "test_r2": r2_score(y_test, clipped),
            },
        ]
    )
    metrics.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    fitted_preprocessor = search.best_estimator_.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    coefficients = pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_}
    )
    coefficients = coefficients.loc[coefficients["coefficient"].abs() > 1e-6]
    coefficients.sort_values(
        "coefficient", key=lambda values: values.abs(), ascending=False
    ).head(100).to_csv(COEFFICIENT_PATH, index=False, encoding="utf-8-sig")

    save_evaluation_plot(y_test, baseline, prediction)
    print(f"样本评分：{len(sample)}；电影：{sample['movie_id'].nunique()}；用户：{sample['user_id'].nunique()}")
    print(
        f"ElasticNet：alpha={model.alpha:.6g}, l1_ratio={model.l1_ratio:.3f}, "
        f"CV RMSE={np.sqrt(-search.best_score_):.4f}"
    )
    print(
        f"5 折 GridSearchCV；随机种子：{SEED}；评分按 1–5 截断后计算测试集指标。"
    )
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"结果表：{RESULT_PATH}")
    print(f"系数摘要：{COEFFICIENT_PATH}")
    print(f"评估图：{PLOT_PATH}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-sample",
        action="store_true",
        help="重新从本地 Netflix Prize 原始包抽取固定样本。",
    )
    args = parser.parse_args()
    sample = load_sample(refresh=args.refresh_sample)
    run_analysis(sample)


if __name__ == "__main__":
    main()
