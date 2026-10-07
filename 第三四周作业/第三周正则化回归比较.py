"""第三周作业 4：在 Hitters 数据上比较 Ridge、Lasso 与 Elastic Net。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    Ridge,
    RidgeCV,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Hitters.csv"
RESULT_PATH = BASE_DIR / "第三周模型比较结果.csv"
PLOT_PATH = BASE_DIR / "第三周系数路径.png"
SEED = 2026
ALPHAS = np.logspace(-3, 4, 80)
L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """读取 ISLR Hitters 数据并剔除 Salary 缺失的记录。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {DATA_PATH}。请按作业目录说明准备 Hitters.csv。"
        )
    data = pd.read_csv(DATA_PATH).drop(columns=["rownames"], errors="ignore")
    if "Salary" not in data:
        raise ValueError("Hitters.csv 中缺少目标列 Salary。")
    data = data.dropna(subset=["Salary"]).reset_index(drop=True)
    return data.drop(columns="Salary"), data["Salary"].astype(float)


def select_one_se_model(
    model: ElasticNetCV, x_train: np.ndarray, y_train: pd.Series
) -> tuple[float, float, float, float, ElasticNet]:
    """在所有 ElasticNetCV 网格点中，按 1-SE 选择非零系数最少的模型。"""
    mse_path = np.asarray(model.mse_path_)
    if mse_path.ndim == 2:
        mse_path = mse_path[np.newaxis, :, :]
    if mse_path.ndim != 3:
        raise ValueError(f"无法识别 ElasticNetCV.mse_path_ 形状：{mse_path.shape}")

    ratios = np.atleast_1d(np.asarray(model.l1_ratio, dtype=float))
    alpha_grid = np.asarray(model.alphas_, dtype=float)
    if alpha_grid.ndim == 1:
        alpha_grid = np.tile(alpha_grid, (len(ratios), 1))
    mean_mse = mse_path.mean(axis=2)
    best_ratio_index, best_alpha_index = np.unravel_index(
        int(np.argmin(mean_mse)), mean_mse.shape
    )
    best_fold_mse = mse_path[best_ratio_index, best_alpha_index]
    standard_error = float(best_fold_mse.std(ddof=1) / np.sqrt(len(best_fold_mse)))
    threshold = float(mean_mse[best_ratio_index, best_alpha_index] + standard_error)

    candidates = []
    for ratio_index, alpha_index in np.argwhere(mean_mse <= threshold):
        alpha = float(alpha_grid[ratio_index, alpha_index])
        ratio = float(ratios[ratio_index])
        fitted = ElasticNet(
            alpha=alpha,
            l1_ratio=ratio,
            max_iter=100_000,
            tol=1e-5,
            random_state=SEED,
        ).fit(x_train, y_train)
        nonzero = int(np.count_nonzero(np.abs(fitted.coef_) > 1e-6))
        # 先选最稀疏解；并列时优先选择更强的 L1 比例和更大的惩罚强度。
        candidates.append((nonzero, -ratio, -alpha, alpha, ratio, fitted))
    _, _, _, selected_alpha, selected_ratio, selected_model = min(candidates)
    return (
        selected_alpha,
        selected_ratio,
        float(mean_mse[best_ratio_index, best_alpha_index]),
        standard_error,
        selected_model,
    )


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """数值特征插补；分类特征独热编码；最终统一 Z-score。"""
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def save_coefficient_paths(
    x_train: np.ndarray, y_train: pd.Series, feature_names: np.ndarray
) -> None:
    """繪製三种惩罚模型的系数路径。"""
    path_alphas = np.logspace(-2, 4, 50)
    models = [
        ("Ridge", Ridge),
        ("Lasso", Lasso),
        ("Elastic Net (l1_ratio=0.5)", ElasticNet),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for axis, (label, estimator) in zip(axes, models):
        coefficient_path = []
        for alpha in path_alphas:
            if estimator is Ridge:
                fit = estimator(alpha=float(alpha))
            elif estimator is Lasso:
                fit = estimator(alpha=float(alpha), max_iter=100_000, tol=1e-5)
            else:
                fit = estimator(
                    alpha=float(alpha), l1_ratio=0.5, max_iter=100_000, tol=1e-5
                )
            fit.fit(x_train, y_train)
            coefficient_path.append(fit.coef_)
        coefficient_path = np.asarray(coefficient_path)
        for index, name in enumerate(feature_names):
            axis.plot(path_alphas, coefficient_path[:, index], linewidth=1.1, label=name)
        axis.set_xscale("log")
        axis.set_title(label)
        axis.set_xlabel("alpha (log scale)")
        axis.axhline(0, color="black", linewidth=0.6)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Standardized coefficient")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Hitters: coefficient paths on the training set")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=SEED
    )

    categorical = [name for name in ["League", "Division", "NewLeague"] if name in x_train]
    numeric = [column for column in x_train.columns if column not in categorical]
    preprocessor = make_preprocessor(numeric, categorical)
    x_train_encoded = preprocessor.fit_transform(x_train)
    x_test_encoded = preprocessor.transform(x_test)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_encoded)
    x_test_scaled = scaler.transform(x_test_encoded)
    feature_names = preprocessor.get_feature_names_out()

    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    ridge = RidgeCV(alphas=ALPHAS, cv=cv, scoring="neg_mean_squared_error")
    lasso = LassoCV(
        alphas=ALPHAS,
        cv=cv,
        max_iter=100_000,
        tol=1e-5,
        n_jobs=-1,
        random_state=SEED,
    )
    elastic = ElasticNetCV(
        alphas=ALPHAS,
        l1_ratio=L1_RATIOS,
        cv=cv,
        max_iter=100_000,
        tol=1e-5,
        n_jobs=-1,
        random_state=SEED,
    )

    fitted: list[tuple[str, object, float, float | None]] = []
    for name, model in [("RidgeCV", ridge), ("LassoCV", lasso), ("ElasticNetCV", elastic)]:
        model.fit(x_train_scaled, y_train)
        fitted.append((name, model, float(model.alpha_), getattr(model, "l1_ratio_", None)))

    rows = []
    for name, model, alpha, l1_ratio in fitted:
        prediction = model.predict(x_test_scaled)
        rmse = float(np.sqrt(mean_squared_error(y_test, prediction)))
        nonzero = int(np.count_nonzero(np.abs(model.coef_) > 1e-6))
        rows.append(
            {
                "model": name,
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "test_rmse": rmse,
                "nonzero_features": nonzero,
            }
        )

    (
        elastic_one_se_alpha,
        elastic_one_se_ratio,
        best_cv_mse,
        se,
        elastic_one_se,
    ) = select_one_se_model(elastic, x_train_scaled, y_train)
    one_se_prediction = elastic_one_se.predict(x_test_scaled)
    rows.append(
        {
            "model": "ElasticNet_1SE",
            "alpha": elastic_one_se_alpha,
            "l1_ratio": elastic_one_se_ratio,
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, one_se_prediction))),
            "nonzero_features": int(np.count_nonzero(np.abs(elastic_one_se.coef_) > 1e-6)),
        }
    )

    results = pd.DataFrame(rows)
    results.insert(0, "n_train", len(y_train))
    results.insert(1, "n_test", len(y_test))
    results.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    save_coefficient_paths(x_train_scaled, y_train, feature_names)

    print(f"有效样本：{len(y)}（Salary 缺失行已删除）")
    print(f"训练/测试：{len(y_train)}/{len(y_test)}；随机种子：{SEED}")
    print("特征均仅使用训练集拟合预处理器并做 Z-score 标准化。")
    print("10 折 CV 与独立测试集结果：")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "Elastic Net 1-SE："
        f"alpha={elastic_one_se_alpha:.6g}, l1_ratio={elastic_one_se_ratio:.3f}, "
        f"最小 CV MSE={best_cv_mse:.4f}, SE={se:.4f}"
    )
    print(f"结果表：{RESULT_PATH}")
    print(f"系数路径图：{PLOT_PATH}")


if __name__ == "__main__":
    main()
