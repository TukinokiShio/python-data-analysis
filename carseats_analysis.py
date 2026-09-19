r"""第二周作业：Carseats 多元线性回归与多重共线性诊断。

运行：
    .venv\Scripts\python.exe carseats_analysis.py

脚本优先读取同目录下的 Carseats.csv；若文件不存在，则从公开数据地址读取。
"""

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Carseats.csv"
DATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Carseats.csv"
REQUIRED_COLUMNS = {
    "Sales",
    "Price",
    "Income",
    "Advertising",
    "ShelveLoc",
}


def load_carseats() -> tuple[pd.DataFrame, str]:
    """读取 Carseats 数据，并返回数据与数据来源说明。"""
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
        source = str(DATA_PATH)
    else:
        with urlopen(DATA_URL, timeout=30) as response:
            data = pd.read_csv(response)
        source = DATA_URL

    # Rdatasets 版本带有 R 行名列，不参与建模。
    data = data.drop(columns=["rownames"], errors="ignore")
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"数据缺少必要列：{sorted(missing)}")
    if data["ShelveLoc"].isna().any():
        raise ValueError("ShelveLoc 存在缺失值，无法稳定解释分类变量系数。")

    # 显式规定顺序，确保 Treatment(reference="Bad") 的基准组就是 Bad。
    data["ShelveLoc"] = pd.Categorical(
        data["ShelveLoc"], categories=["Bad", "Medium", "Good"]
    )
    return data, source


def calculate_vif(model: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    """对设计矩阵中的解释变量计算 VIF，不把截距列作为诊断对象。"""
    design = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    predictors = design.drop(columns=["Intercept"], errors="ignore")
    rows = []
    for index, name in enumerate(predictors.columns):
        rows.append(
            {
                "variable": name,
                "VIF": variance_inflation_factor(predictors.to_numpy(), index),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    data, source = load_carseats()

    # 题目要求的模型：Sales ~ Price + Income + Advertising + ShelveLoc。
    model = smf.ols(
        'Sales ~ Price + Income + Advertising + C(ShelveLoc, Treatment(reference="Bad"))',
        data=data,
    ).fit()
    vif = calculate_vif(model)

    coefficients = model.summary2().tables[1].reset_index().rename(
        columns={"index": "term"}
    )
    coefficients.to_csv(BASE_DIR / "carseats_coefficients.csv", index=False)
    vif.to_csv(BASE_DIR / "carseats_vif.csv", index=False)
    (BASE_DIR / "carseats_model_summary.txt").write_text(
        model.summary().as_text(), encoding="utf-8"
    )

    print(f"数据来源：{source}")
    print(f"样本量：{len(data)}")
    print("\n=== 模型拟合报告 ===")
    print(model.summary())
    print("\n=== VIF（不含截距） ===")
    print(vif.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== 关键系数 ===")
    print(coefficients.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
