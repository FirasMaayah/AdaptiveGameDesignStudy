import pandas as pd
from statsmodels.stats.contingency_tables import cochrans_q
from scipy.stats import friedmanchisquare
import numpy as np
from statsmodels.stats.anova import AnovaRM
from scipy.stats import wilcoxon
import pingouin as pg
import statsmodels.formula.api as smf
import math
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttost_paired

# ---------------------------------------------------
# ------ Aggregate  Mode Scores 
# ---------------------------------------------------
def aggregate_mode_scores(df, mode_prefix, questions):
    """
    Compute the mean score per participant for a given mode.
    
    Columns are constructed as "{mode_prefix}_{question}" for each question.
    Returns a pandas Series of row-wise means.
    """
    cols = [f"{mode_prefix}_{q}" for q in questions]
    return df[cols].mean(axis=1)

# ---------------------------------------------------
# ------ Print Test Results
# ---------------------------------------------------
def print_test_result(result, only_significant: bool = False):
    """
    Print a formatted statistical test result dictionary.
    
    Expects a dictionary containing test statistics, p-value, and effect size.
    If only_significant is True, non-significant results are not printed.
    """
    if only_significant and not result["significant"]:
        return
    print()
    print("========================================================")
    print(f"Test: {result['test']}")
    #print(f"Variable: {result['label']}")
    print(f"Columns: {result['columns']}")
    print(f"N: {result['n']}")
    print(f"Statistic: {result['statistic']}")
    if "other" in result and result["other"] is not None:
        print(result["other"])
    print(f"P-value: {result['p_value']}")
    print(f"Alpha: {result['alpha']}")
    
    if result["significant"]:
        print("Result: Significant difference.")
    else:
        print("Result: No significant difference.")
    print()
    print(f"Effect Size test: {result['effect_size_test']}")
    print(f"Effect Size: {result['effect_size']}")
    print(f"Effect Size is {result['effect_size_text']}")
    print("--------------------------------------------------------")
    print()

# ---------------------------------------------------
# ------ Cochran Q Test 
# ---------------------------------------------------
def cochran_q_test(df, mode_columns, alpha=0.05, column_label=None, dropna=True):
    """
    Run Cochran's Q test on matched binary data across modes.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the columns to test.
    mode_columns : list[str]
        Column names for the matched binary outcomes, one per mode.
        Example: ['Action_Survived', 'Stealth_Survived', 'Adaptive_Survived']
    alpha : float, optional
        Significance threshold.
    column_label : str or None, optional
        Friendly name for printing/reporting.
    dropna : bool, optional
        Whether to drop rows with missing values in any test column.
    
    Returns
    -------
    dict
        Dictionary containing test results and metadata.
    """

    test_df = df[mode_columns].copy()

    if dropna:
        test_df = test_df.dropna()

    if test_df.empty:
        raise ValueError("No valid rows remain after dropping missing values.")

    # Ensure values are binary 0/1
    unique_values = set(test_df.stack().unique())
    if not unique_values.issubset({0, 1, True, False}):
        raise ValueError(
            f"Cochran's Q requires binary matched data (0/1 or True/False). "
            f"Found values: {sorted(unique_values)}"
        )

    # Convert booleans to ints if needed
    test_df = test_df.astype(int)

    result = cochrans_q(test_df.to_numpy())

    output = {
        "test": "Cochran's Q",
        "columns": mode_columns,
        "n": len(test_df),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "alpha": alpha,
        "significant": bool(result.pvalue < alpha),
        "effect_size_test": None,
        "effect_size": None,
        "effect_size_text": None,
    }
    
    print_test_result(output)

    return output

# ---------------------------------------------------
# ------ Standard Deviation
# ---------------------------------------------------
def mode_sd(df: pd.DataFrame, columns : list[str]):
    """
    Compute standard deviation per row across specified columns.
    """
    return df[columns].std(axis=1)

# ---------------------------------------------------
# ------ Friedman Chi Square
# ---------------------------------------------------
def friedman_test_modes(
    df: pd.DataFrame,
    columns_by_mode: dict[str, str],
    alpha = 0.05,
    only_significant = False,
):
    """
    Perform a Friedman test across the three gameplay modes for a given metric.
    
    columns_by_mode should map each mode name to its corresponding column.
    
    Example:
        {
            "Stealth": "Stealth_Enjoyment",
            "Action": "Action_Enjoyment",
            "Adaptive": "Adaptive_Enjoyment",
        }
    
    Returns a dictionary containing test results, sample size, significance,
    and Kendall's W effect size.
    """
    # Extract columns in consistent order
    mode_order = ["Stealth", "Action", "Adaptive"]
    data = []

    for mode in mode_order:
        col = columns_by_mode[mode]
        data.append(pd.to_numeric(df[col], errors="coerce"))

    # Combine and drop rows with any NaNs
    combined = pd.concat(data, axis=1).dropna()
    combined.columns = mode_order

    if len(combined) < 2:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n": len(combined),
        }

    stat, p = friedmanchisquare(
        combined["Stealth"],
        combined["Action"],
        combined["Adaptive"],
    )
    
    effect_size, effect_size_text = kendall_w(stat, len(combined))
    
    results = {
        "test": "Friedman Chi Square",
        "columns": columns_by_mode.values(),
        "n": len(combined),
        "statistic": stat,
        "p_value": p,
        "alpha": alpha,
        "significant": p < alpha,
        "effect_size_test": "Kendall's W",
        "effect_size": f"w = {effect_size}",
        "effect_size_text": effect_size_text,
    }
    
    print_test_result(results, only_significant)

    return results

# ---------------------------------------------------
# ------ repeated measures ANOVA
# ---------------------------------------------------
def repeated_measures_anova(
    df: pd.DataFrame,
    columns_by_mode: dict[str, str],
    subject_col: str | None = None,
    alpha: float = 0.05,
    only_significant: bool = False,
):
    """
    Perform a repeated-measures ANOVA across the three gameplay modes.
    
    columns_by_mode should map each mode name to its corresponding outcome column.
    If subject_col is not provided, row index order is treated as participant identity.
    
    Example:
        {
            "Stealth": "Stealth_Enjoyment",
            "Action": "Action_Enjoyment",
            "Adaptive": "Adaptive_Enjoyment",
        }
    
    Returns a dictionary containing ANOVA results, sample size, significance,
    partial eta squared, the ANOVA table, and the long-format dataframe.
    """
    mode_order = ["Stealth", "Action", "Adaptive"]

    if subject_col is None:
        working_df = df.copy().reset_index(drop=True)
        working_df["_subject_id"] = np.arange(len(working_df))
        subject_col = "_subject_id"
    else:
        working_df = df.copy()

    keep_cols = [subject_col] + [columns_by_mode[mode] for mode in mode_order]
    wide_df = working_df[keep_cols].copy()

    for mode in mode_order:
        wide_df[columns_by_mode[mode]] = pd.to_numeric(
            wide_df[columns_by_mode[mode]],
            errors="coerce"
        )

    # complete cases only
    wide_df = wide_df.dropna()

    if len(wide_df) < 2:
        return {
            "test": "Repeated Measures ANOVA",
            "columns": list(columns_by_mode.values()),
            "n": len(wide_df),
            "statistic": np.nan,
            "p_value": np.nan,
            "alpha": alpha,
            "significant": False,
            "effect_size_test": "partial eta squared",
            "effect_size": "ηp² = nan",
            "effect_size_text": "unknown",
            "df_effect": np.nan,
            "df_error": np.nan,
            "anova_table": None,
            "long_df": None,
        }

    long_df = wide_df.melt(
        id_vars=[subject_col],
        value_vars=[columns_by_mode[mode] for mode in mode_order],
        var_name="ModeRaw",
        value_name="Value"
    )

    reverse_map = {columns_by_mode[mode]: mode for mode in mode_order}
    long_df["Mode"] = long_df["ModeRaw"].map(reverse_map)

    anova = AnovaRM(
        data=long_df,
        depvar="Value",
        subject=subject_col,
        within=["Mode"]
    ).fit()

    anova_table = anova.anova_table.copy()

    # statsmodels AnovaRM typically uses "Mode" as the row index
    row = anova_table.loc["Mode"]

    f_value = float(row["F Value"])
    p_value = float(row["Pr > F"])
    df_effect = float(row["Num DF"])
    df_error = float(row["Den DF"])

    effect_size, effect_size_text = partial_eta_squared(
        f_value,
        df_effect,
        df_error
    )

    results = {
        "test": "Repeated Measures ANOVA",
        "columns": list(columns_by_mode.values()),
        "n": len(wide_df),
        "statistic": f_value,
        "p_value": p_value,
        "alpha": alpha,
        "significant": p_value < alpha,
        "effect_size_test": "partial eta squared",
        "effect_size": f"ηp² = {effect_size}",
        "effect_size_text": effect_size_text,
        "df_effect": df_effect,
        "df_error": df_error,
        "anova_table": anova_table,
        "long_df": long_df,
    }

    print_test_result(results, only_significant)

    return results

# ---------------------------------------------------
# ------ Wilcoxon Signed-rank test
# ---------------------------------------------------
def wilcoxon_test_pair(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    only_significant = False,
):
    """
    Perform a Wilcoxon signed-rank test between two paired columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    col_a : str
        First paired column.
    col_b : str
        Second paired column.
    alternative : str
        Alternative hypothesis: "two-sided", "greater", or "less".
    
    Returns
    -------
    dict
        Dictionary containing test results, sample size, significance,
        alternative hypothesis, and effect size.
    """
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")

    # Pair and drop missing values
    paired = pd.concat([a, b], axis=1).dropna()

    if len(paired) < 1:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n": 0,
        }

    stat, p = wilcoxon(
        paired.iloc[:, 0],
        paired.iloc[:, 1],
        alternative=alternative,
    )
    
    effect_size, effect_size_text = rank_biserial(stat, len(paired))
    
    results = {
        "test": "Wilcoxon signed-rank test",
        "columns": [col_a, col_b],
        "n": len(paired),
        "statistic": stat,
        "p_value": p,
        "alpha": alpha,
        "significant": p < alpha,
        "other": f"alternative: {alternative}",
        "effect_size_test": "rank-biserial correlation",
        "effect_size": f"r = {effect_size}",
        "effect_size_text": effect_size_text,
    }
    
    print_test_result(results, only_significant)
    
    return results


def wilcoxon_equivalence_test_pair(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    lower_bound: float,
    upper_bound: float,
    alpha: float = 0.05,
    only_significant: bool = False,
    zero_method: str = "wilcox",
):
    """
    Perform a Wilcoxon signed-rank equivalence test (paired nonparametric TOST)
    between two paired columns.

    Equivalence is tested on the paired difference:
        d = col_a - col_b

    Null hypotheses
    ---------------
    H01: median(d) <= lower_bound
    H02: median(d) >= upper_bound

    Alternative hypotheses
    ----------------------
    HA1: median(d) > lower_bound
    HA2: median(d) < upper_bound

    Equivalence is concluded only if both one-sided tests are significant.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    col_a : str
        First paired column.
    col_b : str
        Second paired column.
    lower_bound : float
        Lower equivalence bound on the paired difference scale.
    upper_bound : float
        Upper equivalence bound on the paired difference scale.
    alpha : float
        Significance level.
    only_significant : bool
        If True, only print results when equivalence is concluded.
    zero_method : str
        Passed to scipy.stats.wilcoxon. Common options:
        "wilcox", "pratt", "zsplit"

    Returns
    -------
    dict
        {
            "test": str,
            "columns": [str, str],
            "n": int,
            "lower_bound": float,
            "upper_bound": float,
            "p_lower": float,
            "p_upper": float,
            "stat_lower": float,
            "stat_upper": float,
            "alpha": float,
            "equivalent": bool,
            "median_difference": float
        }
    """
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound.")

    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")

    paired = pd.concat([a, b], axis=1).dropna()

    if len(paired) < 1:
        return {
            "test": "Wilcoxon signed-rank equivalence test",
            "columns": [col_a, col_b],
            "n": 0,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "stat_lower": np.nan,
            "p_lower": np.nan,
            "stat_upper": np.nan,
            "p_upper": np.nan,
            "alpha": alpha,
            "equivalent": False,
            "median_difference": np.nan,
        }
    
    # Compute paired differences on the same scale as the equivalence bounds
    diff = paired.iloc[:, 0] - paired.iloc[:, 1]

    # First one-sided test:
    # H01: median(diff) <= lower_bound
    # HA1: median(diff) > lower_bound
    stat_lower, p_lower = wilcoxon(
        diff - lower_bound,
        alternative="greater",
        zero_method=zero_method,
    )

    # Second one-sided test:
    # H02: median(diff) >= upper_bound
    # HA2: median(diff) < upper_bound
    stat_upper, p_upper = wilcoxon(
        diff - upper_bound,
        alternative="less",
        zero_method=zero_method,
    )

    equivalent = (p_lower < alpha) and (p_upper < alpha)

    results = {
        "test": "Wilcoxon signed-rank equivalence test",
        "columns": [col_a, col_b],
        "n": len(paired),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "stat_lower": stat_lower,
        "p_lower": p_lower,
        "stat_upper": stat_upper,
        "p_upper": p_upper,
        "alpha": alpha,
        "equivalent": equivalent,
        "median_difference": float(np.median(diff)),
        "other": f"Equivalence Bounds: [{lower_bound}, {upper_bound}], median_difference: {float(np.median(diff))}, p_values: [{p_lower:6f}, {p_upper:6f}]",
        "interpretation": (
            f"Equivalent within [{lower_bound}, {upper_bound}]"
            if equivalent
            else f"Not equivalent within [{lower_bound}, {upper_bound}]"
            ),
        "significant": equivalent,
        "p_value": max(p_lower, p_upper),
        "statistic": [stat_lower, stat_upper],
        "effect_size_test": None,
        "effect_size": None,
        "effect_size_text": None,
    }

    print_test_result(results, only_significant)

    return results

# ---------------------------------------------------
# ------ T-test
# ---------------------------------------------------
def ttest_pair(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    only_significant = False,
):
    """
    Perform a paired t-test between two paired columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    col_a : str
        First paired column.
    col_b : str
        Second paired column.
    alternative : str
        Alternative hypothesis: "two-sided", "greater", or "less".
    alpha : float
        Significance level.
    only_significant : bool
        If True, only print significant results.
    
    Returns
    -------
    dict
        Dictionary containing test results, sample size, significance,
        alternative hypothesis, and Cohen's d.
    """
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")

    # Pair and drop missing values
    paired = pd.concat([a, b], axis=1).dropna()

    if len(paired) < 2:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "n": 0,
        }

    stat, p = ttest_rel(
        paired.iloc[:, 0],
        paired.iloc[:, 1],
        alternative=alternative,
    )
    
    effect_size, effect_size_text = cohens_d(
        paired.iloc[:, 0],
        paired.iloc[:, 1]
    )

    results = {
        "test": "Paired t-test",
        "columns": [col_a, col_b],
        "n": len(paired),
        "statistic": stat,
        "p_value": p,
        "alpha": alpha,
        "significant": p < alpha,
        "other": f"alternative: {alternative}",
        "effect_size_test": "Cohen's d",
        "effect_size": f"d = {effect_size}",
        "effect_size_text": effect_size_text,
    }
    
    print_test_result(results, only_significant)
    
    return results

# ---------------------------------------------------
# ------ Intraclass Correlation Coefficient (ICC)
# ---------------------------------------------------
def icc_modes(
    df: pd.DataFrame,
    mode: str,
    metrics: list[str],
    icc_type: str = "ICC(C,1)",
):
    """
    Compute ICC for one mode across multiple metrics/questions.

    Parameters
    ----------
    df : pd.DataFrame
        Wide dataframe with one row per participant.
    mode : str
        One of: "Stealth", "Action", "Adaptive"
    metrics : list[str]
        Base metric names, e.g.
        ["Enjoyment", "Immersion", "Challenge", "Frustration", "Competence", "Autonomy"]

        Assumes columns are named:
            {mode}_{metric}
    icc_type : str
        Pingouin ICC type, e.g.:
        "ICC(1,1)", "ICC(A,1)", "ICC(C,1)", "ICC(C,k)"

    Returns
    -------
    dict
        {
            "mode": str,
            "icc_type": str,
            "icc": float,
            "ci95_low": float,
            "ci95_high": float,
            "f": float,
            "df1": float,
            "df2": float,
            "p_value": float,
            "n": int
        }
    """
    cols = [f"{mode}_{metric}" for metric in metrics]

    wide = df[cols].copy()
    wide.columns = metrics
    wide = wide.apply(pd.to_numeric, errors="coerce").dropna()

    if len(wide) < 2:
        return {
            "mode": mode,
            "icc_type": icc_type,
            "icc": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "f": np.nan,
            "df1": np.nan,
            "df2": np.nan,
            "p_value": np.nan,
            "n": len(wide),
        }

    wide = wide.reset_index(drop=True)
    wide["Participant"] = np.arange(len(wide))

    long_df = wide.melt(
        id_vars="Participant",
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )

    icc_table = pg.intraclass_corr(
        data=long_df,
        targets="Participant",
        raters="Metric",
        ratings="Score",
    )

    row = icc_table.loc[icc_table["Type"] == icc_type]
    if row.empty:
        raise ValueError(
            f"ICC type '{icc_type}' not found. "
            f"Available types: {icc_table['Type'].tolist()}"
        )

    row = row.iloc[0]

    ci_col = "CI95" if "CI95" in row.index else "CI95%"
    ci = row[ci_col]

    return {
        "mode": mode,
        "icc_type": row["Type"],
        "icc": float(row["ICC"]),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "f": float(row["F"]),
        "df1": float(row["df1"]),
        "df2": float(row["df2"]),
        "p_value": float(row["pval"]),
        "n": len(wide),
    }

def icc_modes_multiple(
    df: pd.DataFrame,
    metrics: list[str],
    modes: list[str] = ["Stealth","Action","Adaptive"],
    icc_type: str = "ICC(C,1)",
):
    """
    Compute ICC consistency separately for each gameplay mode.
    
    For each mode, ICC is calculated across the provided metrics/questions.
    Assumes columns follow the format: {Mode}_{Metric}.
    
    Returns a dataframe with one ICC result per mode.
    """
    rows = []

    for mode in modes:
        result = icc_modes(
            df,
            mode,
            metrics,
            icc_type=icc_type,
        )
        result["mode"] = mode
        rows.append(result)

    return pd.DataFrame(rows)

# ---------------------------------------------------
# ------ Mixed Model
# ---------------------------------------------------
def run_mode_mixed_model(
    df,
    preference_col,
    mode_outcome_cols,
    ref:str,
    dropna=True,
    reml=True
):
    """
    Reshape wide data into long format and fit a mixed-effects model:
        outcome ~ mode * preference + (1 | subject)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe in wide format, with one row per subject.
    preference_col : str
        Column name holding the subject preference score.
        Expected interpretation:
            0 = leans Action
            1 = leans Stealth
    mode_outcome_cols : dict
        Dictionary mapping mode names to outcome column names, e.g.:
            {
                "Stealth": "stealth_mean",
                "Action": "action_mean",
                "Adaptive": "adaptive_mean"
            }
    ref : str, either "Action" or "Stealth" determines the reference category in the model.
    dropna : bool, default=True
        If True, drop rows with missing subject, preference, or outcome values.
    reml : bool, default=True
        Whether to fit the model using REML.

    Returns
    -------
    dict
        Dictionary containing the long dataframe, model, fit object,
        summary, coefficient table, and mode order.
    """

    df = df.copy()
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    df["subject"] = range(len(df))
    subject_col = "subject"

    if preference_col not in df.columns:
        raise ValueError(f"preference_col '{preference_col}' not found in dataframe")

    if not isinstance(mode_outcome_cols, dict) or len(mode_outcome_cols) < 2:
        raise ValueError("mode_outcome_cols must be a dict with at least two modes")

    missing_mode_cols = [
        col_name for col_name in mode_outcome_cols.values()
        if col_name not in df.columns
    ]
    if missing_mode_cols:
        raise ValueError(f"These outcome columns were not found: {missing_mode_cols}")
        
    if ref == "Stealth":
        mode_order = ("Stealth", "Action", "Adaptive")
    elif ref == "Action":
        mode_order = ("Action", "Stealth", "Adaptive")
    else:
        raise ValueError(f"Reference value '{ref}' not supported")
    
    # Build long-format dataframe
    long_parts = []
    for mode_name, outcome_col in mode_outcome_cols.items():
        temp = df[[subject_col, preference_col, outcome_col]].copy()
        temp = temp.rename(columns={outcome_col: "outcome"})
        temp["mode"] = mode_name
        long_parts.append(temp)

    long_df = pd.concat(long_parts, ignore_index=True)

    # Standardize column names used by the model formula
    long_df = long_df.rename(columns={
        subject_col: "subject",
        preference_col: "preference"
    })

    # Drop missing values if requested
    if dropna:
        long_df = long_df.dropna(subset=["subject", "preference", "outcome", "mode"]).copy()

    if long_df.empty:
        raise ValueError("No data left after reshaping / dropping missing values")

    # Make mode categorical with a controlled reference order
    present_modes = [m for m in mode_order if m in long_df["mode"].unique()]
    other_modes = [m for m in long_df["mode"].unique() if m not in present_modes]
    final_mode_order = present_modes + other_modes

    long_df["mode"] = pd.Categorical(
        long_df["mode"],
        categories=final_mode_order,
        ordered=False
    )
    long_df["outcome"] = pd.to_numeric(long_df["outcome"], errors="coerce")

    # Fit mixed model
    model = smf.mixedlm(
        "outcome ~ mode * preference",
        data=long_df,
        groups=long_df["subject"]
    )
    fit = model.fit(reml=reml)

    # Build a clean coefficient table
    coef_table = pd.DataFrame({
        "term": fit.params.index,
        "coef": fit.params.values,
        "std_err": fit.bse.values,
        "z": fit.tvalues.values,
        "p_value": fit.pvalues.values
    })

    # Confidence intervals if available
    conf_int = fit.conf_int()
    coef_table["ci_lower"] = conf_int[0].values
    coef_table["ci_upper"] = conf_int[1].values

    return {
        "long_df": long_df,
        "model": model,
        "fit": fit,
        "summary": fit.summary(),
        "coef_table": coef_table,
        "mode_order": mode_order,
    }

# ---------------------------------------------------
# ------ Effect Sizes
# ---------------------------------------------------
def kendall_w(stat, n):
    """
    Compute Kendall's W from Friedman test statistic for three modes.
    """
    w = (stat) / (n*2)
    if w >= 0.5:
        text = "large"
    elif w >= 0.3:
        text = "medium"
    else:
        text = "small"
    return w, text

def cohens_d(a, b):
    diff = a - b
    d = diff.mean() / diff.std(ddof=1)
    if abs(d) >= 0.8:
        text = "large"
    elif abs(d) >= 0.5:
        text = "medium"
    else:
        text = "small"
    return d, text

def partial_eta_squared(f_value, df_effect, df_error):
    """
    Compute partial eta squared (ηp²) for ANOVA.

    Parameters
    ----------
    f_value : float
        F statistic from ANOVA.
    df_effect : float
        Degrees of freedom for the effect (numerator df).
    df_error : float
        Degrees of freedom for the error (denominator df).

    Returns
    -------
    (eta_p2, interpretation)
    """
    eta_p2 = (f_value * df_effect) / ((f_value * df_effect) + df_error)

    if eta_p2 >= 0.14:
        text = "large"
    elif eta_p2 >= 0.06:
        text = "medium"
    elif eta_p2 >= 0.01:
        text = "small"
    else:
        text = "negligible"

    return eta_p2, text

def rank_biserial(stat, n):
    r = 1 - (2 * stat) / (n * (n + 1))
    if abs(r) >= 0.5:
        text = "large"
    elif abs(r) >= 0.3:
        text = "medium"
    else:
        text = "small"
    return r, text

# ---------------------------------------------------
# ------ Corrections
# ---------------------------------------------------
def cal_holm(pdict):
    """
    Apply Holm correction to a dictionary of p-values and print corrected values.
    """
    metrics = list(pdict.keys())
    pvals = list(pdict.values())

    holm = multipletests(pvals, method="holm")
    corrected_pvals = holm[1]

    print("Holm:")
    for metric, pval in zip(metrics, corrected_pvals):
        print(f"{metric} p-value: {pval}")
    print()

# ---------------------------------------------------
# ------ TOST
# ---------------------------------------------------
def tost_pair(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    low: float,
    high: float,
    alpha: float = 0.05,
    only_significant: bool = False,
):
    """
    Perform paired TOST equivalence test between two paired columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    col_a : str
        First column.
    col_b : str
        Second column.
    low : float
        Lower equivalence bound for (col_a - col_b).
    high : float
        Upper equivalence bound for (col_a - col_b).
    alpha : float
        Significance level.
    only_significant : bool
        Whether to print only significant results.

    Returns
    -------
    dict
        Dictionary containing TOST results, equivalence bounds,
        mean difference, equivalence decision, and Cohen's d.
    """
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")

    # Pair and drop missing values
    paired = pd.concat([a, b], axis=1).dropna()

    if len(paired) < 2:
        return {
            "statistic_low": np.nan,
            "p_value_low": np.nan,
            "statistic_high": np.nan,
            "p_value_high": np.nan,
            "p_value": np.nan,
            "n": 0,
        }

    x = paired.iloc[:, 0]
    y = paired.iloc[:, 1]

    # TOST: H0 is non-equivalence, H1 is equivalence within [low, high]
    p_value, (stat_low, p_value_low, df_low), (stat_high, p_value_high, df_high) = ttost_paired(
        x, y, low, high
    )

    # Mean difference for interpretation
    mean_diff = (x - y).mean()

    effect_size, effect_size_text = cohens_d(x, y)

    results = {
        "test": "Paired TOST equivalence test",
        "columns": [col_a, col_b],
        "n": len(paired),
        "equivalence_bounds": [low, high],
        "mean_difference": mean_diff,
        "statistic_low": stat_low,
        "p_value_low": p_value_low,
        "statistic_high": stat_high,
        "p_value_high": p_value_high,
        "statistic": 0, 
        "p_value": p_value,  # overall TOST p-value
        "alpha": alpha,
        "significant": p_value < alpha,
        "equivalent": p_value < alpha,
        "other": f"Equivalence Bounds: [{low}, {high}], Mean Difference: {mean_diff}",
        "effect_size_test": "Cohen's d",
        "effect_size": f"d = {effect_size}",
        "effect_size_text": effect_size_text,
    }

    print_test_result(results, only_significant)

    return results