"""
Run statistical analyses and generate figures for the study dataset.

Includes preprocessing, statistical tests, mixed models, and visualization outputs.
"""

import pandas as pd
from pathlib import Path
import numpy as np
import Analysis_Functions as analysis
import Plot_Functions as plot

# =========================
# Global settings
# =========================
ALPHA = 0.05

# =========================================================
# FILE PATHS
# =========================================================
DATA_CSV = Path("Adaptive_Game_Design_Study_Data.csv")

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(DATA_CSV)

# =========================================================
# Data Processing
# =========================================================
for col in plot.mode_columns("Rank", as_list = True):
    df.loc[df[col] == "Highest enjoyment", col] = 3
    df.loc[df[col] == "Middle enjoyment", col] = 2
    df.loc[df[col] == "Lowest enjoyment", col] = 1
    
for col in plot.mode_columns("Frustrated", as_list = True):
    df[col] *= -1
likert_cols = ["Absorbed","Skilled","Enjoy","Challenge","Frustrated","Again"]

total_raw_w_cols = []
for mode in ["Stealth","Action","Adaptive"]:
    df[f"{mode}_LikertMean"] = analysis.aggregate_mode_scores(df, mode, likert_cols)
    SD_cols = []
    for col in likert_cols:
        SD_cols.append(f"{mode}_{col}")
    df[f"{mode}_LikertSD"] = analysis.mode_sd(df, SD_cols)
    
    raw_w_cols = []
    for i in range(1,6):
        raw_w_cols.append(f"{mode}_Stealth_W_{i}")
        total_raw_w_cols.append(f"{mode}_Stealth_W_{i}")
    df[f"{mode}_Stealth_W_SD"] = analysis.mode_sd(df, raw_w_cols)
    
df["Stealth_Tendency"] = df[total_raw_w_cols].mean(axis=1)

# =========================================================
# Ranking Analysis
# =========================================================
print("\n=====================================")
print("========== Ranking Analysis =========")
print("=====================================")
bool_col = plot.mode_columns("Not_Ranked_Lowest", as_list = True)
rank_col = plot.mode_columns("Rank", as_list = True)
for i in range(3):
    df[bool_col[i]] = 0
    df.loc[df[rank_col[i]] >=2, bool_col[i]] = 1
    print(f'{bool_col[i]} = {np.sum(df[bool_col[i]])}')
print()
analysis.cochran_q_test(df, 
                   plot.mode_columns("Not_Ranked_Lowest", as_list = True), 
                   alpha=ALPHA)

analysis.cochran_q_test(df, 
                   ["Stealth_Not_Ranked_Lowest","Adaptive_Not_Ranked_Lowest"], 
                   alpha=ALPHA)

analysis.cochran_q_test(df, 
                   ["Action_Not_Ranked_Lowest","Adaptive_Not_Ranked_Lowest"], 
                   alpha=ALPHA)

analysis.friedman_test_modes(df, plot.mode_columns("Rank"), ALPHA)

# =========================================================
# Likert Variability Analysis
# =========================================================
print("\n=====================================")
print("==== Likert Variability Analysis ====")
print("=====================================")
analysis.repeated_measures_anova(df, plot.mode_columns("LikertMean"), alpha=ALPHA)

st_mean = df["Stealth_LikertMean"].mean()
ac_mean = df["Action_LikertMean"].mean()
ad_mean = df["Adaptive_LikertMean"].mean()

print(f"Stealth Score Mean: {st_mean}")
print(f"Action Score Mean: {ac_mean}")
print(f"Adaptive Score Mean: {ad_mean}")

plot.plot_mode_points(df,
                    plot.mode_columns("LikertMean"),
                    title="Participant Mean Experience Scores Across Gameplay Modes",
                    filename="mean_score_point.png",
                    ylabel="Mean Likert Score",
                    y_min = -0.5,
                    y_max = 3.2,)

for col in likert_cols:
    analysis.friedman_test_modes(df, plot.mode_columns(col), ALPHA)

as_results = analysis.ttest_pair(df, "Adaptive_LikertMean", "Stealth_LikertMean", alpha = ALPHA)
aa_results = analysis.ttest_pair(df, "Adaptive_LikertMean", "Action_LikertMean", alpha = ALPHA) 

pdict = {"Mean Adaptive vs Stealth": as_results["p_value"], 
         "Mean Adaptive vs Action": aa_results["p_value"]}
analysis.cal_holm(pdict)

metrics = likert_cols
pdict = {}
for metric in metrics:
    col_a = f"Adaptive_{metric}"
    for mode in ["Stealth", "Action"]:
        col_b = f"{mode}_{metric}"
        result = analysis.wilcoxon_test_pair(df, col_a, col_b, alternative="greater", alpha = ALPHA, only_significant=True)  
        pdict[f"{metric} Adaptive vs {mode}"] = result["p_value"]
analysis.cal_holm(pdict)

# =========================================================
# Consistency Analysis
# =========================================================
print("\n=====================================")
print("======= Consistency Analysis ========")
print("=====================================")

analysis.repeated_measures_anova(df, plot.mode_columns("LikertSD"), alpha=ALPHA)

as_results = analysis.ttest_pair(df, "Adaptive_LikertSD", "Stealth_LikertSD", alternative="less", alpha = ALPHA)   
aa_results = analysis.ttest_pair(df, "Adaptive_LikertSD", "Action_LikertSD", alternative="less", alpha = ALPHA)  

pdict = {"SD Adaptive vs Stealth": as_results["p_value"], 
         "SD Adaptive vs Action": aa_results["p_value"]}
analysis.cal_holm(pdict)


icc_df = analysis.icc_modes_multiple(df, likert_cols)
print(icc_df)
icc_df.to_csv("ICC_Likert.csv", index=False)

plot.plot_bar_xy(icc_df, 
               "mode", "icc", 
               "Intraclass Correlation Coefficient (ICC) by Mode", 
               "ICC_Bar.png",
               ylabel = "ICC (Consistency)", 
               y_min = 0, y_max = 1)

plot.plot_mode_box_likert(df,
                        plot.mode_columns("LikertSD"),
                        title = "Standard Deviation of Likert Scores by Mode",
                        filename = "SD_Box.png",
                        ylabel = "Standard Deviation of Likert Scores",
                        y_min = -0.5,
                        y_max = 3)

# =========================================================
# Mixed Model Analysis
# =========================================================
print("\n=====================================")
print("======= Mixed Model Analysis ========")
print("=====================================")
results = analysis.run_mode_mixed_model(
    df,
    preference_col="Stealth_Tendency",
    mode_outcome_cols= plot.mode_columns("LikertMean"),
    ref = "Action")

print(results["summary"])

plot.plot_mixed_model_predictions(
                results, 
                filename="mixed_model_plot.png",
                title="Mode Performance vs Player Preference",
                y_label="Predicted Mean Likert Score")

for mode in ["Action", "Stealth", "Adaptive"]:
    plot.plot_mixed_model_predictions(
                    results, 
                    filename=f"mixed_model_plot_{mode}.png",
                    title="Mode Performance vs Player Preference",
                    y_label="Predicted Mean Likert Score",
                    show_raw_points = True,
                    highlight_mode = mode)

# =========================================================
# Preference Analysis
# =========================================================
print("\n=====================================")
print("======== Preference Analysis ========")
print("=====================================")

threshold = 0.5
df["preferred_mode"] = np.where(df["Stealth_Tendency"] > (1-threshold), "Stealth", "Neutral")
df.loc[df["Stealth_Tendency"] <= threshold, "preferred_mode"] = "Action"

plot.plot_mode_difference_by_participant(
    df, 
    "preferred_mode", 
    plot.mode_columns("LikertMean"),
    subtract="target_minus_reference",
    title="Difference Between Adaptive and Preferred Mode (Per Participant)",
    filename="adaptive_minus_preferred.png",
    y_label= "Mean Likert Score Difference (Adaptive − Preferred)",
    y_min=-3.0,
    y_max=3.0,
    legend_mean = "Difference")

df["best_mode"] = df[plot.mode_columns("LikertMean", as_list=True)].idxmax(axis=1).str.split("_").str[0]
plot.plot_mode_difference_by_participant(
    df, 
    "best_mode", 
    plot.mode_columns("LikertMean"),
    subtract="reference_minus_target",
    title="Regret Relative to Adaptive Mode (Best − Adaptive)",
    filename="adaptive_regret.png",
    y_label= "Mean Likert Score Difference (Best − Adaptive)",
    y_min=-0.5,
    y_max=3.0,
    sort_values=False,
    legend_prefix = "Best is",
    legend_mean = "Regret")


df["preferred_LikertMean"] = np.where(df["preferred_mode"] == "Action",
                                df["Action_LikertMean"],
                                df["Stealth_LikertMean"])

df["unpreferred_LikertMean"] = np.where(df["preferred_mode"] == "Stealth",
                                df["Action_LikertMean"],
                                df["Stealth_LikertMean"])

ap_results = analysis.tost_pair(df, 
                                "Adaptive_LikertMean", 
                                "preferred_LikertMean", 
                                -0.3, 
                                0.3, 
                                alpha = ALPHA)

au_results = analysis.ttest_pair(df, 
                                 "Adaptive_LikertMean", 
                                 "unpreferred_LikertMean", 
                                 alternative="greater", 
                                 alpha = ALPHA)

pdict = {"Mean Adaptive ~ Preferred": ap_results["p_value"], 
         "Mean Adaptive > unpreferred": au_results["p_value"]}
analysis.cal_holm(pdict)

for col in likert_cols:
    df[f"preferred_{col}"] = np.where(df["preferred_mode"] == "Action",
                                    df[f"Action_{col}"],
                                    df[f"Stealth_{col}"])
    
    df[f"unpreferred_{col}"] = np.where(df["preferred_mode"] == "Stealth",
                                    df[f"Action_{col}"],
                                    df[f"Stealth_{col}"])

metrics = likert_cols
pdict = {}
for metric in metrics:
    col_a = f"Adaptive_{metric}"
    col_b = f"preferred_{metric}"
    p_results = analysis.wilcoxon_equivalence_test_pair(df, 
                                                        col_a, 
                                                        col_b, 
                                                        -0.5, 
                                                        0.5, 
                                                        alpha = ALPHA, 
                                                        only_significant=False) 
    col_b = f"unpreferred_{metric}"
    u_results = analysis.wilcoxon_test_pair(df, 
                                            col_a, 
                                            col_b, 
                                            alternative="greater", 
                                            alpha = ALPHA, 
                                            only_significant=False)
    
    pdict[f"{metric} Adaptive ~ Preferred"] = p_results["p_value"]
    pdict[f"{metric} Adaptive > unpreferred"] = u_results["p_value"]
analysis.cal_holm(pdict)


plot.plot_single_column_by_participant(df,
                                     "Stealth_Tendency",
                                     "Participant Stealth Tendency Distribution",
                                     "Stealth_Tendency_dot.png",
                                     y_label="Stealth Tendency (0.0 = Action, 1.0 = Stealth)",
                                     y_min = 0.0,
                                     y_max = 1.0,
                                     use_threshold = True,
                                     threshold = threshold,
                                     zero_line_value = 0.5,
                                     show_zero_line=True,
                                     zero_line_label="Threshold 0.5",
                                     draw_zone=True)

plot.plot_preference_binned_mode_boxplots(df,
                                        "Stealth_Tendency",
                                        plot.mode_columns("LikertMean"),
                                        title = "Mean Experience Scores by Preference Group and Gameplay Mode",
                                        filename = "preference_binned_boxplots.png",
                                        y_label="Mean Likert Score",
                                        threshold=threshold,)

plot.plot_mode_dumbbell(df,
                      plot.mode_columns("LikertMean"),
                      left_mode="Action",
                      right_mode="Adaptive",
                      title="Participant-Level Comparison: Adaptive vs Action",
                      filename="adaptive_vs_action_dumbbell.png",
                      y_label="Mean Likert Score",
                      sort_by="difference",
                      )

plot.plot_mode_dumbbell(df,
                      plot.mode_columns("LikertMean"),
                      left_mode="Stealth",
                      right_mode="Adaptive",
                      title="Participant-Level Comparison: Adaptive vs Stealth",
                      filename="adaptive_vs_stealth_dumbbell.png",
                      y_label="Mean Likert Score",
                      sort_by="difference",
                      )

plot.plot_mode_dumbbell_preference(df,
                      {
                          "Stealth": "Stealth_LikertMean",
                          "Action": "Action_LikertMean",
                          "Adaptive": "Adaptive_LikertMean",
                          "Preferred": "preferred_LikertMean",
                          "unpreferred": "unpreferred_LikertMean",
                      },
                      left_mode="Preferred",
                      right_mode="Adaptive",
                      title="Participant-Level Comparison: Adaptive vs Preferred Mode",
                      filename="adaptive_vs_Preferred_dumbbell.png",
                      preferred_mode_col = "preferred_mode",
                      y_label="Mean Likert Score",
                      sort_by="difference",
                      )

plot.plot_mode_dumbbell_preference(df,
                      {
                          "Stealth": "Stealth_LikertMean",
                          "Action": "Action_LikertMean",
                          "Adaptive": "Adaptive_LikertMean",
                          "Preferred": "preferred_LikertMean",
                          "unpreferred": "unpreferred_LikertMean",
                      },
                      left_mode="unpreferred",
                      right_mode="Adaptive",
                      title="Participant-Level Comparison: Adaptive vs Non-Preferred Mode",
                      filename="adaptive_vs_unpreferred_dumbbell.png",
                      preferred_mode_col = "preferred_mode",
                      y_label="Mean Likert Score",
                      sort_by="difference",
                      )

# =========================================================
# Playstyle Consistency Analysis
# =========================================================
print("\n=====================================")
print("=== Playstyle Consistency Analysis ==")
print("=====================================")

analysis.repeated_measures_anova(df, plot.mode_columns("Stealth_W_SD"), alpha=ALPHA)

as_results = analysis.ttest_pair(df, "Adaptive_Stealth_W_SD", "Stealth_Stealth_W_SD", alpha = ALPHA)   
aa_results = analysis.ttest_pair(df, "Adaptive_Stealth_W_SD", "Action_Stealth_W_SD", alpha = ALPHA)
sa_results = analysis.ttest_pair(df, "Stealth_Stealth_W_SD", "Action_Stealth_W_SD", alpha = ALPHA)

pdict = {"SD Adaptive vs Stealth": as_results["p_value"], 
         "SD Adaptive vs Action": aa_results["p_value"],
         "SD Stealth vs Action": sa_results["p_value"]}
analysis.cal_holm(pdict)

plot.plot_mode_box_likert(df,
                        plot.mode_columns("Stealth_W_SD"),
                        title = "Standard Deviation of Stealth Weights Across Levels by Gameplay Mode",
                        filename = "SD_W_Box.png",
                        ylabel = "Standard Deviation of Stealth Weight",
                        y_min = -0.1,
                        y_max = 0.5)




