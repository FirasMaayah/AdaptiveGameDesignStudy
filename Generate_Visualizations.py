"""Generate descriptive visualization graphs from the cleaned study dataset."""

from pathlib import Path
import pandas as pd
import Plot_Functions as plot

# =========================================================
# FILE PATHS
# =========================================================
DATA_CSV = Path("Adaptive_Game_Design_Study_Data.csv")

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(DATA_CSV)

# =========================================================
# GRAPH CALLS
# =========================================================

plot.plot_pie(
    df, 
    "Plays?", 
    "Participant Video Game Engagement Frequency", 
    "Plays_Pie.png",
    answer_order = ["Yes, regularly", "Yes, occasionally", "Rarely", "Never"],
    subtitle = '"Do you Play Video Games?"')

plot.plot_pie(
    df, 
    "Approach", 
    "Preferred Gameplay Approach Among Participants", 
    "Approach_Pie.png",
    use_legend =  False,
    wrap_label_width = 20,
    answer_order = ['Stealthier / Tactical / Planned / Slower / Careful',
                    'Action / Fast-paced / “Guns-blazing”',
                    'A mix of both (sometimes stealth, sometimes action)'],)

plot.plot_pie(
    df, 
    "Experience", 
    "Participant Experience with Keyboard and Mouse Controls", 
    "Experience_Pie.png",
    use_legend =  False,)

plot.plot_bar(
    df, 
    "PlayFrequency", 
    "Weekly Video Game Play Time Distribution", 
    "PlayFrequency_Bar.png",
    answer_order = ["Less than 1 hour", "1-5 hours", "6-10 hours", "11-20 hours", "More than 20 hours"])

plot.plot_bar(
    df, 
    "Platforms", 
    "Distribution of Gaming Platforms Used by Participants", 
    "Platforms_Bar.png",
    is_multiselect = True,
    wrap_label_width = 11,
    subtitle = "(Multiple selections allowed)",
    sub_pos = (1.5, 22.5))

plot.plot_bar(
    df, 
    "GameTypes", 
    "Distribution of Preferred Game Genres", 
    "GameTypes_Bar.png",
    is_multiselect = True,
    skewed_labels = True,
    label_rotation = 45,
    wrap_label_width = 20,
    subtitle = "(Multiple selections allowed)",
    sub_pos = (5, 21))

rank_order = ["Lowest enjoyment", "Middle enjoyment", "Highest enjoyment"]

rank_score_map = {
    "Lowest enjoyment": 1,
    "Middle enjoyment": 2,
    "Highest enjoyment": 3,
}

plot.plot_rank_100_stacked_bar(
    df,
    plot.mode_columns("Rank"),
    "Distribution of Enjoyment Rankings Across Gameplay Modes", 
    "Rank_StackedBar100.png", 
    rank_order,
    ylabel = "Percentage of Participants (%)")

plot.plot_mean_rank_score_bar(
    df,
    plot.mode_columns("Rank"),
    title="Mean Enjoyment Rank Scores Across Gameplay Modes",
    filename="Rank_MeanScore.png",
    rank_score_map=rank_score_map,
    ylabel = "Mean Rank Score (1 = Lowest, 3 = Highest)"
)

plot.plot_bar(
    df, 
    "mode_order", 
    "Mode Order", 
    "mode_order_Bar.png",
    is_multiselect = False,
    skewed_labels = False,
    label_rotation = 0,
    wrap_label_width = 10)

plot.plot_mode_weight_stacked_bar(
    df,
    title = "Average Stealth and Action Weights Across Gameplay Modes",
    filename = "weightsavg_stacked.png",
    ylabel="Average Weight (0.0 = Action leaning, 1.0 = Stealth leaning)")

plot.plot_mode_percentage_bar(
    df, 
    plot.mode_columns("SurvivalRate"), 
    "Average Survival Rate Across Gameplay Modes", 
    "SurvivalRate.png",
    ylabel="Survival Rate (%)")

plot.plot_likert_multi_question_box(
    df,
    ["Absorbed", "Skilled", "Enjoy", "Challenge", "Frustrated", "Again"],
    "Distribution of Likert Ratings Across Gameplay Modes and Items",
    "Likert_Box.png",
    ylabel="Likert Score (−3 to 3)",
    question_labels=[
        '"I felt absorbed in the game."',
        '"I felt skilled at the game."',
        '"I enjoyed playing this mode."',
        '"The game provided a good level of challenge."',
        '"I felt frustrated while playing." (Inverted)',
        '"I would like to play this mode again."',
    ],
    inverted_metrics = ["Frustrated"],
    label_rotation = 45,
    wrap_label_width = 20,
    show_points = False,)

plot.plot_raw_weight_progression(
    df,
    "Stealth",
    title = "Stealth Mode - Raw Stealth Weight Progression",
    filename = "Progression_R_Stealth.png",
    ylabel="Raw Stealth Weight (%)")

plot.plot_raw_weight_progression(
    df,
    "Action",
    title = "Action Mode - Raw Stealth Weight Progression",
    filename = "Progression_R_Action.png",
    ylabel="Raw Stealth Weight (%)")

plot.plot_raw_weight_progression(
    df,
    "Adaptive",
    title = "Adaptive Mode - Raw Stealth Weight Progression",
    filename = "Progression_R_Adaptive.png",
    ylabel="Raw Stealth Weight (%)")
