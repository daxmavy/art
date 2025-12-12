import pandas as pd
from pandas.api.types import CategoricalDtype

def get_winners_and_losers_dfs(results_dict):
    outcome_df = results_dict['outcomes']

    model_cat_type = CategoricalDtype(
        categories=['Human'] + sorted(list(outcome_df['Model'].unique()))
    )

    artist_cat_type = CategoricalDtype(
        sorted(list(outcome_df['artist'].unique()))
    )

    # model for winner
    winner_model = outcome_df['Model'].case_when(
        [(outcome_df['real_win']== 1, 'Human')]
    ).astype(model_cat_type)
    # model for loser
    loser_model = outcome_df['Model'].case_when(
        [(outcome_df['ai_win']== 1, 'Human')]
    ).astype(model_cat_type)

    winner_is_ai = outcome_df['ai_win'] == 1
    loser_is_ai = outcome_df['real_win'] == 1

    # artist
    artist = outcome_df['artist'].astype(artist_cat_type)

    winner_df = pd.DataFrame({
        'is_ai': winner_is_ai,
        'model': winner_model,
        'artist': artist
    })

    loser_df = pd.DataFrame({
        'is_ai': loser_is_ai,
        'model': loser_model,
        'artist': artist
    })

    return {
        'winner' : winner_df,
        'loser' : loser_df
}