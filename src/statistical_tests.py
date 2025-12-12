from scipy import stats
from src import simulate as sim
from src import data_prep
import statsmodels.api as sm
import numpy as np
import pandas as pd
from firthmodels import FirthLogisticRegression
from patsy import dmatrix
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import Table
import matplotlib.pyplot as plt

formula_specifications = {
    'basic': 'is_ai',
    'basic_across_artists': 'is_ai:artist',
    'model': 'model',
    'model_across_artists': 'model:artist'
}

def get_regression_matrix(formula_specification, winners_losers_df_dict):
    winner_dmatrix = dmatrix(formula_specification, winners_losers_df_dict['winner'], return_type='dataframe')
    loser_dmatrix = dmatrix(winner_dmatrix.design_info, winners_losers_df_dict['loser'])
    cov_matrix = (winner_dmatrix - loser_dmatrix)
    cov_matrix = cov_matrix.drop(columns='Intercept')
    cov_matrix = cov_matrix.drop(columns=[col for col in cov_matrix.columns if col.startswith('artist')])
    return cov_matrix

def flip_part_of_matrix_with_matching_outcome(matrix):
    keep, flip = train_test_split(matrix, test_size = 0.5)
    flip = -flip
    return pd.concat([keep, flip]), np.concatenate([np.ones(len(keep)), np.zeros(len(flip))])

# print(get_regression_matrix(formula_specifications['basic'], winners_losers_df))
def logistic_model_for_ai_human_comparison(hypothesis, winners_losers_df):
    X = get_regression_matrix(formula_specifications[hypothesis], winners_losers_df)
    X, y = flip_part_of_matrix_with_matching_outcome(X)

    # Standard logistic regression
    mod = sm.Logit(y, X)
    res = mod.fit(disp=0)
    logit_results = pd.concat([res.params.rename('logit_coef').to_frame(), res.conf_int().rename(columns={0:'logit_lower', 1:'logit_upper'}), res.pvalues.rename('logit_p').to_frame()], axis = 1)

    # Firth regression (deals with some nasty scenarios)
    firth_model = FirthLogisticRegression(fit_intercept=False).fit(X, y)
    firth_results = pd.concat([pd.Series(firth_model.coef_, name='firth_coef'), pd.DataFrame(firth_model.conf_int()).rename(columns={0:'firth_lower', 1:'firth_upper'}), pd.Series(firth_model.pvalues_, name = 'firth_p')], axis = 1)
    firth_results.index = logit_results.index

    return res, pd.concat([logit_results, firth_results], axis = 1)

def multinomial_model_for_ai_human_comparison(hypothesis, winners_losers_df, reason):
    X = get_regression_matrix(formula_specifications[hypothesis], winners_losers_df)
    mod = sm.MNLogit(reason, X)
    res = mod.fit(disp=0)
    conf_int = res.conf_int()
    ind = conf_int.index
    coef = res.params.unstack().rename('coef').to_frame()
    coef.index = ind
    p = res.pvalues.unstack().rename('p').to_frame()
    p.index = ind
    return res, pd.concat([coef, conf_int, p], axis = 1)

def test_for_ai_difference(logistic_model):
    return logistic_model.pvalues.iloc[0]

def test_for_ai_difference_across_artists(logistic_model):
    return logistic_model.pvalues

def test_for_coefficient_equality(logistic_model, terms):
    return logistic_model.wald_test(','.join([f'({terms[i]} = {terms[i+1]})' for i in range(len(terms) - 1)]), scalar=True).pvalue

def test_for_ai_model_differences(logistic_model):
    names = logistic_model.model.exog_names
    return test_for_coefficient_equality(logistic_model, names)

def test_for_ai_model_differences_across_artists(logistic_model):
    names = logistic_model.model.exog_names
    split_names = list(map(lambda el: el.split(':'), names))
    res_dict = {}
    for artist in np.unique(list(zip(*split_names))[1]):
        vars = [var_name for var_name in names if var_name.split(':')[1] == artist]
        res_dict[artist] = test_for_coefficient_equality(logistic_model, vars)
    return pd.Series(res_dict)

def test_for_coefficient_non_zero(logistic_model, terms):
    return logistic_model.wald_test(','.join([f'({term} = 0)' for term in terms]), scalar=True).pvalue

def test_for_ai_model_nonzero(logistic_model):
    names = logistic_model.model.exog_names
    return test_for_coefficient_non_zero(logistic_model, names)

def test_for_ai_model_nonzero_across_artists(logistic_model):
    names = logistic_model.model.exog_names
    split_names = list(map(lambda el: el.split(':'), names))
    res_dict = {}
    for artist in np.unique(list(zip(*split_names))[1]):
        vars = [var_name for var_name in names if var_name.split(':')[1] == artist]
        res_dict[artist] = test_for_coefficient_non_zero(logistic_model, vars)
    return pd.Series(res_dict)

def test_for_relationship_between_reasons_and_variable(outcomes, factor):
    df = pd.concat(
        [outcomes['reason'], outcomes[factor]],
        axis=1
    ).rename(columns={0: 'reason', 1: factor})

    tab = Table.from_data(df)
    # print(tab.resid_pearson)
    return tab.test_nominal_association().pvalue

def test_for_ai_influence_on_reasons(outcomes):
    return test_for_relationship_between_reasons_and_variable(outcomes, 'ai_win')

def test_for_model_influence_on_reasons(outcomes):
    return test_for_relationship_between_reasons_and_variable(outcomes, 'Model')

def test_for_ai_influence_on_reasons_across_artists(outcomes):
    res_dict = {}
    for artist in outcomes['artist'].unique():
        res_dict[artist] = test_for_ai_influence_on_reasons(outcomes[outcomes['artist']==artist])
    return pd.Series(res_dict)

def test_for_model_influence_on_reasons_across_artists(outcomes):
    res_dict = {}
    for artist in outcomes['artist'].unique():
        res_dict[artist] = test_for_model_influence_on_reasons(outcomes[outcomes['artist']==artist])
    return pd.Series(res_dict)

def test_for_homogeneity_of_association_between_reason_and_factor_across_artists(outcomes, factor):
    # warning: need to think hard about this test
    data = outcomes[[factor, 'reason', 'artist']].astype('category').value_counts().to_frame().reset_index()

    X_homog = dmatrix(f'C({factor}):C(reason) + C(reason):C(artist) + C({factor}):C(artist)', data)
    X_hetero = dmatrix(f'C({factor}):C(reason):C(artist)', data)

    fit = lambda X: sm.GLM(data['count'], X, family=sm.families.Poisson()).fit(disp=0, method_kwargs={'warn_convergence': False})
    mod_homog, mod_hetero = fit(X_homog), fit(X_hetero)

    G2 = mod_homog.deviance - mod_hetero.deviance
    df = mod_homog.df_resid - mod_hetero.df_resid
    p = 1 - stats.chi2.cdf(G2, df)
    return p

def test_for_homogeneity_of_association_between_reason_and_model_across_artists(outcomes):
    return test_for_homogeneity_of_association_between_reason_and_factor_across_artists(outcomes, 'Model')

def test_for_homogeneity_of_association_between_reason_and_ai_across_artists(outcomes):
    return test_for_homogeneity_of_association_between_reason_and_factor_across_artists(outcomes, 'ai_win')

def run_all_tests(winner_losers_df, outcomes, tests=None, no_homog=True):
    test_definitions = {
        # AI grouped
        'does_ai_have_effect': lambda: test_for_ai_difference(
            logistic_model_for_ai_human_comparison('basic', winner_losers_df)[0]),
        
        # model
        'are_ai_models_equal': lambda: test_for_ai_model_differences(
            logistic_model_for_ai_human_comparison('model', winner_losers_df)[0]),
        'does_ai_model_have_effect': lambda: test_for_ai_model_nonzero(
            logistic_model_for_ai_human_comparison('model', winner_losers_df)[0]),
        
        # AI across artists
        'is_ai_have_effect_across_artists': lambda: test_for_ai_difference_across_artists(
            logistic_model_for_ai_human_comparison('basic_across_artists', winner_losers_df)[0]),
        
        # model across artists
        'does_an_ai_model_have_effect_across_artists': lambda: test_for_ai_model_nonzero_across_artists(
            logistic_model_for_ai_human_comparison('model_across_artists', winner_losers_df)[0]),
        'do_ai_models_have_differing_effects_across_artists': lambda: test_for_ai_model_differences_across_artists(
            logistic_model_for_ai_human_comparison('model_across_artists', winner_losers_df)[0]),
        
        # multinomial
        'ai_influence_on_reasons': lambda: test_for_ai_influence_on_reasons(outcomes),
        'model_influence_on_reasons': lambda: test_for_model_influence_on_reasons(outcomes),
        'ai_influence_on_reasons_across_artists': lambda: test_for_ai_influence_on_reasons_across_artists(outcomes),
        'model_influence_on_reasons_across_artists': lambda: test_for_model_influence_on_reasons_across_artists(outcomes),
        'homogeneity_of_rel_between_reason_and_model_across_artists': lambda: test_for_homogeneity_of_association_between_reason_and_model_across_artists(outcomes),
        'homogeneity_of_rel_between_reason_and_ai_across_artists': lambda: test_for_homogeneity_of_association_between_reason_and_ai_across_artists(outcomes),
    }
    
    if tests is None:
        tests = test_definitions.keys()

    if no_homog:
        tests = [t for t in tests if not t.startswith('homog')]
    
    return {k: test_definitions[k]() for k in tests if k in test_definitions}