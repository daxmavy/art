from matplotlib import pyplot as plt
import pandas as pd
from src import statistical_tests, data_prep, simulate as sim
from matplotlib.ticker import PercentFormatter
import math

def reformat_series_of_dfs(ser_of_dfs):
    return pd.DataFrame(ser_of_dfs.to_list())

def generate_data_and_run_tests(
    N,
    test_keys=None,
    simulation_params=None,
    plot=False,
    no_homog=True
):
    li = []
    for i in range(N):
        if simulation_params is not None:
            sim_results = sim.generate_data_for_simulation(**simulation_params)
        else:
            sim_results = sim.generate_data_for_simulation()
        winners_losers_df = data_prep.get_winners_and_losers_dfs(sim_results)
        tests = statistical_tests.run_all_tests(winners_losers_df, sim_results['outcomes'], test_keys, no_homog)
        li.append(tests)
    p_value_results = pd.DataFrame(li)

    agged_results = {}
    for test in p_value_results.columns:
        try:
            test_results = (p_value_results[test] < 0.05).mean()
            agged_results[test] = test_results.mean()
        except:
            reformatted = reformat_series_of_dfs(p_value_results[test])
            agged_results[test] = (reformatted < 0.05).mean()
    return agged_results


def run_test_over_varying_parameters(N, parameter_settings, test_keys, plot=True, title=None):
    result_dict = {}
    for k, v in parameter_settings.items():
        temp = generate_data_and_run_tests(N, simulation_params = v, test_keys = test_keys)
        result_dict[k] = temp

    transposed = {}
    for test in test_keys:
        transposed[test] = {}
        for k, v in result_dict.items():
            transposed[test][k] = v[test]

    final = {}
    for k, v in transposed.items():
        final[k] = pd.Series(v)
        if final[k].dtype == 'O':
            final[k] = reformat_series_of_dfs(final[k])
            final[k].index = parameter_settings.keys()
    
    n = len(final)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))

    for ax, (k, v) in zip(axes.flat, final.items()):
        v.plot(ax=ax, marker='x')
        ax.set(title=k, ylabel='% sig.')
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    # hide unused axes
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return final