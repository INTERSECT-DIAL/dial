"""
Benchmark which is meant to test core DIAL functionality (without the INTERSECT or MONGO pieces) for the Strain-Mapping benchmark problem.
"""

import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

# from pytest_benchmark.fixture import BenchmarkFixture
from dial_dataclass import (
    DialInputSingleOtherStrategy,
)
from dial_service import (
    core as dial_core,
)
from dial_service.serverside_data import (
    ServersideInputBase,
    ServersideInputSingle,
)
from dial_service.service_specific_dataclasses import DialWorkflowCreationParamsService

logger = logging.getLogger(__name__)

MOCK_WORKFLOW_ID = '6984e6a6ef6e6290dabced91'
"""fake ObjectID for testing purposes, we do not actually interact with a DB in these tests."""


###############
low_f_data_name = '../fixtures/adaptive_strain_manufacturing_low_fidelity.csv'
high_f_data_name = '../fixtures/adaptive_strain_manufacturing_high_fidelity.csv'

def normalize_data(in_data):
    m_data = (in_data.max() + in_data.min()) * 0.5
    d_data = (in_data.max() - in_data.min()) * 0.5
    print(m_data, d_data)
    return (in_data - m_data) / d_data

wall_st_index = 0
wall_st_end = 676
x_col_index = 0 # z follow next
e_col_index = 2 # e11, e22 and e33
nrows = int(26)  #Rows in sim strain map of the wall
ncols = int(26)  #Cols in sim strain map of the wall
data = pd.read_csv(low_f_data_name)

Y, Z, E11, E22, E33, R11, R22, R33 = np.array([data.values[:,x_col_index],
                                               data.values[:,x_col_index+1],
                                               data.values[:,e_col_index],
                                               data.values[:,e_col_index+1],
                                               data.values[:,e_col_index+2],
                                               data.values[:,e_col_index+3],
                                               data.values[:,e_col_index+4],
                                               data.values[:,e_col_index+5]])

x1 = Y.astype(np.float32).reshape(nrows,ncols)
x2 = Z.astype(np.float32).reshape(nrows,ncols)

x1_norm = normalize_data(x1)
x2_norm = normalize_data(x2)

Sim_e11 = np.array(E11).astype(np.float32).reshape(nrows,ncols)
Real_e11 = np.array(R11).astype(np.float32).reshape(nrows,ncols)

Sim_e22 = np.array(E22).astype(np.float32).reshape(nrows,ncols)
Real_e22 = np.array(R22).astype(np.float32).reshape(nrows,ncols)

Sim_e33 = np.array(E33).astype(np.float32).reshape(nrows,ncols)
Real_e33 = np.array(R33).astype(np.float32).reshape(nrows,ncols)

# Normalize data
Sim_e11_norm = normalize_data(Sim_e11)
Real_e11_norm = normalize_data(Real_e11)

Sim_e22_norm = normalize_data(Sim_e22)
Real_e22_norm = normalize_data(Real_e22)

Sim_e33_norm = normalize_data(Sim_e33)
Real_e33_norm = normalize_data(Real_e33)


data = pd.read_csv(high_f_data_name)

Y, Z, E11, E22, E33, R11, R22, R33 = np.array([data.values[:,x_col_index],
                                               data.values[:,x_col_index+1],
                                               data.values[:,e_col_index],
                                               data.values[:,e_col_index+1],
                                               data.values[:,e_col_index+2],
                                               data.values[:,e_col_index+3],
                                               data.values[:,e_col_index+4],
                                               data.values[:,e_col_index+5]])

Sim_hi_e11 = np.array(E11).astype(np.float32).reshape(nrows,ncols)
Sim_hi_e22 = np.array(E22).astype(np.float32).reshape(nrows,ncols)
Sim_hi_e33 = np.array(E33).astype(np.float32).reshape(nrows,ncols)

# Transpose the high-fidelity simulation data
# TODO, figure out why this is transposed
Sim_hi_e11 = Sim_hi_e11.T
Sim_hi_e22 = Sim_hi_e22.T
Sim_hi_e33 = Sim_hi_e33.T

# Normalize data
Sim_hi_e11_norm = normalize_data(Sim_hi_e11)
Sim_hi_e22_norm = normalize_data(Sim_hi_e22)
Sim_hi_e33_norm = normalize_data(Sim_hi_e33)

###############

## Select data for ground truth

# default inputs
INITIAL_BOUNDS = [[-1.0, 1.0], [-1.0, 1.0]]
NUM_DIMS = len(INITIAL_BOUNDS)

MESHGRID_SIZE = nrows
INITIAL_MESHGRIDS = (x1_norm, x2_norm)
INITIAL_POINTS_TO_PREDICT = np.hstack([mg.reshape(-1, 1) for mg in INITIAL_MESHGRIDS])

INITIAL_PREDICTIONS = Sim_hi_e33_norm.reshape(-1, 1)

###############

NOISE_LEVEL = 1e-2

from scipy.interpolate import LinearNDInterpolator
truth_interp = LinearNDInterpolator(INITIAL_POINTS_TO_PREDICT, INITIAL_PREDICTIONS)

@dataclass
class StrainMap:
    """Strain Map function."""

    noise_level: float

    truth_interp: LinearNDInterpolator
    """Strain interpolator that is used as 'ground_truth'"""

    def strain_map(self, x) -> float:
        """
        Represents a measured strain as a function of two simulation parameters.
        """
        x = np.asarray(x).reshape((-1, 2))
        x1, x2 = x[:, 0], x[:, 1]

        y_true = truth_interp(np.asarray(x1), np.asarray(x2))

        print("Evaluated truth model", np.hstack((x, y_true)))

        y_noise = y_true + np.random.normal(size=y_true.shape)
        return y_noise

# build a strain map from the selected truth values
truth_strain_map = StrainMap(truth_interp=truth_interp,
                             noise_level=NOISE_LEVEL)

###############

# test parameters
INITIAL_NUM_POINTS = 9
MAX_ITERATIONS = 10  # only allow a maximum of this many iterations in tests

INITIAL_DATASET_X = np.random.uniform(-1.0, 1.0, size=(INITIAL_NUM_POINTS, NUM_DIMS)).tolist()

MAX_TARGET_Y = -100.

def run_simulation(
    dataset_x: list[list[float]], dataset_y: list[float], strategy: str, strategy_args: object
) -> None:

    # important "Hyper-parameters"
    length_scale = 0.2
    noise_level = NOISE_LEVEL
    constant_value = 1.

    # train model with new data
    client_state = DialWorkflowCreationParamsService(
        dataset_x=dataset_x,
        dataset_y=dataset_y,
        bounds=INITIAL_BOUNDS,
        kernel='rbf',
        kernel_args = {'length_scale': length_scale,
                       'length_scale_bounds': 'fixed',
                       'noise_level': noise_level,
                       'noise_level_bounds': 'fixed',
                       'constant_value': constant_value,
                       'constant_value_bounds': 'fixed',
                       },
        y_is_good=False,  # we wish to minimize y (the error)
        backend='sklearn',  # "sklearn" or "gpax"
        seed=-1,  # Use seed = -1 for random results
        dim_x=2,
    )
    data = ServersideInputBase(client_state)
    model = dial_core.train_model(data)

    # get_surrogate_values
    """
    data = ServersideInputPrediction(
        client_state,
        DialInputPredictions(
            workflow_id=MOCK_WORKFLOW_ID,
            points_to_predict=INITIAL_POINTS_TO_PREDICT,
        )
    )
    surrogate_results = dial_core.get_surrogate_values(data, model)
    mean_grid = np.array(surrogate_results).reshape((MESHGRID_SIZE,) * NUM_DIMS)
    """

    # get_next_point
    data = ServersideInputSingle(
        client_state,
        DialInputSingleOtherStrategy(
            workflow_id=MOCK_WORKFLOW_ID,
            bounds=INITIAL_BOUNDS,
            y_is_good=False,  # we wish to minimize y (the error)
            seed=-1,  # Use seed = -1 for random results
            strategy=strategy,
            strategy_args=strategy_args,
        ),
    )
    next_point = dial_core.get_next_point(data, model)
    dataset_x.append(next_point)

    # compute at next point
    next_point_y = truth_strain_map.strain_map(next_point).reshape(-1).tolist()
    dataset_y.append(next_point_y[0])


def accuracy_benchmark(strategy: str, strategy_args: object) -> tuple[int, float, list[float]]:
    """
    returns:
      - number of iterations taken to reach an acceptably accurate target
      - target value achieved
      - best guess from the input parameters
    """

    iterations = 1
    target = float('inf')

    initial_dataset_x = np.random.uniform(-1.0, 1.0, size=(INITIAL_NUM_POINTS, NUM_DIMS)).tolist()

    dataset_x = INITIAL_DATASET_X
    dataset_y = truth_strain_map.strain_map(dataset_x).reshape(-1).tolist()
    minpos = np.argmin(dataset_y)

    # run simulations until we reach an acceptable target range
    while iterations <= MAX_ITERATIONS:
        try:
            run_simulation(dataset_x, dataset_y, strategy, strategy_args)
        except Exception as e:
            logger.exception('Error during simulation')
            raise AssertionError from e
        minpos = np.argmin(dataset_y)
        target = dataset_y[minpos]
        guess = dataset_x[minpos]
        if target <= MAX_TARGET_Y:
            break
        iterations += 1
        # if iterations >= MAX_ITERATIONS:
        # target = float('inf')
        # iterations = 1 << 63

    return (iterations, target, guess)


TEST_PARAMS = (
    ('strategy', 'strategy_args'),
    [
        (
            'upper_confidence_bound',
            {'exploit': 0., 'explore': 1.},
        ),
        (
            'uncertainty',
            None,
        ),
        # (
        #'random',
        # None,
        # ),
    ],
)


@pytest.mark.parametrize(*TEST_PARAMS)
def test_benchmark_strainmap_accuracy(
    # benchmark: BenchmarkFixture,
    strategy: str,
    strategy_args,
) -> None:
    # NUM_RUNS = 20
    # for _ in range(NUM_RUNS):
    # iterations, target = benchmark(
    # partial(accuracy_benchmark, strategy, strategy_args)
    # )
    iterations, target, guess = accuracy_benchmark(strategy, strategy_args)
    print(
        'Iterations for',
        strategy,
        ': ',
        iterations,
        ' best guess:',
        guess,
        ' with target value:',
        target,
    )
    print(
        'Maximum early terminus value',
        MAX_TARGET_Y,
        ' with ',
        MAX_ITERATIONS,
        ' maximum iterations.',
    )
    print(
        'Accuracy benchmark for strategy:',
        strategy,
        'reached' if iterations <= MAX_ITERATIONS else 'not reached',
    )


if __name__ == '__main__':
    """Generate HTML benchmark report with plots comparing different strategies."""
    import argparse
    import datetime
    import json
    from pathlib import Path

    logger = logging.getLogger(f'{__name__}_runner')

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.use('Agg')  # Use non-interactive backend
    except ImportError:
        logger.error(  # noqa: TRY400
            'Error: matplotlib is required for generating plots. Install it with: pip install matplotlib'
        )
        sys.exit(1)

    def positive_int_type(arg):
        try:
            val = int(arg)
        except ValueError as e:
            msg = 'Must be an integer'
            raise argparse.ArgumentTypeError(msg) from e
        if val < 1:
            msg = 'Argument must be a positive number'
            raise argparse.ArgumentTypeError(msg)
        return val

    parser = argparse.ArgumentParser(description='Generate the Strainmap HTML benchmark pages.')
    parser.add_argument(
        '--num-runs',
        '-n',
        type=positive_int_type,
        default=3,
        help='Number of runs for each strategy.',
    )
    args = parser.parse_args()

    strategies = TEST_PARAMS[1]

    # Run multiple iterations for statistical analysis
    NUM_RUNS = args.num_runs
    logger.info('Running benchmarks with %d iterations per strategy...', NUM_RUNS)

    results = {}
    for strategy, strategy_args in strategies:
        strategy_name = f'{strategy}' + (f' {json.dumps(strategy_args)}' if strategy_args else '')
        logger.info('\nBenchmarking: %s', strategy_name)

        iterations_list = []
        targets_list = []
        guesses_list = []

        for run in range(NUM_RUNS):
            iterations, target, guess = accuracy_benchmark(strategy, strategy_args)
            iterations_list.append(iterations)
            targets_list.append(target)
            guesses_list.append(guess)
            logger.info(
                '  Run %d/%d: iterations=%d, target=%.4f', run + 1, NUM_RUNS, iterations, target
            )

        results[strategy_name] = {
            'strategy': strategy,
            'strategy_args': strategy_args,
            'iterations': iterations_list,
            'targets': targets_list,
            'guesses': guesses_list,
            'avg_iterations': np.mean(iterations_list),
            'std_iterations': np.std(iterations_list),
            'avg_target': np.mean(targets_list),
            'std_target': np.std(targets_list),
            'success_rate': sum(1 for t in targets_list if t <= MAX_TARGET_Y) / NUM_RUNS * 100,
        }

    # Generate plots
    output_dir = Path('reports/benchmarks')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Strainmap Optimization Benchmark Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Average iterations to convergence
    ax1 = axes[0, 0]
    strategy_names = list(results.keys())
    avg_iterations = [results[s]['avg_iterations'] for s in strategy_names]
    std_iterations = [results[s]['std_iterations'] for s in strategy_names]
    bars1 = ax1.bar(
        range(len(strategy_names)), avg_iterations, yerr=std_iterations, capsize=5, alpha=0.7
    )
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Average Iterations')
    ax1.set_title('Iterations to Convergence (Lower is Better)')
    ax1.set_xticks(range(len(strategy_names)))
    ax1.set_xticklabels(
        [s.replace(' ', '\n') for s in strategy_names], rotation=0, ha='center', fontsize=8
    )
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for _i, (bar, val, std) in enumerate(zip(bars1, avg_iterations, std_iterations, strict=False)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{val:.1f}±{std:.1f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    # Plot 2: Average target value achieved
    ax2 = axes[0, 1]
    avg_targets = [results[s]['avg_target'] for s in strategy_names]
    std_targets = [results[s]['std_target'] for s in strategy_names]
    bars2 = ax2.bar(
        range(len(strategy_names)),
        avg_targets,
        yerr=std_targets,
        capsize=5,
        alpha=0.7,
        color='orange',
    )
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Average Target Value')
    ax2.set_title('Final Target Value (Lower is Better)')
    ax2.set_xticks(range(len(strategy_names)))
    ax2.set_xticklabels(
        [s.replace(' ', '\n') for s in strategy_names], rotation=0, ha='center', fontsize=8
    )
    ax2.axhline(
        y=MAX_TARGET_Y, color='r', linestyle='--', label=f'Target Threshold ({MAX_TARGET_Y})'
    )
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for _i, (bar, val, std) in enumerate(zip(bars2, avg_targets, std_targets, strict=False)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{val:.2f}±{std:.2f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    # Plot 3: Success rate
    ax3 = axes[1, 0]
    success_rates = [results[s]['success_rate'] for s in strategy_names]
    bars3 = ax3.bar(range(len(strategy_names)), success_rates, alpha=0.7, color='green')
    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title(f'Success Rate (Target ≤ {MAX_TARGET_Y})')
    ax3.set_xticks(range(len(strategy_names)))
    ax3.set_xticklabels(
        [s.replace(' ', '\n') for s in strategy_names], rotation=0, ha='center', fontsize=8
    )
    ax3.set_ylim([0, 105])
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars3, success_rates, strict=False):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{val:.1f}%',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    # Plot 4: Box plot of iterations distribution
    ax4 = axes[1, 1]
    iterations_data = [results[s]['iterations'] for s in strategy_names]
    bp = ax4.boxplot(
        iterations_data, labels=[s.replace(' ', '\n') for s in strategy_names], patch_artist=True
    )
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('Iterations')
    ax4.set_title('Iterations Distribution')
    ax4.set_xticklabels([s.replace(' ', '\n') for s in strategy_names], fontsize=8)
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'strainmap_benchmark.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info('✓ Plot saved to %s', plot_path)
    plt.close()

    # Generate HTML report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strainmap Optimization Benchmark Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f1f1f1;
        }}
        .best {{
            background-color: #d4edda;
            font-weight: bold;
        }}
        .plot {{
            text-align: center;
            margin: 30px 0;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
        }}
        .summary {{
            background-color: #d1ecf1;
            border-left: 4px solid #0c5460;
            padding: 15px;
            margin: 20px 0;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <h1>📊 Strainmap Optimization Benchmark Report</h1>

    <div class="metadata">
        <p><strong>Generated:</strong> {datetime.datetime.now(tz=datetime.timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Number of Runs per Strategy:</strong> {NUM_RUNS}</p>
        <p><strong>Maximum Iterations:</strong> {MAX_ITERATIONS}</p>
        <p><strong>Target Threshold:</strong> {MAX_TARGET_Y}</p>
        <p><strong>Initial Bounds:</strong> {INITIAL_BOUNDS}</p>
        <p><strong>Initial Points:</strong> {INITIAL_NUM_POINTS}</p>
    </div>

    <div class="summary">
        <h3>🎯 Test Objective</h3>
        <p>This benchmark evaluates different acquisition strategies for Bayesian optimization on the classic Strainmap function.
        The goal is to minimize the function value (error) within {MAX_ITERATIONS} iterations, starting from {INITIAL_NUM_POINTS} initial points.</p>
        <p>The Strainmap function is defined as: <code>f(x,y) = (a-x)² + b(y-x²)²</code> with a=1.0, b=100.0</p>
        <p>The global minimum is at (1.0, 1.0) with f(1,1) = 0</p>
    </div>

    <h2>📈 Benchmark Results</h2>

    <div class="plot">
        <h3>Performance Comparison</h3>
        <img src="strainmap_benchmark.png" alt="Benchmark comparison plots">
    </div>

    <h2>📋 Detailed Statistics</h2>

    <table>
        <thead>
            <tr>
                <th>Strategy</th>
                <th>Strategy Args</th>
                <th>Avg Iterations</th>
                <th>Std Iterations</th>
                <th>Avg Target Value</th>
                <th>Std Target Value</th>
                <th>Success Rate</th>
            </tr>
        </thead>
        <tbody>
"""

    # Find best performers
    best_iterations_idx = min(range(len(strategy_names)), key=lambda i: avg_iterations[i])
    best_target_idx = min(range(len(strategy_names)), key=lambda i: avg_targets[i])
    best_success_idx = max(range(len(strategy_names)), key=lambda i: success_rates[i])

    for i, strategy_name in enumerate(strategy_names):
        result = results[strategy_name]
        row_class = ''
        if i in (best_iterations_idx, best_target_idx, best_success_idx):
            row_class = ' class="best"'

        args_str = json.dumps(result['strategy_args']) if result['strategy_args'] else 'None'

        html_content += f"""            <tr{row_class}>
                <td><code>{result['strategy']}</code></td>
                <td><code>{args_str}</code></td>
                <td>{result['avg_iterations']:.2f}</td>
                <td>{result['std_iterations']:.2f}</td>
                <td>{result['avg_target']:.4f}</td>
                <td>{result['std_target']:.4f}</td>
                <td>{result['success_rate']:.1f}%</td>
            </tr>
"""

    html_content += """        </tbody>
    </table>

    <h2>🏆 Key Findings</h2>
    <div class="summary">
"""

    # Add findings
    best_strategy = strategy_names[best_iterations_idx]
    html_content += f"""        <p><strong>Fastest Convergence:</strong> <code>{best_strategy}</code> with {avg_iterations[best_iterations_idx]:.2f} ± {std_iterations[best_iterations_idx]:.2f} iterations on average</p>
"""

    best_accuracy_strategy = strategy_names[best_target_idx]
    html_content += f"""        <p><strong>Best Accuracy:</strong> <code>{best_accuracy_strategy}</code> with average target value {avg_targets[best_target_idx]:.4f} ± {std_targets[best_target_idx]:.4f}</p>
"""

    best_reliability_strategy = strategy_names[best_success_idx]
    html_content += f"""        <p><strong>Most Reliable:</strong> <code>{best_reliability_strategy}</code> with {success_rates[best_success_idx]:.1f}% success rate</p>
"""

    html_content += """    </div>

    <h2>📊 Raw Data</h2>
    <details>
        <summary>Click to expand raw results JSON</summary>
        <pre style="background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto;">
"""

    # Prepare JSON-serializable results
    json_results = {}
    for strategy_name, result in results.items():
        json_results[strategy_name] = {
            'strategy': result['strategy'],
            'strategy_args': result['strategy_args'],
            'iterations': result['iterations'],
            'targets': result['targets'],
            'guesses': [[float(x) for x in guess] for guess in result['guesses']],
            'statistics': {
                'avg_iterations': float(result['avg_iterations']),
                'std_iterations': float(result['std_iterations']),
                'avg_target': float(result['avg_target']),
                'std_target': float(result['std_target']),
                'success_rate': float(result['success_rate']),
            },
        }

    html_content += json.dumps(json_results, indent=2)
    html_content += """
        </pre>
    </details>

    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; text-align: center;">
        <p>Generated by test_strainmap.py benchmark suite</p>
    </footer>
</body>
</html>
"""

    # Save HTML report
    html_path = output_dir / 'strainmap_benchmark.html'
    html_path.write_text(html_content)
    logger.info('✓ HTML report saved to %s', html_path)

    # Save JSON data
    json_path = output_dir / 'strainmap_benchmark.json'
    json_path.write_text(json.dumps(json_results, indent=2))
    logger.info('✓ JSON data saved to %s', json_path)

    logger.info('\n%s', '=' * 60)
    logger.info('📊 Benchmark Summary')
    logger.info('%s', '=' * 60)
    for strategy_name in strategy_names:
        result = results[strategy_name]
        logger.info('\n%s:', strategy_name)
        logger.info(
            '  Avg Iterations: %.2f ± %.2f', result['avg_iterations'], result['std_iterations']
        )
        logger.info('  Avg Target:     %.4f ± %.4f', result['avg_target'], result['std_target'])
        logger.info('  Success Rate:   %.1f%%', result['success_rate'])

    logger.info('\n%s', '=' * 60)
    logger.info('✓ Open %s in your browser to view the full report', html_path)
    logger.info('%s', '=' * 60)
    # assert iterations <= MAX_ITERATIONS
