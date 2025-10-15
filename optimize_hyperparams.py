
import os
import optuna
import torch
from placement import generate_placement_input, train_placement, calculate_normalized_metrics

def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    lambda_wirelength = trial.suggest_loguniform('lambda_wirelength', 0.1, 100.0)
    lambda_overlap = trial.suggest_loguniform('lambda_overlap', .01, 100.0)
    num_epochs = 5000

    # Fix random seed for reproducibility
    torch.manual_seed(42)
    num_macros = 3
    num_std_cells = 50
    cell_features, pin_features, edge_list = generate_placement_input(num_macros, num_std_cells)
    # Random initial positions (as in main)
    total_cells = cell_features.shape[0]
    spread_radius = 30.0
    angles = torch.rand(total_cells) * 2 * 3.14159
    radii = torch.rand(total_cells) * spread_radius
    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    # Run placement
    result = train_placement(
        cell_features,
        pin_features,
        edge_list,
        num_epochs=num_epochs,
        lr=lr,
        lambda_wirelength=lambda_wirelength,
        lambda_overlap=lambda_overlap,
        verbose=False,
    )
    final_cell_features = result['final_cell_features']
    metrics = calculate_normalized_metrics(final_cell_features, pin_features, edge_list)
    # Multi-objective: return (overlap_ratio, normalized_wl)
    # Optuna will collect Pareto-optimal trials when study is created with two 'minimize' directions
    trial.set_user_attr('overlap_ratio', metrics['overlap_ratio'])
    trial.set_user_attr('normalized_wl', metrics['normalized_wl'])
    return metrics['overlap_ratio'], metrics['normalized_wl']

if __name__ == '__main__':
    # Allow quick local runs by setting OPTUNA_TRIALS environment variable
    #n_trials = int(os.environ.get('OPTUNA_TRIALS', '1000'))
    study = optuna.create_study(directions=['minimize', 'minimize'])
    study.optimize(objective, n_trials=100, n_jobs=-1)

    # For multi-objective studies there is no single "best" trial — study.best_trials
    # returns the Pareto front (non-dominated trials). We'll print the Pareto front
    # and present it sorted two ways: by overlap_ratio (objective 0) and by
    # normalized_wl (objective 1) so you can inspect the extremes for each metric.
    pareto_trials = study.best_trials
    print(f'Found {len(pareto_trials)} Pareto-optimal trials')

    def print_trial(t):
        print(f'  Trial number: {t.number}')
        print(f'    Values (overlap_ratio, normalized_wl): {t.values}')
        print('    Params:')
        for k, v in t.params.items():
            print(f'      {k}: {v}')
        print('    Overlap Ratio:', t.user_attrs.get('overlap_ratio'))
        print('    Normalized Wirelength:', t.user_attrs.get('normalized_wl'))

    # Sort Pareto front by overlap_ratio (objective 0)
    sorted_by_overlap = sorted(pareto_trials, key=lambda tr: tr.values[0])
    print('\nPareto front sorted by overlap ratio (best overlap first):')
    for t in sorted_by_overlap:
        print_trial(t)

    # Sort Pareto front by normalized wirelength (objective 1)
    sorted_by_wl = sorted(pareto_trials, key=lambda tr: tr.values[1])
    print('\nPareto front sorted by normalized wirelength (best wirelength first):')
    for t in sorted_by_wl:
        print_trial(t)

    # Also print the extreme bests for quick reference
    if sorted_by_overlap:
        best_overlap = sorted_by_overlap[0]
        print('\nExtreme best for overlap ratio:')
        print_trial(best_overlap)
    if sorted_by_wl:
        best_wl = sorted_by_wl[0]
        print('\nExtreme best for normalized wirelength:')
        print_trial(best_wl)
