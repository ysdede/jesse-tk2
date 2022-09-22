import contextlib
import json
import signal
import sys
from subprocess import PIPE, Popen, call
import click

import jessetk2.utils as utils
from optuna import samplers
import optuna

try:
    from routes import routes as routes_cli
except ImportError:
    print("Error: routes.py not found")
    sys.exit(1)

try:
    from config import config
except ImportError:
    print(
        "Check your config.py file or project folder structure. You need legacy jesse cli!"
    )
    sys.exit(1)

# Multi-objective NSGAII hyperparamaters optimization with Optuna
# Wraps Jesse ai's backtest function as an objective function

note = "MBR_test"
n_of_trials = 5000
workers = 8

start_date = '2021-05-01'
finish_date = '2022-05-01'

# Set required scoring metrics and directions
# Options   : max_dd, total_profit, total_trades, lpr, insuff_margin_count, min_margin, max_margin_ratio, sharpe, serenity, sortino, calmar, win_rate, paid_fees for now.
# Directions: maximize, minimize
objectives = (
    {'id': 0, 'metric': 'total_profit', 'direction': 'maximize'},
    # {'id': 1, 'metric': 'max_dd', 'direction': 'maximize'},
    {'id': 1, 'metric': 'mbr', 'direction': 'minimize'},
)

exchange = routes_cli[0][0]
symbol = routes_cli[0][1]
timeframe = routes_cli[0][2]
strategy = routes_cli[0][3]

symbols = ""

for r in routes_cli:
    print(r[1])
    symbols = f'{symbols}-{r[1].split("-")[0]}'

print(f"Symbol(s): {symbols}")
print(f"Exchange: {exchange} Symbol: {symbol} Timeframe: {timeframe} Strategy: {strategy}")

db_host = config["databases"]["optuna_db_host"]
db_port = config["databases"]["optuna_db_port"]
db_name = config["databases"]["optuna_db"]
db_user = config["databases"]["optuna_user"]
db_password = config["databases"]["optuna_password"]


def objective(trial):  # sourcery skip: for-append-to-extend, list-comprehension
    tp = trial.suggest_int('tp', 30, 120, step=5)
    dev = trial.suggest_int('dev', 100, 600, step=10)
    boost = trial.suggest_int('boost', 0, 100, step=10)
    div = trial.suggest_int('div', 1, 40, step=1)

    parameters = {"tp": tp, "dev": dev, "boost": boost, "div": div}
    hps = json.dumps(parameters)

    o, err, output, metrics1 = None, None, None, None
    # Long run, _---__---_
    process = Popen(
        ["jesse-tk2", "backtest", start_date, finish_date, "--hp", hps],
        shell=False,
        stdout=PIPE,
    )

    (o, err) = process.communicate()
    exit_code = process.wait()
    output = o.decode("utf-8")
    # print(output)
    metrics1 = utils.get_metrics3(output)

    obj1 = {
        "max_dd": metrics1["max_dd"],
        "total_profit": metrics1["total_profit"],
        "trades": metrics1["total_trades"],
        "lpr": metrics1["lpr"],
        "imcount": metrics1["insuff_margin_count"],
        "min_margin": metrics1["min_margin"],
        "max_margin_ratio": metrics1["max_margin_ratio"],
        "sharpe": metrics1["sharpe"],
        "serenity": metrics1["serenity"],
        "sortino": metrics1["sortino"],
        "calmar": metrics1["calmar"],
        "win_rate": metrics1["win_rate"],
        "paid_fees": metrics1["paid_fees"],
        "mbr": metrics1['mbr']
    }

    trial.set_user_attr("obj1", obj1)

    res = []

    for o in objectives:
        res.append(metrics1[o["metric"]])

    return res


def print_best_params():
    print("Number of finished trials: ", len(study.trials))

    trials = sorted(study.best_trials, key=lambda t: t.values)

    for trial in trials:
        print(f"Trial #{trial.number} Values: { trial.values} {trial.params}")


def save_best_params():
    with open("results.txt", "a") as f:
        f.write(f"Number of finished trials: {len(study.trials)}\n")

        trials = sorted(study.best_trials, key=lambda t: t.values)

        for trial in trials:
            f.write(
                f"Trial: {trial.number} Values: {trial.values} Params: {trial.params}\n"
            )


def signal_handler(sig, frame):

    with contextlib.suppress(Exception):
        print_best_params()
        save_best_params()
    print("You pressed Ctrl+C!")
    sys.exit(0)


# sourcery skip: for-append-to-extend, list-comprehension
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    study_name = f"{note}{symbol}-{strategy}-{exchange.replace(' ', '-')}-{timeframe}-{start_date}-{finish_date}"
    storage = f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}"

    print(f"Running {n_of_trials} trials...")
    print(f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}")
    # my_sampler = optuna.samplers.NSGAIISampler(population_size=80)
    # study = optuna.create_study(study_name=study_name, directions=dirs, storage=storage, load_if_exists=True)

    dirs = []
    for o in objectives:
        dirs.append(o["direction"])

    print("dirs:", dirs)

    try:
        study = optuna.create_study(
            study_name=study_name,
            directions=dirs,
            storage=storage,
            load_if_exists=False,
        )
    except optuna.exceptions.DuplicatedStudyError:
        if click.confirm(
            "Previous study detected. Do you want to resume?", default=True
        ):
            study = optuna.create_study(
                study_name=study_name,
                directions=dirs,
                storage=storage,
                load_if_exists=True,
            )
        elif click.confirm("Delete previous study and start new?", default=False):
            optuna.delete_study(study_name=study_name, storage=storage)
            study = optuna.create_study(
                study_name=study_name,
                directions=dirs,
                storage=storage,
                load_if_exists=False,
            )
        else:
            print("Exiting.")
            exit(1)

    study.set_user_attr("objectives", objectives)
    study.set_user_attr("strategy", strategy)
    study.set_user_attr("exchange", exchange)
    study.set_user_attr("symbols", symbols)
    study.set_user_attr("timeframe", timeframe)

    print(f"Running {n_of_trials} trials...")
    # my_sampler = optuna.samplers.NSGAIISampler(population_size=80)
    study.optimize(objective, n_jobs=workers, n_trials=n_of_trials)

    print_best_params()
    save_best_params()
