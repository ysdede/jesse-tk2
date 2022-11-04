import sys
import click
import optuna
import statistics
import json
from jessetk2.utils import hp_to_seq
import csv
import jesse.helpers as jh
from jessetk2.Vars import DEFAULT

try:
    from config import config
except ImportError:
    print('Check your config.py file or project folder structure!')
    sys.exit(1)

try:
    from routes import routes as routes_cli
except ImportError:
    print("Error: routes.py not found")
    sys.exit(1)



class OptunaPick:
    def __init__(self, dd, mr, lpr, sharpe, calmar, serenity, profit, imcount, trades, mbr, udd):
        self.dd = dd
        self.mr = mr
        self.lpr = lpr
        self.sharpe = sharpe
        self.calmar = calmar
        self.serenity = serenity
        self.profit = profit
        self.imcount = imcount
        self.trades = trades
        self.mbr = mbr
        self.udd = udd

        try:
            self.db_host = config['databases']['optuna_db_host']
            self.db_port = config['databases']['optuna_db_port']
            self.db_name = config['databases']['optuna_db']
            self.db_user = config['databases']['optuna_user']
            self.db_password = config['databases']['optuna_password']
        except:
            print(
                'Check your config.py file for optuna database settings! example configuration:')
            print("""
                'databases': {
                'postgres_host': '127.0.0.1',
                'postgres_name': 'jesse_db',
                'postgres_port': 5432,
                'postgres_username': 'jesse_user',
                'postgres_password': 'password',
                
                'optuna_db_host': '127.0.0.1',
                'optuna_db_port': 5432,
                'optuna_db': 'optuna_db',
                'optuna_user': 'optuna_user',
                'optuna_password': 'password',
                },
            """)
            exit()

        self.storage = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}"

    def pick(self):
        study_summaries = optuna.study.get_all_study_summaries(storage=self.storage)
        # Sort study_summaries by datetime_start
        studies_sorted = sorted(study_summaries, key=lambda x: x._study_id)

        print(f"{'-'*10} {'-'*8} {'-'*26} {'-'*64}")
        print(f"{'Study ID':<10} {'Trials':<8} {'Datetime':<26} {'Study Name':<64}")
        print(f"{'-'*10} {'-'*8} {'-'*26} {'-'*64}")

        studies_dict = {}

        for ss in studies_sorted:
            studies_dict[ss._study_id] = ss.study_name
            print(
                f"{ss._study_id:<10} {ss.n_trials:<8} {str(ss.datetime_start):<26} {ss.study_name:<64}")

        value = click.prompt('Pick a study', type=int)

        try:
            study_name = studies_dict[value]
            study = optuna.load_study(study_name=study_name, storage=self.storage)
        except Exception as e:
            print(e)
            print('Study not found!')
            exit()

        print(value, study_name)
        print("Number of finished trials: ", len(study.trials))

        # sorted(study.best_trials, key=lambda t: t.values)
        trials = study.trials
        results = []
        parameter_list = []  # to eliminate redundant trials with same parameters
        candidates = {}

        exchange = routes_cli[0][0]
        symbol = routes_cli[0][1]
        timeframe = routes_cli[0][2]
        strategy_name = routes_cli[0][3]
        StrategyClass = jh.get_strategy_class(strategy_name)
        strategy = StrategyClass()

        for trial in trials:

            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            # Check each trial values
            # if any(v < 0 for v in trial.values):
            #     continue

            # Get metrics for objective function 1
            # We may have multiple objective functions
            try:
                obj1 = trial.user_attrs['obj1']
            except:
                obj1 = {
                    'max_dd': trial.user_attrs['max_dd1'] if 'max_dd1' in trial.user_attrs else None,
                    'total_profit': trial.user_attrs['total_profit1'] if 'total_profit1' in trial.user_attrs else None,
                    'trades': trial.user_attrs['trades1'] if 'trades1' in trial.user_attrs else None,
                    'lpr': trial.user_attrs['lpr'] if 'lpr' in trial.user_attrs else None,
                    'mbr': trial.user_attrs['mbr'] if 'mbr' in trial.user_attrs else None,
                    'imcount': trial.user_attrs['imcount1'] if 'imcount1' in trial.user_attrs else None,
                    'udd': trial.user_attrs['udd1'] if 'udd1' in trial.user_attrs else None,
                    'min_margin': trial.user_attrs['min_margin'] if 'min_margin' in trial.user_attrs else None,
                    'max_margin_ratio': trial.user_attrs['max_margin_ratio'] if 'max_margin_ratio' in trial.user_attrs else None,
                    'sharpe': trial.user_attrs['sharpe1'] if 'sharpe1' in trial.user_attrs else None,
                    'calmar': trial.user_attrs['calmar1'] if 'calmar1' in trial.user_attrs else None,
                    'sortino': trial.user_attrs['sortino1'] if 'sortino1' in trial.user_attrs else None,
                    'serenity': trial.user_attrs['serenity1'] if 'serenity1' in trial.user_attrs else None,
                    'win_rate': trial.user_attrs['wr1'] if 'wr1' in trial.user_attrs else None,
                    'paid_fees': trial.user_attrs['paid_fees1'] if 'paid_fees1' in trial.user_attrs else None,
                }


            # Exception for min_trades
            # If backtest fails there'll be no trades1 attribute
            if (not obj1['trades']) or obj1['trades'] <= self.trades:
                continue

            # filters: dd, mr, lpr, sharpe, calmar, serenity, profit, imcount, min_trades
            if obj1['udd'] and obj1['udd'] < self.udd:
                continue

            if obj1['max_dd'] and obj1['max_dd'] < self.dd:
                continue

            if obj1['max_margin_ratio'] and obj1['max_margin_ratio'] > self.mr:
                continue

            if obj1['lpr'] and obj1['lpr'] > self.lpr:
                continue

            if obj1['sharpe'] and obj1['sharpe'] < self.sharpe:
                continue

            if obj1['calmar'] and obj1['calmar'] < self.calmar:
                continue

            if obj1['serenity'] and obj1['serenity'] < self.serenity:
                continue

            if obj1['total_profit'] and obj1['total_profit'] < self.profit:
                continue

            if obj1['imcount'] and obj1['imcount'] > self.imcount:
                continue

            try:
                if obj1['mbr'] and obj1['mbr'] > self.mbr:
                    continue
            except:
                pass

            # Statistics test are useful for some strategies!
            # mean_value = round(statistics.mean((*trial.values, trial.user_attrs['sharpe3'])), 3)
            # std_dev = round(statistics.stdev((*trial.values, trial.user_attrs['sharpe3'])), 5)

            # {key : round(trial.params[key], 5) for key in trial.params}
            rounded_params = trial.params

            # Inject payload HP to route
            hp_new = {
                p['name']: rounded_params[p['name']]
                for p in strategy.hyperparameters()
            }


            rounded_params = hp_new

            # Remove duplicates
            # and mean_value > score_treshold and std_dev < std_dev_treshold:
            if trial.params not in parameter_list:
                hash = hp_to_seq(rounded_params)
                # candidates.append([hash, hp])
                candidates[hash] = rounded_params
                # print(type(trial.values), trial.values)

                # This is also hardcoded for k series for now.
                result_line = [trial.number, f"'{hash}'", *trial.values]

                result_line.extend(iter(obj1.values()))
                result_line.append(rounded_params)

                results.append(result_line)
                parameter_list.append(trial.params)

                        # If parameters meet criteria, add to candidates
                        # if mean_value > score_treshold and std_dev < std_dev_treshold and  trial.user_attrs['sharpe3'] > 2:

        results = sorted(results, key=lambda x: x[2], reverse=True)
        print(f"Picked {len(results)} trials")

        if len(results) == 0:
            print("No candidates found! Check your filters, market types (spot vs perpetual).")
            print("eg. You are using liquidation metrics to filter out spot markets.")
            exit()

        # field names
        fields = ['Trial #', 'Seq']  #, 'Profit', 'insufMargin',
        # find first trial with COMPLETE status
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                break

        print(f"{trial.values=}")

        try:
            # If study has objectives data (added with last version) read metric names and add them as csv field names
            objectives = study.user_attrs['objectives']

            fields.extend(f"Score {i} ({o['metric']})" for i, o in enumerate(objectives))
        except:
            # Otherwise use default string eg. 'Objective n'
            fields.extend(f"Score {i}" for i in range(len(trial.values)))
        fields.extend(k for k, v in obj1.items())
        fields.append('HP')

        res_fn = f'Pick-{self.db_name}-{study_name.replace(" ", "-")}.csv'
        with open(res_fn, 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f, delimiter='\t', lineterminator='\n')
            write.writerow(fields)
            write.writerows(results)

        seq_fn = f'SEQ-{self.db_name}-{study_name.replace(" ", "-")}.py'
        with open(seq_fn, 'w') as f:
            f.write("hps = ")
            f.write(json.dumps(candidates, indent=1))

        print(f"Results saved to {res_fn}")
        print(f"Candidates saved to {seq_fn}")
