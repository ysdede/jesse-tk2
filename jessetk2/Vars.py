datadir = 'jessetkdata'
initial_test_message = 'Please wait while performing initial test...'
csvd = '\t'  # csv delimiter

DEFAULT = {'dd': -90.0, 'mr': 200.0, 'lpr': 10.0, 'sharpe': -100.0, 'calmar': -100.0, 'serenity': -100.0, 'sortino': -100.0, 'profit': -100.00, 'imcount': 999_999, 'trades': 0}

USER_ATTRS = ['trades', 'dd', 'mr', 'lpr', 'sharpe', 'calmar', 'serenity', 'profit', 'imcount']

Metrics = {
    'start_date': None,
    'finish_date': None,
    'exchange': 'None',
    'symbol': None,
    'tf': None,  # timeframe
    'strategy': None,
    'dna': 'None',
    'total_trades': None,
    'total_profit': 0.0,
    'max_dd': 0.0,
    'annual_return': 0.0,
    'max_margin_ratio': None,
    'lpr': None,
    'pmr': None,
    'insuff_margin_count': None,
    'min_margin': None,
    'paid_fees': 0.0,
    'win_rate': 0,
    'n_of_longs': 0,
    'n_of_shorts': 0,
    'serenity': None,
    'sharpe': None,
    'calmar': None,
    'sortino': None,
    'smart_sharpe': None,
    'smart_sortino': None,
    'win_strk': None,
    'lose_strk': None,
    'largest_win': None,
    'largest_lose': None,
    'n_of_wins': 0,
    'n_of_loses': 0,
    'market_change': None,
    'seq_hps': 'None',
    'parameters': 'None',
    'bench_vol': None,
}

refine_file_header = ['Pair',
                      'TF',
                      'Dna',
                      'Start Date',
                      'End Date',
                      'Total Trades',
                      'Longs %',
                      'Short %',
                      'Total Net Profit',
                      'Max. MR',
                      'PMR',
                      'Max. LP Rate',
                      'Insf. Margin Count',
                      'Max.DD',
                      'Annual Profit',
                      'Winrate',
                      'Serenity',
                      'Sharpe',
                      'Calmar',
                      'Winning Strike',
                      'Losing Strike',
                      'Largest Winning',
                      'Largest Losing',
                      'Num. of Wins',
                      'Num. of Losses',
                      'Paid Fees',
                      'Market Change',
                      'Benchmark Volatility'
                      ]

refine_console_header1 = ['Dna',
                          'Total',
                          'Longs',
                          'Shorts',
                          'Total Net',
                          'Max.',
                          'PMR',
                          'LP',
                          'Insff.',
                          'Max.',
                          'Annual',
                          'Win',
                          'Serenity',
                          'Sharpe',
                          'Calmar',
                          'Winning',
                          'Losing',
                          'Largest',
                          'Largest',
                          'Winning',
                          'Losing',
                          'Paid',
                          'Market',
                          'Benchmark'
                          ]

refine_console_header2 = ['String',
                          'Trades',
                          '%',
                          '%',
                          'Profit %',
                          'Margin %',
                          '%',
                          'Rate',
                          'M.Count',
                          'DD %',
                          'Return %',
                          'Rate %',
                          'Index',
                          'Ratio',
                          'Ratio',
                          'Streak',
                          'Streak',
                          'Win. Trade',
                          'Los. Trade',
                          'Trades',
                          'Trades',
                          'Fees',
                          'Change %',
                          'Volat. %'
                          ]

refine_console_formatter = '{: <22} {: <6} {: <5} {: <7}{: <12} {: <8} {: <8} {: <8} {: <8} {: <8} {: <10} {: <8} {: <8} {: <8} {: <8} {: <8} {: <8} ' \
                           '{: <12} {: <12} {: <10} {: <8} {: <8} {: <8} {: <8}'

random_console_formatter = '{: <12} {: <12} {: <6} {: <5} {: <7} {: <12} {: <8} {: <8} {: <8} {: <8} {: <8} {: <10} {: <8} {: <8} {: <8} {: <8} {: <8} {: <8} ' \
                           '{: <12} {: <12} {: <10} {: <8} {: <8} {: <8}'

random_file_header = ['Pair',  # TODO Pairs for multi routes?
                      'TF',
                      'Dna',
                      'Start Date',
                      'End Date',
                      'Total Trades',
                      'Longs %',
                      'Short %',
                      'Total Net Profit',
                      'Max. MR',
                      'PMR',
                      'Max. LP Rate',
                      'Insf. Margin Count',
                      'Max. DD',
                      'Annual Profit',
                      'Winrate',
                      'Serenity',
                      'Sharpe',
                      'Calmar',
                      'Winning Strike',
                      'Losing Strike',
                      'Largest Winning',
                      'Largest Losing',
                      'Num. of Wins',
                      'Num. of Losses',
                      'Paid Fees',
                      'Market Change',
                      'Benchmark Volatility'
                      ]

random_console_header1 = ['Start',
                          'End',
                          'Total',
                          'Longs',
                          'Shorts',
                          'Total Net',
                          'Max.',
                          'PMR',
                          'LP',
                          'Insff.',
                          'Max.',
                          'Annual',
                          'Win',
                          'Serenity',
                          'Sharpe',
                          'Calmar',
                          'Winning',
                          'Losing',
                          'Largest',
                          'Largest',
                          'Winning',
                          'Losing',
                          'Paid',
                          'Market',
                          'Benchmark'
                          ]
                          
random_console_header2 = ['Date',
                          'Date',
                          'Trades',
                          '%',
                          '%',
                          'Profit %',
                          'Margin %',
                          '%',
                          'Rate',
                          'M.Count',
                          'DD %',
                          'Return %',
                          'Rate %',
                          'Index',
                          'Ratio',
                          'Ratio',
                          'Streak',
                          'Streak',
                          'Win. Trade',
                          'Los. Trade',
                          'Trades',
                          'Trades',
                          'Fees',
                          'Change %',
                          'Volat. %'
                          ]
