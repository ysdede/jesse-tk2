from typing import List, Any, Union, Dict, Optional
import numpy as np
import jesse.helpers as jh

def portfolio_metrics(data) -> List[
    Union[Union[List[Union[str, Any]], List[str], List[Union[Union[str, float], Any]]], Any]]:

    metrics = [
        ['Total Closed Trades', data['total']],
        ['Total Net Profit',
         f"{jh.format_currency(round(data['net_profit'], 4))} ({str(round(data['net_profit_percentage'], 2))}%)"],
        ['Starting => Finishing Balance',
         f"{jh.format_currency(round(data['starting_balance'], 2))} => {jh.format_currency(round(data['finishing_balance'], 2))}"],
        ['Total Open Trades', data['total_open_trades']],
        ['Open PL', jh.format_currency(round(data['open_pl'], 2))],
        ['Total Paid Fees', jh.format_currency(round(data['fee'], 2))],
        ['Max Drawdown', f"{round(data['max_drawdown'], 2)}%"],
        ['Annual Return', f"{round(data['annual_return'], 2)}%"],
        ['Expectancy',
         f"{jh.format_currency(round(data['expectancy'], 2))} ({str(round(data['expectancy_percentage'], 2))}%)"],
        ['Avg Win | Avg Loss',
         f"{jh.format_currency(round(data['average_win'], 2))} | {jh.format_currency(round(data['average_loss'], 2))}"],
        ['Ratio Avg Win / Avg Loss', round(data['ratio_avg_win_loss'], 2)],
        ['Percent Profitable', f"{str(round(data['win_rate'] * 100))}%"],
        ['Longs | Shorts', f"{round(data['longs_percentage'])}% | {round(data['shorts_percentage'])}%"],
        ['Avg Holding Time', jh.readable_duration(data['average_holding_period'], 3)],
        ['Winning Trades Avg Holding Time',
         np.nan if np.isnan(data['average_winning_holding_period']) else jh.readable_duration(
             data['average_winning_holding_period'], 3)],
        ['Losing Trades Avg Holding Time',
         np.nan if np.isnan(data['average_losing_holding_period']) else jh.readable_duration(
             data['average_losing_holding_period'], 3)]
    ]

    metrics.append(['Serenity Index', round(data['serenity_index'], 2)])
    metrics.append(['Smart Sharpe', round(data['smart_sharpe'], 2)])
    metrics.append(['Smart Sortino', round(data['smart_sortino'], 2)])
    metrics.append(['Sharpe Ratio', round(data['sharpe_ratio'], 2)])
    metrics.append(['Calmar Ratio', round(data['calmar_ratio'], 2)])
    metrics.append(['Sortino Ratio', round(data['sortino_ratio'], 2)])
    metrics.append(['Omega Ratio', round(data['omega_ratio'], 2)])
    metrics.append(['Winning Streak', data['winning_streak']])
    metrics.append(['Losing Streak', data['losing_streak']])
    metrics.append(['Largest Winning Trade', jh.format_currency(round(data['largest_winning_trade'], 2))])
    metrics.append(['Largest Losing Trade', jh.format_currency(round(data['largest_losing_trade'], 2))])
    metrics.append(['Total Winning Trades', data['total_winning_trades']])
    metrics.append(['Total Losing Trades', data['total_losing_trades']])

    return metrics
