import pathlib
import pickle
from jesse.research import get_candles
from numpy import ndarray


def get_candles_with_cache(exchange: str, symbol: str, start_date: str, finish_date: str, subs_candles: dict = {}) -> ndarray:
    if not subs_candles:
        return _get_candles_with_cache(exchange, symbol, start_date, finish_date)
    print('Using substitute candles!')
    print(
        f'\t\t{exchange} {symbol} -> {subs_candles["exchange"]} {subs_candles["symbol"]}')
    return _get_candles_with_cache(subs_candles['exchange'], subs_candles['symbol'], start_date, finish_date)


def _get_candles_with_cache(exchange: str, symbol: str, start_date: str, finish_date: str) -> ndarray:
    cache_path = 'storage/cache'
    path = pathlib.Path(cache_path)
    path.mkdir(parents=True, exist_ok=True)

    cache_file_name = f"{exchange}-{symbol}-1m-{start_date}-{finish_date}.pkl"
    cache_file = pathlib.Path(f'{cache_path}/{cache_file_name}')

    if cache_file.is_file():
        with open(f'{cache_path}/{cache_file_name}', 'rb') as handle:
            candles = pickle.load(handle)
    else:
        try:
            candles = get_candles(exchange, symbol, '1m',
                                  start_date, finish_date)
        except Exception as e:
            print(e)
            exit(1)
        with open(f'{cache_path}/{cache_file_name}', 'wb') as handle:
            pickle.dump(candles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return candles
