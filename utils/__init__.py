from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Sequence

from tqdm import tqdm


def threaded(func: Callable):
    def wrapper(tasks: Sequence[Any], **kwargs) -> None:
        with ThreadPoolExecutor(10) as e:
            pbar = tqdm(total=len(tasks))
            futures: List = [e.submit(func, task, **kwargs) for task in tasks]
            for future in as_completed(futures):
                pbar.update(1)
                future.result()
            pbar.close()

    return wrapper
