import importlib
from typing import Any, List

from rich.console import Console

error_console = Console(stderr=True, style='bold red')


def check_installed_package(package_name: str) -> bool:
    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError:
        error_console.log(f'Could not load the package: {package_name}')
        return False
    return True


def chunk_with_size_n(list_of_items: List[Any], chunk_size: int) -> List[Any]:
    for i in range(0, len(list_of_items), chunk_size):
        yield list_of_items[i:i + chunk_size]


def read_lines(file_path: str) -> List[str]:
    lines = []
    with open(file_path, 'r') as inf:
        for line in inf:
            lines.append(line.rstrip('\n'))
    return lines


def write_lines(lines: List[str], file_path: str) -> None:
    with open(file_path, 'w') as out_file:
        for line in lines:
            out_file.write(f'{line}\n')
