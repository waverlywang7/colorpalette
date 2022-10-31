from typing import Any

def read_dimension(source): ...

class ReadOnlyWorksheet:
    cell: Any
    iter_rows: Any
    @property
    def values(self): ...
    @property
    def rows(self): ...
    __getitem__: Any
    __iter__: Any
    parent: Any
    title: Any
    sheet_state: str
    def __init__(self, parent_workbook, title, worksheet_path, shared_strings) -> None: ...
    def calculate_dimension(self, force: bool = ...): ...
    def reset_dimensions(self) -> None: ...
    @property
    def min_row(self): ...
    @property
    def max_row(self): ...
    @property
    def min_column(self): ...
    @property
    def max_column(self): ...
