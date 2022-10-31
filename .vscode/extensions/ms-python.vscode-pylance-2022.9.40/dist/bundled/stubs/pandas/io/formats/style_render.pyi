from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)

import jinja2
from pandas import Index
from pandas.core.indexing import _IndexSlice

from pandas._typing import (
    AxisType,
    HashableT,
    Level,
)

BaseFormatter = Union[str, Callable[[object], str]]
ExtFormatter = Union[BaseFormatter, dict[Any, Optional[BaseFormatter]]]
CSSPair = tuple[str, Union[str, float]]
CSSList = list[CSSPair]
CSSProperties = Union[str, CSSList]

class CSSDict(TypedDict):
    selector: str
    props: CSSProperties

class StyleExportDict(TypedDict, total=False):
    apply: Any
    table_attributes: Any
    table_styles: Any
    hide_index: bool
    hide_columns: bool
    hide_index_names: bool
    hide_column_names: bool
    css: dict[str, str | int]

CSSStyles = list[CSSDict]
Subset = Union[_IndexSlice, slice, tuple[slice, ...], list[HashableT], Index]

_StylerT = TypeVar("_StylerT", bound=StylerRenderer)

class StylerRenderer(Generic[_StylerT]):
    loader: jinja2.loaders.PackageLoader
    env: jinja2.environment.Environment
    template_html: jinja2.environment.Template
    template_html_table: jinja2.environment.Template
    template_html_style: jinja2.environment.Template
    template_latex: jinja2.environment.Template
    def format(
        self,
        formatter: ExtFormatter | None = ...,
        subset: Subset | None = ...,
        na_rep: str | None = ...,
        precision: int | None = ...,
        decimal: str = ...,
        thousands: str | None = ...,
        escape: str | None = ...,
        hyperlinks: Literal["html", "latex"] | None = ...,
    ) -> _StylerT: ...
    def format_index(
        self,
        formatter: ExtFormatter | None = ...,
        axis: AxisType = ...,
        level: Level | list[Level] | None = ...,
        na_rep: str | None = ...,
        precision: int | None = ...,
        decimal: str = ...,
        thousands: str | None = ...,
        escape: str | None = ...,
        hyperlinks: Literal["html", "latex"] | None = ...,
    ) -> _StylerT: ...
