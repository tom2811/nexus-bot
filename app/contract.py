"""Versioned host<->bot contract and grounding registry.

Hosts declare the *shape* of the data they send (`grounding.kind`); the bot
never guesses keys. Unknown kinds are rejected, not silently mis-rendered.
"""

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

API_VERSION = "1"


class Grounding(BaseModel):
    """A declared data payload from a host app."""

    kind: str = Field(..., description="Registered grounding kind")
    data: dict[str, Any] = Field(..., description="Data matching the kind's shape")


class UnknownGroundingKind(ValueError):
    """Raised when a host declares a grounding kind this bot does not know."""


class ProductGrounding(BaseModel):
    """Shoe/inventory catalog shape (Kickskraze and compatible stores)."""

    name: str
    brand: str = ""
    colorway: Optional[str] = None
    retailPrice: Optional[float] = None
    availableSizes: list[str]
    inStock: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            for snake, camel in (
                ("available_sizes", "availableSizes"),
                ("in_stock", "inStock"),
                ("retail_price", "retailPrice"),
            ):
                if snake in data and camel not in data:
                    data[camel] = data.pop(snake)
        return data


class OrderGrounding(BaseModel):
    """Order-status shape."""

    order_id: Any
    status: str
    total_amount: Optional[float] = None
    customer_name: Optional[str] = None


class GenericGrounding(BaseModel):
    """Any other declared shape — rendered by the LLM over the raw data."""

    model_config = ConfigDict(extra="allow")


def _render_product(question: str, ctx: dict, size: Optional[str] = None) -> str:
    name = (ctx.get("name") or "this product").strip()
    brand = (ctx.get("brand") or "").strip()
    if brand and not name.lower().startswith(brand.lower()):
        label = f"{brand} {name}"
    else:
        label = name
    sizes = [str(s) for s in (ctx.get("availableSizes") or [])]
    price = ctx.get("retailPrice")
    in_stock = ctx.get("inStock")

    target = size
    if not target:
        match = re.search(r"\b\d+(?:\.\d+)?\b", question)
        target = match.group(0) if match else None

    price_bit = f" at ${float(price):.2f}" if price is not None else ""

    if target:
        if str(target) in sizes:
            return f"Yes, size {target} for {label} is in stock{price_bit}."
        available = ", ".join(sizes) or "none"
        return (
            f"Size {target} for {label} is currently out of stock. "
            f"Available sizes: {available}."
        )
    if in_stock is False or not sizes:
        return f"{label} is currently out of stock."
    return f"{label} is in stock{price_bit}. Available sizes: {', '.join(sizes)}."


def _render_order(question: str, ctx: dict, size: Optional[str] = None) -> str:
    bit = f"Order #{ctx.get('order_id')} status: {ctx.get('status')}."
    if ctx.get("total_amount") is not None:
        bit += f" Total: ${float(ctx['total_amount']):.2f}."
    return bit


@dataclass
class GroundingSpec:
    kind: str
    data_model: type[BaseModel]
    tool: str
    render: Optional[Callable[[str, dict, Optional[str]], str]] = None


GROUNDING_REGISTRY: dict[str, GroundingSpec] = {
    "product": GroundingSpec("product", ProductGrounding, "caller_catalog", _render_product),
    "order": GroundingSpec("order", OrderGrounding, "caller_order", _render_order),
    "generic": GroundingSpec("generic", GenericGrounding, "caller_data", None),
}


@dataclass
class ResolvedGrounding:
    kind: str
    data: dict
    tool: str
    render: Optional[Callable[[str, dict, Optional[str]], str]]


def resolve_grounding(kind: str, data: dict) -> ResolvedGrounding:
    """Validate declared data against its registered shape. Raises on unknown kind."""
    spec = GROUNDING_REGISTRY.get(kind)
    if spec is None:
        raise UnknownGroundingKind(
            f"Unknown grounding kind '{kind}'. Valid kinds: {sorted(GROUNDING_REGISTRY)}"
        )
    validated = spec.data_model.model_validate(data)
    return ResolvedGrounding(spec.kind, validated.model_dump(), spec.tool, spec.render)


def classify_legacy_context(ctx: dict) -> Grounding:
    """Map a legacy `product_context` to a declared kind, without guessing keys."""
    try:
        ProductGrounding.model_validate(ctx)
        return Grounding(kind="product", data=ctx)
    except ValidationError:
        return Grounding(kind="generic", data=ctx)
