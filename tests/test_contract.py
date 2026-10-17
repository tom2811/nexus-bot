"""Tests for the versioned contract and grounding registry."""

import pytest
from pydantic import ValidationError

from app.contract import (
    API_VERSION,
    UnknownGroundingKind,
    classify_legacy_context,
    resolve_grounding,
)


def test_api_version_is_declared():
    assert API_VERSION == "1"


class TestResolveGrounding:
    def test_product_camel_case(self):
        resolved = resolve_grounding(
            "product",
            {"name": "Air Max", "brand": "Nike", "availableSizes": ["9", "10"]},
        )
        assert resolved.tool == "caller_catalog"
        assert resolved.render is not None
        assert "Yes, size 10" in resolved.render("size 10?", resolved.data, "10")

    def test_product_accepts_snake_case_aliases(self):
        resolved = resolve_grounding(
            "product",
            {"name": "Air Max", "available_sizes": ["9"], "in_stock": True},
        )
        assert resolved.data["availableSizes"] == ["9"]
        assert resolved.data["inStock"] is True

    def test_product_out_of_stock_size(self):
        resolved = resolve_grounding(
            "product",
            {"name": "Air Max", "availableSizes": ["9"]},
        )
        answer = resolved.render("size 12?", resolved.data, "12")
        assert "out of stock" in answer

    def test_product_name_including_brand_not_duplicated(self):
        resolved = resolve_grounding(
            "product",
            {"name": "Crocs Classic Clog", "brand": "Crocs", "availableSizes": ["9"]},
        )
        answer = resolved.render("size 9?", resolved.data, "9")
        assert "Crocs Crocs" not in answer
        assert "Crocs Classic Clog" in answer

    def test_product_requires_sizes(self):
        with pytest.raises(ValidationError):
            resolve_grounding("product", {"name": "Air Max"})

    def test_order_renders(self):
        resolved = resolve_grounding(
            "order",
            {"order_id": 1001, "status": "shipped", "total_amount": 99.5},
        )
        answer = resolved.render("status?", resolved.data, None)
        assert "1001" in answer and "shipped" in answer and "$99.50" in answer

    def test_generic_has_no_renderer(self):
        resolved = resolve_grounding("generic", {"anything": [1, 2, 3]})
        assert resolved.tool == "caller_data"
        assert resolved.render is None
        assert resolved.data["anything"] == [1, 2, 3]

    def test_unknown_kind_rejected(self):
        with pytest.raises(UnknownGroundingKind):
            resolve_grounding("mystery", {})


class TestClassifyLegacyContext:
    def test_product_shaped_payload(self):
        grounding = classify_legacy_context(
            {"name": "Air Max", "availableSizes": ["9", "10"]}
        )
        assert grounding.kind == "product"

    def test_unknown_shape_becomes_generic(self):
        grounding = classify_legacy_context({"foo": "bar"})
        assert grounding.kind == "generic"
