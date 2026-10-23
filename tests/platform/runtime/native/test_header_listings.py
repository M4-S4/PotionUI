from __future__ import annotations

import pytest

from . import _header_listings as listings


def test_listing_round_trips_a_header_and_drops_metadata_and_offsets():
    header = {
        "__metadata__": {"format": "pt"},
        "b.weight": {"dtype": "BF16", "shape": [4, 2], "data_offsets": [0, 16]},
        "a.scale": {"dtype": "F32", "shape": [], "data_offsets": [16, 20]},
    }
    text = listings.listing_text(header)
    assert text == "a.scale - F32\nb.weight 4,2 BF16\n"
    assert listings.parse_listing(text) == {
        "a.scale": {"dtype": "F32", "shape": []},
        "b.weight": {"dtype": "BF16", "shape": [4, 2]},
    }


def test_there_are_committed_listings():
    assert listings.listing_names()


@pytest.mark.parametrize("name", listings.listing_names())
def test_committed_listing_is_sorted_unique_and_well_formed(name):
    lines = listings.listing_path(name).read_text(encoding="utf-8").splitlines()
    assert lines
    keys = [line.split(" ")[0] for line in lines]
    assert keys == sorted(set(keys))
    assert all(len(line.split(" ")) == 3 for line in lines)
    assert listings.listing_text(listings.load_header(name)).splitlines() == lines


def test_committed_listings_match_the_local_source_headers():
    present = [name for name in listings.listing_names() if listings.source_path(name).is_file()]
    if not present:
        pytest.skip("source headers under ai/ are not part of the checkout")
    stale = [
        name
        for name in present
        if listings.derive_listing(name) != listings.listing_path(name).read_text(encoding="utf-8")
    ]
    assert stale == []
