# -*- coding: utf-8 -*-
from styxx import epoch, beacon
from styxx.checksum import canary_sha256


def test_commit_reveal_round_trip_and_tamper():
    items = beacon.select("c" * 64, 32)
    salt = epoch.new_salt()
    c = epoch.commit(items, salt)
    assert epoch.reveal_ok(c["commitment"], items, salt)
    assert c["canary_sha256"] == canary_sha256(items)
    swapped = list(items); swapped[0], swapped[1] = swapped[1], swapped[0]
    assert not epoch.reveal_ok(c["commitment"], swapped, salt)
    assert not epoch.reveal_ok(c["commitment"], items, epoch.new_salt())


def test_a_short_or_non_hex_salt_is_refused():
    import pytest
    items = beacon.select("c" * 64, 4)
    for bad in ("0" * 32, "0" * 63, "zz" * 32, ""):
        with pytest.raises(ValueError):
            epoch.commit(items, bad)
