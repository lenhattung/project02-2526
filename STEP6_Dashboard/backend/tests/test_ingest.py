from app.services.ingest import make_hash, normalize_text, parse_count


def test_normalize_text():
    assert normalize_text("  A\n\nB   C ") == "A B C"


def test_make_hash_is_stable():
    assert make_hash("a", 1) == make_hash("a", 1)


def test_parse_count():
    assert parse_count("1.2K") == 1200
    assert parse_count("12 binh luan") == 12
