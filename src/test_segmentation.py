from src.utils import check_if_token_belongs


def test_token_fully_before():
    tok_start = 201.0
    tok_end = 203.0

    seg_start = 300.5
    seg_end = 400.5

    assert not check_if_token_belongs(token_start=tok_start, token_end=tok_end,
                                  segment_start=seg_start, segment_end=seg_end)


def test_token_fully_after():
    tok_start = 300.5
    tok_end = 400.5

    seg_start = 201.0
    seg_end = 203.0

    assert not check_if_token_belongs(token_start=tok_start, token_end=tok_end,
                                  segment_start=seg_start, segment_end=seg_end)


def test_overlap():

    tok_start = 726.23
    tok_end = 726.5

    seg_start = 726.23
    seg_end = 728.59

    assert check_if_token_belongs(token_start=tok_start, token_end=tok_end,
                                  segment_start=seg_start, segment_end=seg_end)


def test_other():
    tok_start = 1064.83
    tok_end = 1065.36

    seg_start = 1064.87
    seg_end = 1067.57

    assert check_if_token_belongs(token_start=tok_start, token_end=tok_end,
                                  segment_start=seg_start, segment_end=seg_end)
