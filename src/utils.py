def check_if_token_belongs(token_start: float, token_end: float, segment_start: float, segment_end: float):
    # Token fully before segment
    if token_start < segment_start and token_end < segment_start:
        return False
    # Token fully after segment
    elif token_start > segment_end:
        return False
    # Token fully inside segment:
    elif token_start >= segment_start and token_end <= segment_end:
        return True
    elif token_start <= segment_start <= token_end <= segment_end:
        overlap = token_end - segment_start
        if overlap >= (token_end - token_start) / 2.0:
            return True
        else:
            return False
    elif segment_start <= token_start <= segment_end <= token_end:
        overlap = segment_end - token_start
        if overlap >= (token_end - token_start) / 2.0:
            return True
        else:
            return False
    else:
        return False
