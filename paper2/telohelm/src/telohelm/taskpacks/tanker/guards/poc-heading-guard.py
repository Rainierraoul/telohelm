def check(ctx, out):
    # Trusted interface guard (option A plan): accept a heading that is a finite number at least 0 and below 360.
    # It claims nothing about progress or safety; the trial judges whether a procedure joins the tanker.
    try:
        raw = out.heading_deg
    except Exception:
        return 'reject'
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 'reject'
    h = float(raw)
    if h != h or h < 0.0 or h >= 360.0:
        return 'reject'
    return 'accept'
