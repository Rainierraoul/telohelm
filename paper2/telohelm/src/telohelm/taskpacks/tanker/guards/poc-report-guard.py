def check(ctx, out):
    # Trusted report guard (step 3): accept a report action that follows the public rules, and nothing more. A report
    # is a nonempty string of at most 200 characters; a trigger is null or a finite number above 0 and at most 1000;
    # an action with neither does nothing and is accepted. It claims nothing about whether the report is correct.
    try:
        text = out.report
    except AttributeError:
        text = None
        has_text = False
    else:
        has_text = True
    if has_text:
        if not isinstance(text, str) or not text.strip() or len(text) > 200:
            return 'reject'
    try:
        d = out.report_trigger_nm
    except AttributeError:
        return 'accept'
    if d is None:
        return 'accept'
    if isinstance(d, bool) or not isinstance(d, (int, float)):
        return 'reject'
    d = float(d)
    if d != d or d <= 0.0 or d > 1000.0:
        return 'reject'
    return 'accept'
