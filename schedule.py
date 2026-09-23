"""Deterministic full-denoiser schedules for Qwen-Image-2.1 Turbo+.

The seven-evaluation schedules are the existing, image-tested presets. Counts
above seven refine their longest intervals; six removes one early anchor.
"""

BASE_FULL_EVALUATIONS = 7
DEFAULT_FULL_EVALUATIONS = 12
MIN_FULL_EVALUATIONS = 6

SEVEN_EVALUATION_SCHEDULES = {
    20: (0, 1, 2, 3, 5, 11, 18),
    25: (0, 1, 2, 3, 7, 14, 22),
    30: (0, 1, 2, 3, 8, 17, 27),
    40: (0, 1, 3, 4, 11, 23, 36),
    50: (0, 1, 3, 5, 14, 29, 46),
}


def full_evaluation_schedule(steps: int, count: int = DEFAULT_FULL_EVALUATIONS) -> tuple[int, ...]:
    """Return zero-based full evaluations, including the first and a late step.

    More evaluations refine the previous schedule, so increasing ``count``
    never removes an existing full evaluation. This controls model calls; all
    sampler steps are still performed.
    """
    if steps < 20:
        raise ValueError("Turbo+ requires at least 20 sampler steps")
    if not MIN_FULL_EVALUATIONS <= count <= steps:
        raise ValueError(f"full evaluations must be between {MIN_FULL_EVALUATIONS} and {steps}")
    if count == steps:
        return tuple(range(steps))

    if steps in SEVEN_EVALUATION_SCHEDULES:
        chosen = set(SEVEN_EVALUATION_SCHEDULES[steps])
    else:
        # Preserve the four early evaluations, then scale the later anchors.
        chosen = {0, 1, 2, 3}
        chosen.update(3 + round((index - 3) * (steps - 4) / 26) for index in (8, 17, 27))

    if count < BASE_FULL_EVALUATIONS:
        # The six-evaluation setting removes the third early anchor.
        chosen.remove(sorted(chosen)[2])
    else:
        while len(chosen) < count:
            # Split the largest remaining gap. Include the last sampler step as
            # a boundary so dense settings can reach the no-prediction case.
            anchors = sorted(chosen | {steps - 1})
            gaps = [(right - left, left, right) for left, right in zip(anchors, anchors[1:]) if right - left > 1]
            if not gaps:
                chosen.add(steps - 1)
                continue
            _, left, right = max(gaps, key=lambda gap: (gap[0], -gap[1]))
            chosen.add((left + right) // 2)
    return tuple(sorted(chosen))
