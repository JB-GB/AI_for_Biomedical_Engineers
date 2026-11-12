"""EULAR-OMERACT ultrasound synovitis scores (grey-scale, Power Doppler, combined)."""

GRADES = (0, 1, 2, 3)


def validate_grade(value: int, name: str = "grade") -> int:
    """Return ``value`` if it is a valid 0-3 grade, otherwise raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in GRADES:
        raise ValueError(f"{name} must be an integer in 0-3, got {value!r}")
    return value


def combined_score(gs: int, pd: int) -> int:
    """Combine grey-scale (GS) and Power Doppler (PD) grades into the EULAR-OMERACT score.

    Grade 0: no GS and no PD. With GS >= 1 the combined grade is max(GS, PD).
    PD without synovial hypertrophy (GS 0, PD > 0) is not scored as synovitis
    and is rejected so the label can be reviewed.
    """
    validate_grade(gs, "gs")
    validate_grade(pd, "pd")
    if gs == 0:
        if pd > 0:
            raise ValueError("PD signal without synovial hypertrophy (GS 0): review label")
        return 0
    return max(gs, pd)
