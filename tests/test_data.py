from meridian.config import settings
from meridian.data import generate_baseline, generate_skewed


def test_schema_and_target():
    df = generate_baseline(n=500)
    for col in settings.feature_cols:
        assert col in df.columns
    assert settings.target_col in df.columns
    assert (df[settings.target_col] >= 1.0).all()


def test_skew_shifts_distribution():
    base = generate_baseline(n=5000)
    skew = generate_skewed(n=5000)
    # Skewed traffic is wetter and colder by construction.
    assert skew["precip_mm"].mean() > base["precip_mm"].mean()
    assert skew["temp_c"].mean() < base["temp_c"].mean()
