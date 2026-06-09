import numpy as np

from meridian.config import settings
from meridian.data import generate_baseline
from meridian.model import StandardScaler, TripDurationNet
from meridian.pyfunc import TripDurationModel


def test_scaler_roundtrip():
    x = np.random.rand(100, 4).astype("float32")
    s = StandardScaler().fit(x)
    z = s.transform(x)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-5)
    s2 = StandardScaler.from_state(s.state())
    assert np.allclose(s.transform(x), s2.transform(x))


def test_model_predicts_one_row_per_input():
    cols = list(settings.feature_cols)
    df = generate_baseline(n=32)
    scaler = StandardScaler().fit(df[cols].to_numpy("float32"))
    net = TripDurationNet(n_features=len(cols))
    model = TripDurationModel(net, scaler, cols)
    preds = model.predict_df(df)
    assert preds.shape == (32,)
    assert np.isfinite(preds).all()
