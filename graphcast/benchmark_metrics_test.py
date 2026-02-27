import numpy as np
import xarray
from absl.testing import absltest


def _crps_bias_corrected(targets: xarray.DataArray, predictions: xarray.DataArray) -> xarray.DataArray:
  if predictions.sizes.get("sample", 1) < 2:
    raise ValueError("predictions must have dim 'sample' with size at least 2.")
  sum_dims = ["sample", "sample2"]
  preds2 = predictions.rename({"sample": "sample2"})
  num_samps = predictions.sizes["sample"]
  num_samps2 = num_samps - 1
  mean_abs_diff = np.abs(predictions - preds2).sum(dim=sum_dims, skipna=False) / (num_samps * num_samps2)
  mean_abs_err = np.abs(targets - predictions).sum(dim="sample", skipna=False) / num_samps
  return mean_abs_err - 0.5 * mean_abs_diff


class BenchmarkMetricsTest(absltest.TestCase):
  def test_crps_matches_hand_computation(self):
    # One spatial point, two ensemble members.
    targets = xarray.DataArray(np.array([2.0]), dims=("point",))
    preds = xarray.DataArray(
        np.array([[1.0], [3.0]]),
        dims=("sample", "point"),
        coords={"sample": [0, 1]},
    )
    # mean_abs_err = (|2-1| + |2-3|)/2 = 1
    # mean_abs_diff = |1-3| / (2*(2-1)) = 2/2 = 1
    # crps = 1 - 0.5*1 = 0.5
    crps = _crps_bias_corrected(targets, preds)
    np.testing.assert_allclose(crps.data, np.array([0.5]))

  def test_crps_requires_at_least_two_samples(self):
    targets = xarray.DataArray(np.array([0.0]), dims=("point",))
    preds = xarray.DataArray(np.array([[0.0]]), dims=("sample", "point"), coords={"sample": [0]})
    with self.assertRaises(ValueError):
      _crps_bias_corrected(targets, preds)

  def test_weighted_rmse_matches_unweighted_for_uniform_weights(self):
    # If weights are uniform, weighted RMSE should match unweighted RMSE.
    targets = xarray.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    preds = xarray.DataArray(
        np.array([[2.0, 2.0], [2.0, 2.0]]),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    err2 = (preds - targets) ** 2
    unweighted_rmse = (err2.mean()) ** 0.5

    weights = xarray.DataArray(np.ones((2, 2)), dims=("lat", "lon"))
    weighted_rmse = ((err2 * weights).sum() / weights.sum()) ** 0.5
    np.testing.assert_allclose(weighted_rmse.data, unweighted_rmse.data)


if __name__ == "__main__":
  absltest.main()
