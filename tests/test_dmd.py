from __future__ import annotations

from typing import Any, ClassVar

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from make_test_data import DataGenerator, SignalGenerator
from pydmd import BOPDMD
from pydmd.preprocessing import hankel_preprocessing as hankel_preprocessing_pydmd

import svdrom.config as config
from svdrom import OptDMD, TruncatedSVD
from svdrom.preprocessing import hankel_preprocessing

# set the dask scheduler to single-threaded
dask.config.set(scheduler="single-threaded")


class BaseTestOptDMD:
    """Base test class for OptDMD containing all generic test methods."""

    u: ClassVar[xr.DataArray]
    s: ClassVar[np.ndarray]
    v: ClassVar[xr.DataArray]
    t: ClassVar[np.ndarray]
    optdmd: ClassVar[OptDMD]
    optdmd_bagging: ClassVar[OptDMD]
    hankel_preprocessing: ClassVar[bool]
    d: ClassVar[int]
    components: ClassVar[list[dict[str, Any]]]
    svd_rank: ClassVar[int]
    X: ClassVar[xr.DataArray]
    X_d: ClassVar[xr.DataArray]

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_basic(self, solver: str) -> None:
        """Basic test to check attributes of the OptDMD class."""
        model: OptDMD = getattr(self, solver)
        assert hasattr(
            model, "modes"
        ), "OptDMD object is missing the 'modes' attribute."
        assert hasattr(model, "eigs"), "OptDMD object is missing the 'eigs' attribute."
        assert hasattr(
            model, "amplitudes"
        ), "OptDMD object is missing the 'amplitudes' attribute."
        assert hasattr(
            model, "modes_std"
        ), "OptDMD object is missing the 'modes_std' attribute."
        assert hasattr(
            model, "eigs_std"
        ), "OptDMD object is missing the 'eigs_std' attribute."
        assert hasattr(
            model, "amplitudes_std"
        ), "OptDMD object is missing the 'amplitudes_std' attribute."
        assert hasattr(
            model, "time_fit"
        ), "OptDMD object is missing the 'time_fit' attribute."
        assert hasattr(
            model, "time_fit_original"
        ), "OptDMD object is missing the 'time_fit_original' attribute."
        assert hasattr(
            model, "num_trials"
        ), "OptDMD object is missing the 'num_trials' attribute."
        assert hasattr(
            model, "trial_size"
        ), "OptDMD object is missing the 'trial_size' attribute."
        assert hasattr(
            model, "parallel_bagging"
        ), "OptDMD object is missing the 'parallel_bagging' attribute."
        assert hasattr(
            model, "dynamics"
        ), "OptDMD object is missing the 'dynamics' attribute."
        assert hasattr(
            model, "time_units"
        ), "OptDMD object is missing the 'time_units' attribute."
        assert hasattr(
            model, "input_time_units"
        ), "OptDMD object is missing the 'input_time_units' attribute."
        assert hasattr(
            model, "hankel_d"
        ), "OptDMD object is missing the 'hankel_d' attribute."
        assert hasattr(
            model, "modes_averaged"
        ), "OptDMD object is missing the 'modes_averaged' attribute."
        assert hasattr(
            model, "modes_std_averaged"
        ), "OptDMD object is missing the 'modes_std_averaged' attribute."

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_fit_basic(self, solver: str) -> None:
        """Test the fit() method of the OptDMD class."""
        model: OptDMD = getattr(self, solver)
        model.fit(
            self.u,
            self.s,
            self.v,
            varpro_opts_dict={"maxiter": 15},
            eig_sort="imag",
        )

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_fit_outputs(self, solver: str) -> None:
        """Test data types and shapes of attributes after
        calling the fit() method."""
        model: OptDMD = getattr(self, solver)
        assert isinstance(model.modes, xr.DataArray), (
            "Expected 'modes' to be of type 'xr.DataArray', "
            f"but got {type(model.modes)} instead."
        )
        assert isinstance(model.eigs, np.ndarray), (
            "Expected 'eigs' to be of type 'np.ndarray', "
            f"but got {type(model.eigs)} instead."
        )
        assert isinstance(model.amplitudes, np.ndarray), (
            "Expected 'amplitudes' to be of type 'np.ndarray', "
            f"but got {type(model.amplitudes)} instead."
        )
        assert isinstance(model.time_fit, np.ndarray), (
            "Expected 'time_fit' to be of type 'np.ndarray', "
            f"but got {type(model.time_fit)} instead."
        )
        assert np.array_equal(
            model.time_fit, self.v.time.values
        ), "Expected 'time_fit' vector to be strictly equal to 'v.time.values'."
        if self.hankel_preprocessing:
            assert isinstance(model.time_fit_original, np.ndarray), (
                "Expected 'time_fit_original' to be of type 'np.ndarray', "
                f"but got {type(model.time_fit_original)} instead."
            )
            time_mapping = self.v.attrs[config.get("hankel_time_mapping_attr")]
            expected_time_fit_original = np.sort(list(time_mapping.keys()))
            assert np.array_equal(
                model.time_fit_original, expected_time_fit_original
            ), (
                "Expected 'time_fit_original' vector to be strictly equal to "
                "the keys of the Hankel time mapping dictionary."
            )
        else:
            assert model.time_fit_original is None, (
                "Expected 'time_fit_original' to be None when "
                "Hankel pre-processing has not been applied."
            )

        assert isinstance(model._t_fit, np.ndarray), (
            "Expected 't_fit' to be of type 'np.ndarray', "
            f"but got {type(model._t_fit)} instead."
        )
        assert np.issubdtype(model._t_fit.dtype, float), (
            f"Expected 't_fit' vector to have data type float, "
            f"but got {model._t_fit.dtype.name}."
        )
        assert model.modes.shape == self.u.shape, (
            f"Expected 'modes.shape' to be {self.u.shape}, "
            f"but got {model.modes.shape} instead."
        )
        if self.hankel_preprocessing:
            expected_shape = (self.u.shape[0] // self.d, self.u.shape[1])
            assert model.modes_averaged is not None
            assert model.modes_averaged.shape == expected_shape, (
                f"For an input dataset with time-delay embedding of {self.d}, "
                f"expected 'modes_averaged.shape' to be {expected_shape}, "
                f"but got {model.modes.shape} instead."
            )
        assert model.eigs.shape == (model.modes.shape[1],), (
            f"Expected 'eigs.shape' to be {(model.modes.shape[1],)}, "
            f"but got {model.eigs.shape} instead."
        )
        assert model.amplitudes.shape == (model.modes.shape[1],), (
            f"Expected 'amplitudes.shape' to be {(model.modes.shape[1],)}, "
            f"but got {model.amplitudes.shape} instead."
        )
        if model.num_trials == 0:
            # no bagging
            assert (
                model.modes_std is None
            ), f"Expected 'modes_std' to be None, but got {model.modes_std} instead."
            assert (
                model.eigs_std is None
            ), f"Expected 'eigs_std' to be None, but got {model.eigs_std} instead."
            assert model.amplitudes_std is None, (
                "Expected 'amplitudes_std' to be None, "
                f"but got {model.amplitudes_std} instead."
            )
        else:
            # with bagging
            assert isinstance(model.modes_std, xr.DataArray), (
                "Expected 'modes_std' to be xr.DataArray, "
                f"but got {model.modes_std} instead."
            )
            assert model.modes_std.shape == model.modes.shape, (
                "Expected 'modes_std' and 'modes' to have the same shape, "
                f"but got shapes {model.modes_std.shape} and {model.modes.shape}, "
                "respectively."
            )
            if self.hankel_preprocessing:
                expected_shape = (self.u.shape[0] // self.d, self.u.shape[1])
                assert model.modes_std_averaged is not None
                assert model.modes_std_averaged.shape == expected_shape, (
                    f"For an input dataset with time-delay embedding of {self.d}, "
                    f"expected 'modes_std_averaged.shape' to be {expected_shape}, "
                    f"but got {model.modes.shape} instead."
                )
            assert isinstance(model.eigs_std, np.ndarray), (
                "Expected 'eigs_std' to be np.ndarray, "
                f"but got {model.eigs_std} instead."
            )
            assert isinstance(model.amplitudes_std, np.ndarray), (
                "Expected 'amplitudes_std' to be np.ndarray, "
                f"but got {model.amplitudes_std} instead."
            )

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_dynamics_attr(self, solver: str) -> None:
        """Test the dynamics attribute."""
        model: OptDMD = getattr(self, solver)
        dynamics = model.dynamics
        assert isinstance(dynamics, xr.DataArray), (
            "Expected 'dynamics' to be of type 'xr.DataArray', "
            f"but got {type(dynamics)} instead."
        )
        assert isinstance(dynamics.data, np.ndarray), (
            "Expected the 'dynamics' DataArray to be backed by a "
            f"np.ndarray, but got a {type(dynamics.data)} instead."
        )
        assert model.time_fit is not None
        assert dynamics.shape == (model.n_modes, len(model.time_fit))
        assert dynamics.dims == ("components", "time")

    @pytest.mark.parametrize("forecast_span", ["20 s", 20])
    @pytest.mark.parametrize("dt", ["2 s", 10, None])
    def test_generate_forecast_time_vector(
        self, forecast_span: str | int, dt: str | int | None
    ) -> None:
        """Test the method to generate the forecast time vector
        with different inputs for the forecast span and forecast
        time step.
        """
        solver = self.optdmd
        assert solver._t_fit is not None
        assert solver._time_fit is not None
        if dt is not None:
            expected_delta_t: int | np.floating = 2
            expected_delta_time: np.timedelta64 | np.floating = np.timedelta64(2, "s")
            expected_len: float | np.floating = 10
        else:
            expected_delta_time = np.mean(np.diff(solver._time_fit))
            expected_delta_t = np.mean(np.diff(solver._t_fit))
            expected_len = 20 / expected_delta_t
        t_forecast, time_forecast = solver._generate_forecast_time_vector(
            forecast_span, dt
        )
        assert isinstance(t_forecast, np.ndarray), (
            "Expected 't_forecast' to be of type 'np.ndarray', "
            f"but got {type(t_forecast)} instead."
        )
        assert isinstance(time_forecast, np.ndarray), (
            "Expected 'time_forecast' to be of type 'np.ndarray', "
            f"but got {type(time_forecast)} instead."
        )
        assert np.unique(np.diff(t_forecast)) == expected_delta_t, (
            "Expected the difference between consecutive elements in "
            f"'t_forecast' to be {expected_delta_t}, but got "
            f"{np.unique(np.diff(t_forecast))} instead."
        )
        if np.issubdtype(self.t.dtype, float):
            assert np.unique(np.diff(time_forecast)) == expected_delta_t, (
                "Expected the difference between consecutive elements in "
                f"'time_forecast' to be {expected_delta_t}, but got "
                f"{np.unique(np.diff(time_forecast))} instead."
            )
        else:
            assert np.unique(np.diff(time_forecast)) == expected_delta_time, (
                "Expected the difference between consecutive elements in "
                f"'time_forecast' to be {expected_delta_time}, but got "
                f"{np.unique(np.diff(time_forecast))} instead."
            )
        assert len(t_forecast) == expected_len, (
            f"Expected the length of 't_forecast' to be {expected_len}, but got "
            f"{len(t_forecast)} instead."
        )
        assert len(time_forecast) == expected_len, (
            f"Expected the length of 'time_forecast' to be {expected_len}, but got "
            f"{len(time_forecast)} instead."
        )
        assert t_forecast[0] == solver._t_fit[-1] + expected_delta_t, (
            "Expected 't_forecast[0]' to be ahead of t_fit[-1] "
            f"by a value of {expected_delta_t},"
            f"but got a value of {t_forecast[0] - solver._t_fit[-1]} instead."
        )
        assert solver.time_fit is not None
        if np.issubdtype(self.t.dtype, float):
            if not self.hankel_preprocessing:
                assert time_forecast[0] == solver.time_fit[-1] + expected_delta_t, (
                    "Expected 'time_forecast[0]' to be ahead of time_fit[-1] "
                    f"by a value of {expected_delta_t} "
                    f"but got {time_forecast[0] - solver._time_fit[-1]} instead."
                )
            else:
                assert solver.time_fit_original is not None
                assert (
                    time_forecast[0] == solver.time_fit_original[-1] + expected_delta_t
                ), (
                    "Expected 'time_forecast[0]' to be ahead of time_fit_original[-1] "
                    f"by a value of {expected_delta_t} "
                    f"but got {time_forecast[0] - solver._time_fit[-1]} instead."
                )

        else:
            if not self.hankel_preprocessing:
                assert time_forecast[0] == solver.time_fit[-1] + expected_delta_time, (
                    "Expected 'time_forecast[0]' to be ahead of time_fit[-1] "
                    f"by a value of {expected_delta_time} "
                    f"but got {time_forecast[0] - solver._time_fit[-1]} instead."
                )
            else:
                assert solver.time_fit_original is not None
                assert (
                    time_forecast[0]
                    == solver.time_fit_original[-1] + expected_delta_time
                ), (
                    "Expected 'time_forecast[0]' to be ahead of time_fit_original[-1] "
                    f"by a value of {expected_delta_time} "
                    f"but got {time_forecast[0] - solver._time_fit[-1]} instead."
                )

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_predict(self, solver: str) -> None:
        """Test the private predict() method, ensuring it returns the
        same output as BOPDMD.forecast() from PyDMD.
        """
        model: OptDMD = getattr(self, solver)
        assert model.solver is not None
        t, _ = model._generate_forecast_time_vector(
            forecast_span="20 s",
            dt="2 s",
        )
        if model.num_trials == 0:
            # without bagging

            # no Dask
            forecast_np = model._predict(t, use_dask=False)
            forecast_np_pydmd = model.solver.forecast(t)
            assert isinstance(forecast_np, np.ndarray), (
                "Expected the forecast to be a np.ndarray, "
                f"but got {type(forecast_np)} instead."
            )
            np.testing.assert_allclose(
                forecast_np.real,
                forecast_np_pydmd.real,
                err_msg="Expected the forecast to match the output of PyDMD.",
            )

            # with Dask
            forecast_da = model._predict(t, use_dask=True)
            assert isinstance(forecast_da, da.Array), (
                "Expected the forecast to be a da.Array, "
                f"but got {type(forecast_da)} instead."
            )
            forecast_np = forecast_da.compute()
            np.testing.assert_allclose(
                forecast_np.real,
                forecast_np_pydmd.real,
                err_msg="Expected the forecast to match the output of PyDMD.",
            )
        else:
            # with bagging

            # no Dask
            prediction_np = model._predict(t, use_dask=False)
            assert isinstance(prediction_np, tuple)
            forecast_np, forecast_var_np = prediction_np
            forecast_np_pydmd, forecast_var_np_pydmd = model.solver.forecast(t)
            assert isinstance(forecast_np, np.ndarray), (
                "Expected the mean forecast to be a np.ndarray, "
                f"but got {type(forecast_np)} instead."
            )
            assert isinstance(forecast_var_np, np.ndarray), (
                "Expected the forecast variance to be a np.ndarray, "
                f"but got {type(forecast_var_np)} instead."
            )
            np.testing.assert_allclose(
                forecast_np.real,
                forecast_np_pydmd.real,
                err_msg="Expected the mean forecast to match the output of PyDMD.",
            )
            np.testing.assert_allclose(
                forecast_var_np.real,
                forecast_var_np_pydmd.real,
                err_msg="Expected the forecast variance to match the output of PyDMD.",
            )

            # with Dask
            prediction_da = model._predict(t, use_dask=True)
            assert isinstance(prediction_da, tuple)
            forecast_da, forecast_var_da = prediction_da
            assert isinstance(forecast_da, da.Array), (
                "Expected the mean forecast to be a da.Array, "
                f"but got {type(forecast_da)} instead."
            )
            assert isinstance(forecast_var_da, da.Array), (
                "Expected the forecast variance to be a da.Array, "
                f"but got {type(forecast_var_da)} instead."
            )
            forecast_np, forecast_var_np = (
                forecast_da.compute(),
                forecast_var_da.compute(),
            )
            np.testing.assert_allclose(
                forecast_np.real,
                forecast_np_pydmd.real,
                err_msg="Expected the mean forecast to match the output of PyDMD.",
            )
            np.testing.assert_allclose(
                forecast_var_np.real,
                forecast_var_np_pydmd.real,
                err_msg="Expected the forecast variance to match the output of PyDMD.",
            )

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_forecast(self, solver: str) -> None:
        """Test for the forecast() method."""
        model: OptDMD = getattr(self, solver)
        forecast_span, dt = "10 s", "1 s"
        if self.hankel_preprocessing:
            expected_forecast_shape = (
                self.u.shape[0] // self.d,
                10,
            )  # 10: 10s span, 1s interval
        else:
            expected_forecast_shape = (self.u.shape[0], 10)  # 10: 10s span, 1s interval
        expected_forecast_dims = (self.u.dims[0], model.time_dimension)
        _, expected_forecast_t_vector = model._generate_forecast_time_vector(
            forecast_span=forecast_span,
            dt=dt,
        )
        if model.num_trials == 0:
            # no bagging
            forecast = model.forecast(forecast_span=forecast_span, dt=dt)
            assert isinstance(forecast, xr.DataArray), (
                "Expected 'forecast' to be of type 'xr.DataArray', "
                f"but got {type(forecast)} instead."
            )
            assert forecast.dims == expected_forecast_dims, (
                f"Expected 'forecast' to have dimensions {expected_forecast_dims}, "
                f"but got {forecast.dims} instead."
            )
            assert forecast.shape == expected_forecast_shape, (
                f"Expected 'forecast' to have shape {expected_forecast_shape}, "
                f"but got {forecast.shape}."
            )
            np.testing.assert_equal(
                forecast.time.values,
                expected_forecast_t_vector,
                err_msg=(
                    f"Expected the forecast time vector to be "
                    f"{expected_forecast_t_vector}, but got {forecast.time.values}."
                ),
            )
        else:
            # with bagging
            forecast_result = model.forecast(forecast_span=forecast_span, dt=dt)
            assert isinstance(forecast_result, tuple)
            forecast, forecast_var = forecast_result
            assert isinstance(forecast, xr.DataArray), (
                "Expected 'forecast' to be of type 'xr.DataArray', "
                f"but got {type(forecast)} instead."
            )
            assert forecast.dims == expected_forecast_dims, (
                f"Expected 'forecast' to have dimensions {expected_forecast_dims}, "
                f"but got {forecast.dims} instead."
            )
            assert forecast.shape == expected_forecast_shape, (
                f"Expected 'forecast' to have shape {expected_forecast_shape}, "
                f"but got {forecast.shape}."
            )
            np.testing.assert_equal(
                forecast.time.values,
                expected_forecast_t_vector,
                err_msg=(
                    f"Expected the forecast time vector to be "
                    f"{expected_forecast_t_vector}, but got {forecast.time.values}."
                ),
            )
            assert isinstance(forecast_var, xr.DataArray), (
                "Expected 'forecast_var' to be of type 'xr.DataArray', "
                f"but got {type(forecast)} instead."
            )
            assert forecast_var.dims == expected_forecast_dims, (
                f"Expected 'forecast_var' to have dimensions {expected_forecast_dims}, "
                f"but got {forecast.dims} instead."
            )
            assert forecast_var.shape == expected_forecast_shape, (
                f"Expected 'forecast_var' to have shape {expected_forecast_shape}, "
                f"but got {forecast.shape}."
            )
            np.testing.assert_equal(
                forecast_var.time.values,
                expected_forecast_t_vector,
                err_msg=(
                    "Expected the forecast variance time vector to be "
                    f"{expected_forecast_t_vector}, but got {forecast.time.values}."
                ),
            )

    @pytest.mark.parametrize("t", [slice(10), 10])
    def test_generate_reconstruct_time_vector(self, t: slice | int) -> None:
        """Test for the generate_reconstruct_time_vector()
        private method.
        """
        solver = self.optdmd
        assert solver._t_fit is not None
        assert solver._time_fit is not None
        t_reconstruct, time_reconstruct, _ = solver._generate_reconstruct_time_vector(t)
        expected_t_reconstruct = np.atleast_1d(solver._t_fit[t])
        if not self.hankel_preprocessing:
            expected_time_reconstruct = np.atleast_1d(solver._time_fit[t])
        else:
            assert solver._time_fit_original is not None
            expected_time_reconstruct = np.atleast_1d(solver._time_fit_original[t])
            if isinstance(t, slice):
                expected_t_reconstruct = expected_t_reconstruct[: -self.d + 1]
            else:
                expected_t_reconstruct = solver._t_fit[t - self.d + 1]
            expected_t_reconstruct = np.atleast_1d(expected_t_reconstruct)
        assert np.array_equal(t_reconstruct, expected_t_reconstruct), (
            f"Expected t_reconstruct to be {expected_t_reconstruct}, "
            f"but got {t_reconstruct} instead."
        )
        assert np.array_equal(time_reconstruct, expected_time_reconstruct), (
            f"Expected time_reconstruct to be {expected_time_reconstruct}, "
            f"but got {time_reconstruct} instead."
        )

    @pytest.mark.parametrize("t", [slice(5), slice(5, 10), 10])
    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_reconstruct(self, solver: str, t: slice | int) -> None:
        """Test for the reconstruct() method."""
        model: OptDMD = getattr(self, solver)
        reconstruction = model.reconstruct(t)
        expected_reconstruct_dims = (self.u.dims[0], model.time_dimension)
        assert model._modes is not None
        assert model.time_fit is not None
        modes = model._modes
        time_fit = model.time_fit

        def check_reconstruction_shape(reconstruction: xr.DataArray) -> None:
            if isinstance(t, slice):
                if self.hankel_preprocessing:
                    expected_reconstruction_shape = (
                        modes.shape[0] // self.d,
                        5,
                    )  # 5: len of slice
                else:
                    expected_reconstruction_shape = (
                        modes.shape[0],
                        5,
                    )  # 5: len of slice
                assert reconstruction.shape == expected_reconstruction_shape, (
                    "Expected 'reconstruction' to have shape "
                    f"{expected_reconstruction_shape}, "
                    f"but got {reconstruction.shape} instead."
                )
            else:
                if self.hankel_preprocessing:
                    expected_reconstruction_shape = (
                        modes.shape[0] // self.d,
                        1,
                    )  # 1: single snapshot
                else:
                    expected_reconstruction_shape = (
                        modes.shape[0],
                        1,
                    )  # 1: single snapshot
                assert reconstruction.shape == expected_reconstruction_shape, (
                    "Expected 'reconstruction' to have shape "
                    f"{expected_reconstruction_shape}, "
                    f"but got {reconstruction.shape} instead."
                )

        if model.num_trials == 0:
            # no bagging
            assert isinstance(reconstruction, xr.DataArray), (
                "Expected 'reconstruction' to be of type 'xr.DataArray', "
                f"but got {type(reconstruction)} instead."
            )
            assert reconstruction.dims == expected_reconstruct_dims, (
                "Expected 'reconstruction' to have dimensions "
                f"{expected_reconstruct_dims}, but got {reconstruction.dims} instead."
            )
            check_reconstruction_shape(reconstruction)
            assert np.array_equal(
                reconstruction[model.time_dimension].values,
                time_fit[t],
            ), (
                "Expected the reconstruction time vector to be: "
                f"{time_fit[t]}, "
                f"but got {reconstruction[model.time_dimension].values} instead."
            )
        else:
            # with bagging
            assert isinstance(reconstruction, tuple)
            reconstruction_mean, reconstruction_var = reconstruction
            assert isinstance(reconstruction_mean, xr.DataArray), (
                "Expected 'reconstruction_mean' to be of type 'xr.DataArray', "
                f"but got {type(reconstruction_mean)} instead."
            )
            assert isinstance(reconstruction_var, xr.DataArray), (
                "Expected 'reconstruction_var' to be of type 'xr.DataArray', "
                f"but got {type(reconstruction_var)} instead."
            )
            for array in (reconstruction_mean, reconstruction_var):
                assert array.dims == expected_reconstruct_dims, (
                    "Expected 'reconstruction' to have dimensions "
                    f"{expected_reconstruct_dims}, but got {array.dims} instead."
                )
                check_reconstruction_shape(array)
                assert np.array_equal(
                    array[model.time_dimension].values, time_fit[t]
                ), (
                    "Expected the reconstruction time vector to be: "
                    f"{time_fit[t]}, "
                    f"but got {array[model.time_dimension].values} instead."
                )


class TestOptDMDRandomData(BaseTestOptDMD):
    """Tests for the OptDMD class using a DataGenerator instance
    to generate random input data, with a time vector containing
    datetimes.
    """

    @classmethod
    def setup_class(cls) -> None:
        generator = DataGenerator(seed=1234)
        generator.generate_svd_results(n_components=10)
        cls.u, cls.s, cls.v, cls.t = generator.u, generator.s, generator.v, generator.t
        cls.optdmd = OptDMD()
        cls.optdmd_bagging = OptDMD(num_trials=5, seed=1234)
        cls.hankel_preprocessing = False


class TestOptDMDCoherentSignal(BaseTestOptDMD):
    """Tests for the OptDMD class using a SignalGenerator instance
    to generate coherent spatio-temporal input data, with a time
    vector containing floats.
    """

    @classmethod
    def setup_class(cls) -> None:
        generator = SignalGenerator()
        generator.generate_svd_results(random_seed=1234)
        cls.u, cls.s, cls.v = generator.u, generator.s, generator.v
        cls.t, cls.components = generator.t, generator.components
        cls.optdmd = OptDMD(input_time_units="s")
        cls.optdmd_bagging = OptDMD(
            input_time_units="s", num_trials=10, trial_size=0.9, seed=1234
        )
        cls.hankel_preprocessing = False

    @pytest.mark.parametrize("solver", ["optdmd", "optdmd_bagging"])
    def test_correct_eigs(self, solver: str) -> None:
        """Test that OptDMD can find the correct frequencies of oscillation
        in the data.
        """
        model: OptDMD = getattr(self, solver)
        assert model.eigs is not None
        omegas = np.sort(
            [component["omega"] for component in self.components],
        )[::-1]  # get the temporal frequencies of oscillation from the data generator
        eigs_imag = [
            np.abs(eig.imag) for eig in model.eigs
        ]  # get the imaginary DMD eigenvalues
        eigs_imag_sorted = np.sort(np.unique(np.round(eigs_imag, decimals=2)))[::-1]
        np.testing.assert_array_almost_equal(
            omegas,
            eigs_imag_sorted,
            decimal=2,
            err_msg=(
                f"The expected imaginary eigenvalues are: "
                f"{np.round(omegas, decimals=2)}, "
                "while the computed imaginary eigenvalues are: "
                f"{np.round(eigs_imag_sorted, decimals=2)}."
            ),
        )


class TestOptDMDHankelMatrix(TestOptDMDCoherentSignal):
    """Tests for the OptDMD class using a SignalGenerator instance
    to generate coherent spatio-temporal input data, pre-processed
    with the Hankel pre-processor prior to SVD and DMD to apply
    time-delay embedding.
    """

    @classmethod
    def setup_class(cls) -> None:
        generator = SignalGenerator()
        generator.generate_signal(random_seed=1234)
        cls.components = generator.components
        X = generator.da.transpose("x", "time")
        cls.d = 2
        X_d = hankel_preprocessing(X, d=cls.d)
        cls.hankel_preprocessing = True
        # convert to Dask-backed Xarray as TruncatedSVD currently only
        # supports Dask arrays
        X_d = X_d.copy(data=da.from_array(X_d.data))
        cls.svd_rank = len(cls.components) * cls.d
        tsvd = TruncatedSVD(n_components=cls.svd_rank)
        tsvd.fit(X_d)
        cls.u, cls.s, cls.v = tsvd.u, tsvd.s, tsvd.v
        cls.t = X_d.time.values
        cls.X, cls.X_d = X, X_d
        cls.optdmd = OptDMD()
        cls.optdmd_bagging = OptDMD(num_trials=5, seed=1234)

    # set up multiple (hankel_d, lags) combinations for the next test
    params: ClassVar[dict[int, list]] = {
        2: [None, (0, 1), (1,)],
        3: [None, (0, 2), (1, 2), (2,)],
    }
    cases: ClassVar[list[tuple]] = [
        (hankel_d, lag) for hankel_d, lags in params.items() for lag in lags
    ]

    @pytest.mark.parametrize("array_type", ["numpy", "dask"])
    @pytest.mark.parametrize(("hankel_d", "lags"), cases)
    def test_extract_hankel_prediction(
        self, array_type: str, hankel_d: int, lags: tuple[int] | tuple[int, int] | None
    ) -> None:
        """Test for the extract_hankel_prediction method."""
        arr: np.ndarray | da.Array = np.random.randn(100, 10)
        if array_type == "dask":
            arr = da.from_array(arr)
        # need to transform array into DataArray because
        # hankel_preprocessing() only takes DataArrays
        X = xr.DataArray(
            data=arr,
            dims=("x", "t"),
            coords={
                "x": ("x", np.arange(arr.shape[0])),
                "t": ("t", np.arange(arr.shape[1])),
            },
        )
        X_d = hankel_preprocessing(X, hankel_d)
        X_extracted = self.optdmd._extract_hankel_prediction(X_d.data, hankel_d, lags)
        ind = slice(lags[0], X.shape[-1]) if lags else slice(X.shape[-1])
        if isinstance(X_extracted, da.Array):
            assert np.array_equal(X_extracted.compute(), X.values[:, ind]), (
                "Expected the output of extract_hankel_preprocessing() "
                "to match the input of hankel_preprocessing()."
            )
        else:
            assert np.array_equal(X_extracted, X.values[:, ind]), (
                "Expected the output of extract_hankel_preprocessing() "
                "to match the input of hankel_preprocessing()."
            )

    @pytest.mark.parametrize("t", [0, 1, 5, 10, -2, -1])
    def test_extract_hankel_time(self, t: int) -> None:
        """Test for the extract_hankel_time method."""
        t_hankel, lag = self.optdmd._extract_hankel_time(t)
        if t < 0:
            expected_t_hankel = t
            expected_lag = self.d - 1
        elif t - self.d < 0:
            expected_t_hankel = 0
            expected_lag = t
        else:
            expected_t_hankel = t - self.d + 1
            expected_lag = self.d - 1
        assert (
            t_hankel == expected_t_hankel
        ), f"Expected t_hankel to be {expected_t_hankel}, but got {t_hankel} instead."
        assert (
            lag == expected_lag
        ), f"Expected lag to be {expected_lag}, but got {lag} instead."

    def test_reconstruct_vs_pydmd(self) -> None:
        """Test that the reconstruct() method gives a similar result
        to PyDMD equivalent when using Hankel pre-processing.
        """
        optdmd_pydmd = BOPDMD(svd_rank=self.svd_rank)
        optdmd_pydmd = hankel_preprocessing_pydmd(optdmd_pydmd, d=self.d)
        optdmd_pydmd.fit(self.X.values, self.t)
        reconstruction_pydmd = optdmd_pydmd.reconstructed_data.real
        reconstruction_svdrom = self.optdmd.reconstruct()
        assert isinstance(reconstruction_svdrom, xr.DataArray)
        reconstruction = reconstruction_svdrom.values.real
        # check that the relative Frobenius error is small
        rel_error = np.linalg.norm(
            reconstruction - reconstruction_pydmd
        ) / np.linalg.norm(reconstruction)
        threshold = 1e-4
        assert rel_error < threshold, (
            "The relative Frobenius error between the SVD-ROM and PyDMD "
            f"reconstructions is {rel_error:.4f}. The threshold is {threshold:.4f}."
        )
