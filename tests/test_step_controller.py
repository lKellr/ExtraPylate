import numpy as np
from numpy.typing import NDArray
import pytest
from modules.step_control import *

np.set_printoptions(precision=20)


class TestStepController:
    n_steps = 10

    @pytest.mark.parametrize(
        "p, atol, rtol, h_Fhty_expected, h_HW_expected",
        [
            (1, 1e-5, 1e-3, 0.0008877673421094922, 0.0021648724645535384),
            (3, 1e-5, 1e-3, 0.06547204568641049, 0.04652818999868293),
            (3, 0.0, 1e-7, 0.003034901117051207, 0.004646454762147895),
            (3, 1e-9, 0.0, 0.0004902949924716052, 0.0012133098130233417),
            (5, 1e-6, 1e-4, 0.09345858670641538, 0.08813412669645022),
            (7, 1e-6, 1e-4, 0.15065230667916552, 0.16175513600058486),
        ],
    )
    def test_controller_PI_initial(
        self,
        p: int,
        atol: float,
        rtol: float,
        h_Fhty_expected: float,
        h_HW_expected: float,
    ):
        step_controller = StepControllerPI(
            get_default_PI_parameters(p),
            get_default_PI_parameters_rejected(p),
            atol,
            rtol,
            norm=norm_hairer,
        )

        # Brusselator
        def ode_fun(t: float, x: NDArray[np.floating]):
            return np.array(
                [1.0 + x[0] * x[0] * x[1] - 4 * x[0], 3 * x[0] - x[0] * x[0] * x[1]],
                dtype=x.dtype,
            )

        x0 = np.array([1.5, 3.0])

        h_Fhty = step_controller.get_initial_stepFhty(ode_fun, x0, 2.0, p, t0=0.5)
        h_HW = step_controller.get_initial_stepHW(ode_fun, x0, p, t0=0.5)
        assert h_Fhty == h_Fhty_expected, (
            f"Unexpected initial steo size for Flaherty method h_Fhty = {h_Fhty}, should be {h_Fhty_expected}."
        )
        assert h_HW == h_HW_expected, (
            f"Unexpected initial step size for H&W method h_HW = {h_HW}, should be {h_HW_expected}."
        )

    @pytest.mark.parametrize(
        "p, error_ratio, error_ratio_prev, is_retry, next_step_size_expected, accepted_expected",
        [
            (1, 0.1, 0.9, False, 4.8050399605183545, True),  # normal test
            (5, 0.1, 0.9, False, 1.3687981222413332, True),  # normal test
            (7, 0.05, 0.9, False, 1.3411837593073295, True),  # normal test
            (
                5,
                1.1,
                0.01,
                False,
                0.682660892313579,
                True,
            ),  # test step_rejection_limit below
            (
                5,
                1.3,
                0.9,
                False,
                1.0,
                False,
            ),  # test step_rejection_limit above, willö be limited
            (5, 0.9, 0.9, False, 1.0, True),  # test s_deadzone
            (5, 1e-6, 1.0, False, 5.0, True),  # test s_limits
            (5, 0.1, 0.9, True, 1.3687981222413332, True),  # test is_retry
        ],
    )
    def test_controller_PI(
        self,
        p: int,
        error_ratio: float,
        error_ratio_prev: float,
        is_retry: bool,
        next_step_size_expected: float,
        accepted_expected: bool,
    ):
        step_controller = StepControllerPI(
            get_default_PI_parameters(p),
            get_default_PI_parameters_rejected(p),
            atol=1.0,
            rtol=0.0,
            norm=norm_hairer,
            safety_tol=1.0,
            step_rejection_limit=1.2,
            s_deadzone=(0.8, 1.2),
            h_limits=(1e-3, 1e2),
        )
        step_controller.is_retry = is_retry
        step_controller.err_ratio_prev = error_ratio_prev
        step_controller.prev_step_size = 1.0

        # test step sizes during integration
        next_step_size, accepted = step_controller.evaluate_step(
            tried_step_size=1.0,
            error=np.ones((1,)) * error_ratio,
            x_curr=np.zeros((1,)),
            x_pred=np.zeros((1,)),
        )

        assert next_step_size == next_step_size_expected, (
            f"Unexpected predicted step size h = {next_step_size}, should be {next_step_size_expected}"
        )
        assert accepted == accepted_expected, (
            f"Step erroneously{' not' if accepted_expected else ''} accepted"
        )
        if not accepted:
            assert step_controller.is_retry, (
                "Step is_retry flag not correctly set after step failure"
            )

    # @pytest.mark.parametrize(
    #     "contoller, expected_result, additional_kwargs",
    #     [
    #         (EulerExtrapolation, np.array([1.5176659044290977, 2.977254903226454]), {"substep_seq": "harmonic"}),
    #     ],
    # )
    # def test_controller_extrapolation(
    #     self,
    #     step_controller: StepControllerExtrap,
    # ):
    #     step_controller.initialize_scheme(
    #                 is_symmetric=self.is_symmetric,
    #                 table_size=self.table_size,
    #                 substep_seq=self.substep_seq,
    #                 total_feval_cost_for_k=total_feval_cost_for_k,
    #                 local_order_func=self.get_local_order,
    #             )

    #     self.step_controller.get_initial_stepHW(
    #             self.ode_fun, x0, t0=t0, p=self.get_local_order(k_target)
    #         )
    #     step_controller.get_initial_ktarget()
    #     step_controller.norm(step_info["local_error"])

    #     state = step_controller.evaluate_step(error, x_curr, x_pred, k_curr, k_target, allow_early_check)

    #     k_target, next_step_multiplier = (
    #         self.step_controller.get_next_step_parameters(k_final, k_target, state)
    #     )

    #     assert (result[-1] == expected_result).all(), (
    #         f"Unexpected final result with x = {result[-1]}, should be {expected_result}, error: {result[-1] - expected_result}."
    #     )
