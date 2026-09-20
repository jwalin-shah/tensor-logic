from tensor_logic.kernel_contract import (
    finite_closure_iteration_bound,
    run_with_fuel,
    transitive_closure,
)


def test_finite_transitive_closure_converges_and_derives_paths():
    result = transitive_closure(
        [
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
        ]
    )

    assert result.converged is True
    assert ("a", "d") in result.relation
    assert ("a", "c") in result.relation
    assert result.iterations <= finite_closure_iteration_bound(4)


def test_empty_finite_kernel_terminates_immediately():
    result = transitive_closure([])
    assert result.iterations == 0
    assert result.relation == frozenset()


def test_script_runner_requires_watchdog_for_nonhalting_transition():
    def never_halt(value: int):
        return value + 1, False

    result = run_with_fuel(
        0,
        never_halt,
        fuel=25,
    )

    assert result.halted is False
    assert result.fuel_exhausted is True
    assert result.steps == 25
    assert result.state == 25


def test_script_runner_preserves_expressive_host_logic_when_it_halts():
    # Collatz-like host-language computation: arbitrary branching and updates.
    def step(value: int):
        if value == 1:
            return value, True
        if value % 2 == 0:
            return value // 2, False
        return 3 * value + 1, False

    result = run_with_fuel(
        6,
        step,
        fuel=20,
    )

    assert result.halted is True
    assert result.state == 1
    assert result.fuel_exhausted is False


def test_iteration_bound_is_finite():
    assert finite_closure_iteration_bound(10) == 100
