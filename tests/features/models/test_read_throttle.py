from src.features.models.read_throttle import ReadThrottle
from tests.features.models.fakes import FakeClock


def _throttle(clock, **kwargs):
    return ReadThrottle(clock=clock.monotonic, sleep=clock.sleep, poll_seconds=0.5, **kwargs)


class TestRateCap:
    def test_without_a_cap_nothing_waits(self):
        clock = FakeClock()
        throttle = _throttle(clock)

        assert throttle.wait(10_000_000) is True
        assert throttle.wait(10_000_000) is True

        assert clock.sleeps == []

    def test_reads_are_spaced_by_bytes_over_rate(self):
        clock = FakeClock()
        throttle = _throttle(clock, bytes_per_second=1000)

        for _ in range(4):
            assert throttle.wait(1000) is True

        assert sum(clock.sleeps) == 3.0

    def test_time_spent_reading_counts_against_the_wait(self):
        clock = FakeClock()
        throttle = _throttle(clock, bytes_per_second=1000)
        throttle.wait(1000)

        clock.now += 1.0
        throttle.wait(1000)

        assert clock.sleeps == []

    def test_a_long_idle_gap_does_not_bank_a_burst(self):
        clock = FakeClock()
        throttle = _throttle(clock, bytes_per_second=1000)
        throttle.wait(1000)
        clock.now += 60.0
        throttle.wait(1000)

        throttle.wait(1000)
        throttle.wait(1000)

        assert sum(clock.sleeps) == 2.0

    def test_waits_are_sliced_so_a_stop_request_is_seen_promptly(self):
        clock = FakeClock()
        throttle = _throttle(clock, bytes_per_second=100)
        throttle.wait(100)

        assert throttle.wait(100) is True
        assert clock.sleeps == [0.5, 0.5]

    def test_a_stop_request_ends_a_rate_wait(self):
        clock = FakeClock()
        throttle = _throttle(clock, bytes_per_second=100)
        throttle.wait(100)
        stop = {"now": False}
        clock.on_sleep = lambda: stop.update(now=True)

        assert throttle.wait(100, should_stop=lambda: stop["now"]) is False
        assert len(clock.sleeps) == 1

    def test_reconfiguring_applies_to_the_next_read(self):
        clock = FakeClock()
        throttle = _throttle(clock, bytes_per_second=1000)
        throttle.wait(1000)
        throttle.configure(None, False, False)

        throttle.wait(1000)
        throttle.wait(1000)

        assert clock.sleeps == []


class TestPauseForGeneration:
    def test_generation_running_reports_the_probe(self):
        assert ReadThrottle().generation_running() is False
        assert ReadThrottle(busy=lambda: False).generation_running() is False
        assert ReadThrottle(busy=lambda: True).generation_running() is True

    def test_waits_while_a_generation_runs_then_continues(self):
        clock = FakeClock()
        running = [True]
        throttle = _throttle(clock, pauses_for_generation=True, busy=lambda: running[0])
        clock.on_sleep = lambda: running.__setitem__(0, len(clock.sleeps) < 2)

        assert throttle.wait(1000) is True

        assert clock.sleeps == [0.5, 0.5]

    def test_holding_means_pausing_is_on_and_a_generation_is_running(self):
        running = [False]
        throttle = ReadThrottle(pauses_for_generation=True, busy=lambda: running[0])

        assert throttle.holding() is False
        running[0] = True
        assert throttle.holding() is True
        throttle.configure(None, False, False)
        assert throttle.holding() is False

    def test_a_throttle_without_a_probe_never_holds(self):
        assert ReadThrottle(pauses_for_generation=True).holding() is False

    def test_does_not_consult_the_probe_when_pausing_is_off(self):
        clock = FakeClock()
        calls = []
        throttle = _throttle(clock, busy=lambda: calls.append(1) or True)

        assert throttle.wait(1000) is True

        assert calls == []
        assert clock.sleeps == []

    def test_a_stop_request_ends_a_pause(self):
        clock = FakeClock()
        throttle = _throttle(clock, pauses_for_generation=True, busy=lambda: True)

        assert throttle.wait(1000, should_stop=lambda: True) is False

        assert clock.sleeps == []

    def test_turning_pausing_off_releases_a_waiting_reader(self):
        clock = FakeClock()
        throttle = _throttle(clock, pauses_for_generation=True, busy=lambda: True)
        clock.on_sleep = lambda: throttle.configure(None, False, False)

        assert throttle.wait(1000) is True

        assert clock.sleeps == [0.5]

    def test_the_rate_cap_does_not_catch_up_after_a_pause(self):
        clock = FakeClock()
        running = [False]
        throttle = _throttle(clock, bytes_per_second=1000, pauses_for_generation=True, busy=lambda: running[0])
        clock.on_sleep = lambda: running.__setitem__(0, len(clock.sleeps) < 2)
        throttle.wait(1000)
        running[0] = True

        throttle.wait(1000)

        assert clock.sleeps == [0.5, 0.5]
