import time

from tests.features.models.fakes import CountingScanner, ThreadSpawner, build_coordinator


def test_rescan_on_an_idle_coordinator_runs_the_scan_to_a_terminal_state():
    scanner = CountingScanner()
    spawner = ThreadSpawner()
    coordinator = build_coordinator(scanner, spawner)

    coordinator.rescan(trigger="admin_reindex")
    spawner.join()

    status = coordinator.status()
    assert status["state"] == "done"
    assert status["trigger"] == "admin_reindex"
    assert scanner.calls == 1


def test_start_indexing_runs_the_scan_without_the_caller_scheduling_it():
    scanner = CountingScanner()
    spawner = ThreadSpawner()
    coordinator = build_coordinator(scanner, spawner)

    coordinator.start_indexing(trigger="manual")
    spawner.join()

    assert coordinator.status()["state"] == "done"
    assert scanner.calls == 1


def test_rescan_while_running_leaves_the_scan_alone_and_reruns_exactly_once_after_it():
    scanner = CountingScanner(hold_first_run=True)
    spawner = ThreadSpawner()
    coordinator = build_coordinator(scanner, spawner)
    coordinator.start_indexing(trigger="manual")
    assert scanner.first_run_entered.wait(timeout=5)

    coordinator.rescan(trigger="roots_change")
    coordinator.rescan(trigger="roots_change")
    assert coordinator.status()["restart_pending"] is True
    time.sleep(0.05)
    scanner.release_first_run.set()
    spawner.join()

    status = coordinator.status()
    assert scanner.first_run_was_cancelled is False
    assert scanner.calls == 2
    assert status["state"] == "done"
    assert status["trigger"] == "roots_change"


def test_start_indexing_while_a_scan_is_running_does_not_run_a_second_scan():
    scanner = CountingScanner(hold_first_run=True)
    spawner = ThreadSpawner()
    coordinator = build_coordinator(scanner, spawner)
    coordinator.start_indexing(trigger="manual")
    assert scanner.first_run_entered.wait(timeout=5)

    coordinator.start_indexing(trigger="manual")
    coordinator.start_indexing(trigger="manual")
    assert len(spawner.threads) == 1
    scanner.release_first_run.set()
    spawner.join()

    assert scanner.calls == 1
    assert coordinator.status()["state"] == "done"
