"""megan-tk's DisturbanceAnticipator: learn a recurring disturbance's rhythm and
schedule a protective action before each predicted hit (pure, no mujoco)."""

from megantk.anticipation import DisturbanceAnticipator


def test_learns_period_and_predicts_from_phase():
    a = DisturbanceAnticipator(lead=2.0, guard=1.0, min_events=2)
    for t in (12.0, 24.0, 36.0):
        a.observe_disturbance(t)
    assert a.learned
    assert abs(a.period - 12.0) < 1e-6
    assert abs(a.phase() - 0.0) < 1e-6                 # lurches fall on multiples of 12
    # predicts the NEXT multiple of the period at/after t — even from t=0 (a fresh
    # run), which is what lets it brace the very first lurch once the beat is known
    assert abs(a.next_predicted(0.0) - 0.0) < 1e-6
    assert abs(a.next_predicted(1.0) - 12.0) < 1e-6
    assert abs(a.next_predicted(13.0) - 24.0) < 1e-6


def test_should_protect_window():
    a = DisturbanceAnticipator(lead=2.0, guard=1.0, min_events=2)
    for t in (10.0, 20.0):
        a.observe_disturbance(t)
    # next disturbance at 30 -> protect window [28, 31]
    assert not a.should_protect(27.0)
    assert a.should_protect(28.5)
    assert a.should_protect(30.0)
    assert a.should_protect(30.9)
    assert not a.should_protect(31.5)


def test_not_learned_does_not_protect():
    a = DisturbanceAnticipator(min_events=2)
    a.observe_disturbance(5.0)                          # only one event
    assert not a.learned
    assert a.next_predicted(6.0) is None
    assert not a.should_protect(6.0)


def test_action_repertoire_picks_best():
    a = DisturbanceAnticipator()
    a.register_action("brace")
    a.register_action("widen_base")
    a.record_outcome("brace", saved=True)
    a.record_outcome("brace", saved=True)
    a.record_outcome("widen_base", saved=False)
    assert a.best_action() == "brace"
