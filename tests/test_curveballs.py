from agent.agent import solve

def test_mixed_units_curveball():
    q = "A projectile is launched horizontally from 9 meters with speed 10 mph; how far does it travel?"
    out = solve(q, preferred_units="SI")
    assert len(out.get("steps", [])) > 0

def test_impossible_negative_height():
    q = "From -3 meters height (nonsense), thrown at 10 m/s horizontally; what is the range?"
    out = solve(q, preferred_units="SI")
    assert out.get("trace_path","").endswith(".json")
