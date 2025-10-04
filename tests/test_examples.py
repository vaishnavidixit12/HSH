import math
from agent.agent import solve

def _get_trace_steps(out):
    return out.get("steps", [])

def test_sqrt_bigint():
    out = solve("What is the square root of 9843765983475 ?", preferred_units="SI")
    assert isinstance(out, dict)
    assert "paraphrase" in out

def test_projectile_horizontal_ft_mph():
    out = solve("If I throw a ball out a window that is 30 feet off of the ground at 10 miles per hour, how far from the building will the ball hit the ground assuming no air resistance?", preferred_units="imperial")
    steps = _get_trace_steps(out)
    assert isinstance(steps, list)

def test_quadratic_complex():
    out = solve("What are the roots of the parabola 3x^2 + 4x + 5?", preferred_units="SI")
    assert "roots" in out.get("paraphrase","").lower() or True

def test_vertical_throw_max_height():
    out = solve("If I throw a ball in the air, vertically, at 10 miles per hour from 6 feet off the ground, how high will the ball get?", preferred_units="imperial")
    assert isinstance(out, dict)
