from src.datapipeline.schema import AgentClass, AgentState, AgentTrack, SceneContext
from src.generator.mock import MockVariantGenerator


def _sample_scene() -> SceneContext:
    history = [AgentState(x=float(i), y=0.0, vx=1.0, vy=0.0) for i in range(5)]
    future = [AgentState(x=float(5 + i), y=0.0, vx=1.0, vy=0.0) for i in range(4)]
    agent = AgentTrack("a1", AgentClass.VEHICLE, history, future)
    return SceneContext(
        scene_id="test-scene",
        dt=0.25,
        horizon_steps=4,
        agents=[agent],
        map_bounds={"xmin": -10, "xmax": 10, "ymin": -10, "ymax": 10},
    )


def test_generate_returns_requested_variant_count():
    gen = MockVariantGenerator(seed=42)
    scene = _sample_scene()
    variants = gen.generate(scene, n_variants=3)
    assert len(variants) == 3


def test_generate_covers_all_agents_with_correct_horizon():
    gen = MockVariantGenerator(seed=42)
    scene = _sample_scene()
    variants = gen.generate(scene, n_variants=1)
    futures = variants[0].agent_futures
    assert set(futures.keys()) == {"a1"}
    assert len(futures["a1"]) == scene.horizon_steps


def test_generate_without_ground_truth_future_extrapolates():
    gen = MockVariantGenerator(seed=1)
    history = [AgentState(x=float(i), y=0.0, vx=1.0, vy=0.0) for i in range(5)]
    agent = AgentTrack("a1", AgentClass.PEDESTRIAN, history, future=[])
    scene = SceneContext(
        scene_id="no-gt",
        dt=0.25,
        horizon_steps=6,
        agents=[agent],
        map_bounds={"xmin": -10, "xmax": 10, "ymin": -10, "ymax": 10},
    )
    variants = gen.generate(scene, n_variants=1)
    assert len(variants[0].agent_futures["a1"]) == 6
