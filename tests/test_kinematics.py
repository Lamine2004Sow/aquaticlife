import numpy as np

from aquaticlife.physics.body import MorphologyParameters, RigidSegmentChain


def make_morpho():
    return MorphologyParameters(
        num_segments=2,
        lengths=np.array([0.2, 0.2]),
        masses=np.array([0.2, 0.2]),
        joint_stiffness=np.array([1.0]),
        joint_damping=np.array([0.1]),
        muscle_strength=np.array([1.0]),
    )


def test_forward_kinematics_straight():
    body = RigidSegmentChain(make_morpho())
    pos = body.forward_kinematics()
    assert np.allclose(pos[0], [0.1, 0.0], atol=1e-6)
    assert np.allclose(pos[1], [0.3, 0.0], atol=1e-6)


def test_forward_kinematics_bent():
    body = RigidSegmentChain(make_morpho())
    body.angles[0] = np.pi / 2
    pos = body.forward_kinematics()
    assert np.allclose(pos[1][0], 0.2, atol=1e-6)
    assert np.allclose(pos[1][1], 0.1, atol=1e-6)
