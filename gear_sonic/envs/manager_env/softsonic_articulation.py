"""World-frame zero-order wrench hold on IsaacLab's physical-step clock.

SoftMimic's forcefield is expressed in world coordinates. IsaacLab's permanent
composer instead caches a link-frame conversion until reset (review B1). Keep
our world wrench separate and convert using CURRENT link orientation on EVERY
physics step, then use the upstream instantaneous composer lifecycle. The
spring is still evaluated once per control step; this does not introduce a new
physics-rate spring controller or edit shared IsaacLab code.
"""
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse
import torch


class SoftSONICArticulation(Articulation):
    """Articulation with one replaceable SoftSONIC world-wrench producer."""

    def set_softsonic_world_wrench(self, forces: torch.Tensor, torques: torch.Tensor) -> None:
        # Whole-batch replacement clears released/changed-body events. Cloning
        # prevents callers mutating the hold buffer during a physics interval.
        self._softsonic_force_w = forces.clone()
        self._softsonic_torque_w = torques.clone()

    def reset(self, env_ids=None):
        # IsaacLab scene.reset calls this for exactly the reset environments.
        # Preserve held forces on other environments during asynchronous reset.
        if hasattr(self, "_softsonic_force_w"):
            ids = slice(None) if env_ids is None else env_ids
            self._softsonic_force_w[ids] = 0
            self._softsonic_torque_w[ids] = 0
        super().reset(env_ids)

    def write_data_to_sim(self):
        if hasattr(self, "_softsonic_force_w"):
            # Use link, not COM, orientation: upstream WrenchComposer's output
            # and Articulation.write_data_to_sim both use the link frame.
            quat = self.data.body_link_quat_w
            force_b = quat_apply_inverse(quat, self._softsonic_force_w)
            torque_b = quat_apply_inverse(quat, self._softsonic_torque_w)
            # is_global=False avoids stale pose caches even if another producer
            # touched this composer already. Add preserves its contribution;
            # upstream super writes all wrenches and resets instantaneous data.
            self.instantaneous_wrench_composer.add_forces_and_torques(
                forces=force_b, torques=torque_b, is_global=False,
            )
        super().write_data_to_sim()
