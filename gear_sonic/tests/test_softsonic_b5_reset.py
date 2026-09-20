"""B5 timing: real reset clears the rising-edge history once, per environment."""
import ast
from types import SimpleNamespace as NS
import torch
from softsonic_cpu_harness import BASELINE, close, forcefield_fixture, functions, source

EVENTS = 'gear_sonic/envs/manager_env/mdp/events.py'
COMMANDS = 'gear_sonic/envs/manager_env/mdp/commands.py'
tree = ast.parse(source(COMMANDS))
reset_blocks = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "hasattr(self._env, '_softsonic_last_k_ff')"]
assert len(reset_blocks) == 1
reset_code = compile(ast.Module(body=reset_blocks, type_ignores=[]), COMMANDS, 'exec')


def reset(env, ids):
    exec(reset_code, dict(self=NS(_env=env), env_ids=ids))


def run(revision):
    x = forcefield_fixture()
    event = functions(EVENTS, ['apply_softsonic_force_field'], revision).apply_softsonic_force_field
    # A previous episode already had an active field.
    x.env.episode_length_buf[:] = 8
    event(x.env, None)
    # Internal reset: episode counter is 0 during the interval event, then 1.
    x.robot.data.root_pos_w[:, 0] = .1
    x.env.episode_length_buf[:] = 0
    if revision is None:
        reset(x.env, torch.tensor([0]))
    event(x.env, None)
    initial = x.env._softsonic_ff_anchor_pos.clone()
    x.robot.data.root_pos_w[:, 0] += .01
    x.env.episode_length_buf[:] = 1
    event(x.env, None)
    return x.env._softsonic_ff_anchor_pos - initial


old, new = run(BASELINE), run(None)
close(old, [[.01, 0., 0.]], 1e-7)
close(new, 0.)
print('B5 step-0 -> step-1 anchor drift m before/after:', old.tolist(), new.tolist())
# A reset of one environment must not clear the other environments' history.
env = NS(_softsonic_last_k_ff=torch.tensor([100., 200.]))
reset(env, torch.tensor([0]))
close(env._softsonic_last_k_ff, [0., 200.])
# Initial reset is valid even before the first force event creates the buffers.
reset(NS(), torch.tensor([0]))
# First event after a global reset may arrive at counter 1 (not 0).
x = forcefield_fixture()
event = functions(EVENTS, ['apply_softsonic_force_field']).apply_softsonic_force_field
x.env.episode_length_buf[:] = 1
event(x.env, None)
initial = x.env._softsonic_ff_anchor_pos.clone()
x.robot.data.root_pos_w[:, 0] += .02
x.env.episode_length_buf[:] = 2
event(x.env, None)
close(x.env._softsonic_ff_anchor_pos, initial)
# New force events still re-anchor after an inactive interval.
x.ss[:, 7] = 0.
event(x.env, None)
x.ss[:, 7] = 100.
event(x.env, None)
close(x.env._softsonic_ff_anchor_pos, x.robot.data.root_pos_w)
print('B5 reset timing PASS (root-reference choice tested separately)')
