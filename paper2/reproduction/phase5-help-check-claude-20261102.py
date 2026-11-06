"""Why phase 5 missed: replay the recorded actions of the learning condition, and re-fly the reference pilot with and
without its knowledge of how an orbit changes. Audited streams only; no model is called."""
import gzip, json, math, sys, glob
from pathlib import Path
from paper2.experiment2 import study_runner as S
from telohelm.taskpacks.tanker.world import episode as E, physics as P, phase5 as F, pilot as PL

ROOTS = sys.argv[1:]
rows_out = []
original = PL.OrbitModel.inferred_delta
for root in map(Path, ROOTS):
    body = json.loads((root / 'schedule.json').read_text())['schedule']
    for r in [x for x in body['rows'] if x['phase'] == 5]:
        sc = r['scenario']; o = sc['tankers'][0]['orbit']; ep_id = r['episode']
        orbit = P.Orbit(o['cx'], o['cy'], o['r0'], o['v'], o['theta0'], o['delta'])
        s = json.loads((root / 'arms/L' / ep_id / 'summary.json').read_text()); out = s['outcome']
        help_t = out['help_requests'][0]['t'] if out['help_requests'] else None
        reply_t = help_t + 30 if help_t is not None else None
        frac = 0.0 if reply_t is None or reply_t <= 60 else min(1.0, (1 - math.cos(math.pi * (reply_t - 60) / 120)) / 2)
        ep = E.Episode(S._scenario(sc)); start = ep.reset(); bundle = start['bundle']
        acts = {a['t']: a['actions'] for a in out['actions']}
        best, last, t = 1e9, None, 0
        while bundle is not None:
            own = bundle.get('own') or bundle
            x, y = own['x'], own['y']
            px, py = orbit.position(float(bundle['t']))
            d = math.hypot(px - x, py - y); best = min(best, d); last = d; t = bundle['t']
            bundle = ep.step(acts.get(t, {}))
        PL.OrbitModel.inferred_delta = original
        scen = S._scenario(sc)
        withprior = F.fly(scen, ask_help=True, max_steps=out['goal']['deadline'] + 2)
        PL.OrbitModel.inferred_delta = lambda self: 0.0
        noprior = F.fly(S._scenario(sc), ask_help=True, max_steps=out['goal']['deadline'] + 2)
        PL.OrbitModel.inferred_delta = original
        rows_out.append({'episode': ep_id, 'delta_nm': round(o['delta'], 3), 'blackout_t': sc['dropouts'][0][1], 'help_t': help_t, 'reply_t': reply_t,
                         'frac_at_reply': round(frac, 4), 'radial_offset_at_reply_nm': round(abs(o['delta']) * frac, 3), 'deadline': out['goal']['deadline'],
                         'L_closest_nm': round(best, 3), 'L_range_at_end_nm': round(last, 3), 'L_joined': out['joined'], 'L_report_ok': out['report_correct'],
                         'pilot_with_prior_goal': withprior['goal_met'], 'pilot_no_prior_goal': noprior['goal_met'], 'pilot_no_prior_join': noprior['joined']})
print(json.dumps(rows_out))
