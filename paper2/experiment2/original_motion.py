"""Conditional original motion for B8; no draws or admissible labels.

The constant adapter is analytic. The separate changing-radius adapter retains
quadrature work/error diagnostics, which do not certify its numerical error.
Its frozen numerical inputs and claimed original identity are retained together.
A future protected verifier must join them to the original case. This module
cannot establish that a caller's state or fingerprints are authoritative.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from .monitor_labels import Original; from .monitor_labels import Bounds; from .monitor_labels import propose_heading
from .contact_bounds import contact_lower; from .contact_bounds import Settings; from .contact_bounds import LowerEvidence

def _finite(x, name):
    try:
        valid = type(x) in (int, float) and math.isfinite(x)
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f'{name} must be finite')
    return float(x)


@dataclass(frozen=True)
class Receiver:
    second: float
    x: float
    y: float
    heading_radians: float
    tas_nm_s: float
    turn_rate_rad_s: float


@dataclass(frozen=True)
class CircularTanker:
    centre_x: float
    centre_y: float
    radius_nm: float
    speed_nm_s: float
    angle_at_zero_radians: float


@dataclass(frozen=True)
class FrozenMotion:
    original: Original
    receiver: Receiver
    tanker: CircularTanker
    wind_nm_s: tuple[float, float]
    commanded_heading_deg: float
    fingerprint: str
    tie_band_radians: float = 1e-9  # Existing D17 convention, frozen in this record.
    qualification: str = 'conditional_original_state_and_motion_error_assumptions'
    admissible: bool = False

    @classmethod
    def bind(cls, original, receiver, tanker, wind_nm_s, commanded_heading_deg):
        if not isinstance(original, Original) or not isinstance(receiver, Receiver) \
                or not isinstance(tanker, CircularTanker):
            raise TypeError('Original, Receiver and CircularTanker required')
        check = propose_heading(original, Bounds(original.second))
        if check.disposition == 'invalid' or original.second > 3600:
            raise ValueError('invalid original identity or time')
        r = Receiver(**{k:_finite(v,k) for k,v in asdict(receiver).items()})
        t = CircularTanker(**{k:_finite(v,k) for k,v in asdict(tanker).items()})
        if r.second != original.second or not 0 <= r.heading_radians < math.tau:
            raise ValueError('receiver must match the original second and heading convention')
        if not 0 < r.tas_nm_s <= 480/3600 or r.turn_rate_rad_s <= 0:
            raise ValueError('receiver speed or turn rate outside declared domain')
        if t.radius_nm <= 0 or not 0 <= t.speed_nm_s <= 300/3600:
            raise ValueError('invalid constant tanker orbit')
        if not isinstance(wind_nm_s,(tuple,list)) or len(wind_nm_s) != 2:
            raise ValueError('two wind coordinates required')
        wind = tuple(_finite(x,'wind') for x in wind_nm_s)
        if math.hypot(*wind) > 60/3600:
            raise ValueError('wind exceeds declared domain')
        command = _finite(commanded_heading_deg,'command')
        if not 0 <= command < 360:
            raise ValueError('command must lie in [0,360)')
        payload = {'rule':'B8.constant.motion.v1', 'original':asdict(original),
                   'receiver':asdict(r), 'tanker':asdict(t), 'wind_nm_s':wind,
                   'commanded_heading_deg':command, 'tie_band':1e-9}
        digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),
                                           allow_nan=False).encode()).hexdigest()
        return cls(original,r,t,wind,command,digest)

    def _time(self, second):
        s = _finite(second,'query second')
        if not self.original.second <= s <= 3600:
            raise ValueError('query outside original-to-deadline interval')
        return s

    @property
    def receiver_speed_bound_nm_s(self):
        return self.receiver.tas_nm_s + math.hypot(*self.wind_nm_s)

    @property
    def tanker_speed_bound_nm_s(self):
        return self.tanker.speed_nm_s

    def tanker_at(self, second):
        s = self._time(second)
        t = self.tanker
        a = t.angle_at_zero_radians + t.speed_nm_s*s/t.radius_nm
        return t.centre_x+t.radius_nm*math.sin(a), t.centre_y+t.radius_nm*math.cos(a)

    def receiver_at(self, second):
        """Exact analytic path under one held command, conditional on floating error.

        Clockwise-from-north convention matches world/physics.py. A right turn
        inside the half-turn tie band may exceed pi; do not end it early.
        """
        s = self._time(second)
        r = self.receiver
        dt, h, command = s-r.second, r.heading_radians, math.radians(self.commanded_heading_deg)
        diff = (command-h+math.pi) % math.tau-math.pi
        direction = 0 if diff == 0 else (1 if abs(abs(diff)-math.pi)<=self.tie_band_radians or diff>0 else -1)
        remaining = 0 if direction == 0 else (diff if direction>0 and diff>0 else
                     diff+math.tau if direction>0 else -diff)
        turning = min(dt,remaining/r.turn_rate_rad_s)
        wx,wy = self.wind_nm_s
        x,y = r.x,r.y
        if direction:
            end = h+direction*r.turn_rate_rad_s*turning
            scale = r.tas_nm_s/(direction*r.turn_rate_rad_s)
            x += scale*(math.cos(h)-math.cos(end))+wx*turning
            y += scale*(math.sin(end)-math.sin(h))+wy*turning
            h = end if dt <= remaining/r.turn_rate_rad_s else command
        straight = dt-turning
        return x+(r.tas_nm_s*math.sin(h)+wx)*straight, y+(r.tas_nm_s*math.cos(h)+wy)*straight

    def lower(self, deadline_second, *, prefix_seconds=1, settings=Settings()):
        return contact_lower(self.original,self.tanker_at,self.receiver_at,
                             self.receiver_speed_bound_nm_s,self.tanker_speed_bound_nm_s,
                             deadline_second,prefix_seconds=prefix_seconds,settings=settings)


@dataclass(frozen=True)
class ChangingTanker:
    centre_x: float
    centre_y: float
    radius_nm: float
    speed_nm_s: float
    angle_at_zero_radians: float
    delta_nm: float
    change_start_second: float
    change_duration_seconds: float


@dataclass(frozen=True)
class TankerQuery:
    second: float
    point: tuple[float, float] | None
    angular_error_estimate: float | None
    position_error_estimate_nm: float | None
    integrand_evaluations: int
    state: str
    error: str | None = None


@dataclass(frozen=True)
class ChangingLowerEvidence:
    contact: LowerEvidence
    motion_fingerprint: str
    tanker_queries: tuple[TankerQuery, ...]
    integrand_evaluations: int
    qualification: str = 'conditional_original_motion_and_uncertified_quadrature_error'
    admissible: bool = False


@dataclass(frozen=True)
class ChangingMotion(FrozenMotion):
    """Half-cosine radius change at constant total tanker speed; constant wind.

    Original clocks are absolute. Receiver propagation remains analytic and
    uses the frozen D17 tie. Query counts are finite, not a wall-time bound.
    """
    tanker: ChangingTanker
    quadrature_evaluation_cap: int = 4096
    qualification: str = 'conditional_original_motion_and_uncertified_quadrature_error'

    @classmethod
    def bind(cls, original, receiver, tanker, wind_nm_s, commanded_heading_deg):
        if not isinstance(tanker, ChangingTanker):
            raise TypeError('ChangingTanker required; do not discard change fields')
        t = ChangingTanker(**{k:_finite(v,k) for k,v in asdict(tanker).items()})
        # Reuse the accepted receiver/original/wind/command domain checks, but
        # retain all change fields in this record and its separate fingerprint.
        circle = CircularTanker(t.centre_x,t.centre_y,t.radius_nm,t.speed_nm_s,t.angle_at_zero_radians)
        base = FrozenMotion.bind(original,receiver,circle,wind_nm_s,commanded_heading_deg)
        if t.radius_nm+t.delta_nm <= 0 or t.change_duration_seconds <= 0 \
                or t.change_start_second < 0 or t.change_start_second+t.change_duration_seconds > 3600:
            raise ValueError('change requires positive radius and duration inside the episode')
        radial_peak = abs(t.delta_nm)*math.pi/(2*t.change_duration_seconds)
        if not math.isfinite(radial_peak) or radial_peak > t.speed_nm_s:
            raise ValueError('radial rate exceeds total tanker speed')
        payload = {'rule':'B8.changing.motion.v1','original':asdict(base.original),
                   'receiver':asdict(base.receiver),'tanker':asdict(t),'wind_nm_s':base.wind_nm_s,
                   'commanded_heading_deg':base.commanded_heading_deg,'tie_band':base.tie_band_radians,
                   'quadrature':{'epsabs':1e-13,'epsrel':1e-13,'limit':200,'evaluation_cap':4096}}
        digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),
                                           allow_nan=False).encode()).hexdigest()
        return cls(base.original,base.receiver,t,base.wind_nm_s,base.commanded_heading_deg,digest)

    def tanker_query(self, second):
        """Retain success, cap or error diagnostics for one absolute-time query.

        SciPy's abserr is an estimate. radius*abserr translates just that angle
        estimate to a nominal position estimate; it excludes radius/trig/error
        propagation and is never substituted for a qualified distance allowance.
        """
        from scipy.integrate import quad
        s = self._time(second)
        t = self.tanker
        start,end = t.change_start_second,t.change_start_second+t.change_duration_seconds
        count = 0
        class QueryCap(Exception):
            pass
        def radius(u):
            return t.radius_nm+t.delta_nm*(1-math.cos(math.pi*(u-start)/t.change_duration_seconds))/2
        def rate(u):
            nonlocal count
            if count >= self.quadrature_evaluation_cap:
                raise QueryCap('quadrature integrand evaluation cap')
            count += 1
            radial = t.delta_nm*math.pi/(2*t.change_duration_seconds)*math.sin(math.pi*(u-start)/t.change_duration_seconds)
            return math.sqrt(t.speed_nm_s*t.speed_nm_s-radial*radial)/radius(u)
        try:
            if s <= start or t.delta_nm == 0:
                r,a,err = t.radius_nm,t.angle_at_zero_radians+t.speed_nm_s*s/t.radius_nm,0.0
            else:
                result = quad(rate,start,min(s,end),epsabs=1e-13,epsrel=1e-13,limit=200,full_output=1)
                if len(result) != 3:
                    raise ValueError('quadrature did not converge: '+str(result[3])[:200])
                during,err,info = result
                during,err = _finite(during,'angle integral'),_finite(err,'quadrature error estimate')
                if err < 0 or info.get('neval') != count:
                    raise ValueError('invalid quadrature diagnostics')
                r = radius(s) if s < end else t.radius_nm+t.delta_nm
                a = t.angle_at_zero_radians+t.speed_nm_s*start/t.radius_nm+during
                if s > end:
                    a += t.speed_nm_s*(s-end)/r
            point = (t.centre_x+r*math.sin(a),t.centre_y+r*math.cos(a))
            point = tuple(_finite(x,'tanker coordinate') for x in point)
            estimate = _finite(r*err,'position error estimate')
            return TankerQuery(s,point,err,estimate,count,'ok')
        except Exception as error:
            return TankerQuery(s,None,None,None,count,'evaluation_cap' if isinstance(error,QueryCap) else 'error',
                               f'{type(error).__name__}: {str(error)[:200]}')

    def tanker_at(self, second):
        q = self.tanker_query(second)
        if q.state != 'ok':
            raise ValueError(q.error)
        return q.point

    def lower(self, deadline_second, *, prefix_seconds=1, settings=Settings()):
        queries = []
        def point(s):
            q = self.tanker_query(s)
            queries.append(q)
            if q.state != 'ok':
                raise ValueError(q.error)
            return q.point
        evidence = contact_lower(self.original,point,self.receiver_at,self.receiver_speed_bound_nm_s,
                                 self.tanker_speed_bound_nm_s,deadline_second,
                                 prefix_seconds=prefix_seconds,settings=settings)
        return ChangingLowerEvidence(evidence,self.fingerprint,tuple(queries),
                                     sum(q.integrand_evaluations for q in queries))
