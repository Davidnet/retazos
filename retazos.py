"""Basic circle packing algorithm based on 2 algorithms.

A1.0 by Huang:
https://home.mis.u-picardie.fr/~cli/Publis/circle.pdf and then
Smallest circle: Matoušek-Sharir-Welzl algorithm
https://beta.observablehq.com/@mbostock/miniball

Review by: David Cardozo <davidcardozo@berkeley.edu>
"""

import collections
import itertools
import math
import sys
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt

    def get_default_label(count, circle):
        if circle.ex and "id" in circle.ex:
            label = str(circle.ex["id"])
        elif circle.ex and "datum" in circle.ex:
            label = circle.ex["datum"]
        elif circle.ex:
            label = str(circle.ex)
        else:
            label = "#" + str(count)
        return label

    def _bubbles(circles, labels=None, lim=None):
        """Debugging function displays circles with matplotlib."""
        if not labels:
            labels = range(len(circles))
        _, ax = plt.subplots(figsize=(8.0, 8.0))
        for circle, label in zip(circles, labels):
            x, y, r = circle
            ax.add_patch(plt.Circle((x, y), r, alpha=0.2, linewidth=2, fill=False))
            ax.text(x, y, label)
        enclosure = enclose(circles)
        n = len(circles)
        if enclosure in circles:
            n = n - 1
        d = density(circles, enclosure)
        title = "{} circles packed for density {:.4f}".format(n, d)
        ax.set_title(title)
        if lim is None:
            lim = max(
                max(
                    abs(circle.x) + circle.r,
                    abs(circle.y) + circle.r,
                )
                for circle in circles
            )
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.show()

    def bubbles(circles, labels=None, lim=None):
        if not labels:
            labels = [get_default_label(i, c) for i, c in enumerate(circles)]
        return _bubbles([c.circle for c in circles], labels, lim)


except ImportError:
    pass

_eps = sys.float_info.epsilon
_Circle = collections.namedtuple("_Circle", ["x", "y", "r"])
FieldNames = collections.namedtuple("Field", ["id", "datum", "children"])


class Circle(object):
    """Hierarchy element.

    Used as an intermediate and output data structure.

    """

    __slots__ = ["circle", "level", "ex"]

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        r: float = 1.0,
        level: int = 1,
        ex: Dict = None,
    ):
        """Initialize Output data structure.

        Args:
            x (float): coorofdinate
            y (float): coordinate
            r (float): radius
            ex (dict): metadata

        """
        self.circle = _Circle(x, y, r)
        self.level = level
        self.ex = ex

    def __lt__(self, other) -> bool:
        return (self.level, self.r) < (other.level, other.r)

    def __eq__(self, other) -> bool:
        return (self.level, self.circle, self.ex) == (
            other.level,
            other.circle,
            other.ex,
        )

    def __repr__(self) -> str:
        return "{}(x={}, y={}, r={}, level={}, ex={!r})".format(
            self.__class__.__name__, self.x, self.y, self.r, self.level, self.ex
        )

    def __iter__(self) -> Iterator:
        return [self.x, self.y, self.r].__iter__()

    @property
    def x(self):
        return self.circle.x

    @property
    def y(self):
        return self.circle.y

    @property
    def r(self):
        return self.circle.r


def distance(circle1: Circle, circle2: Circle) -> float:
    """Compute distance between 2 cirlces."""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    x = x2 - x1
    y = y2 - y1
    return math.sqrt(x * x + y * y) - r1 - r2


def get_intersection(
    circle1: Circle, circle2: Circle
) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """Calculate intersections of 2 circles

    Based on https://gist.github.com/xaedes/974535e71009fa8f090e
    Credit to http://stackoverflow.com/a/3349134/798588

    Returns:
        2 pairs of coordinates. Each pair can be None if there are no or just
        one intersection.

    """
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    dx, dy = x2 - x1, y2 - y1
    d = math.sqrt(dx * dx + dy * dy)
    # Protect this part of the algo with try/except because edge cases
    # can lead to divizion by 0 or sqrt of negative numbers. Those indicate
    # that no intersection can be found and the debug log will show more info.
    try:
        a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
        h = math.sqrt(r1 * r1 - a * a)
    except (ValueError, ZeroDivisionError):
        eps = 1e-9
        if d > r1 + r2:
            print(f"no solution, the circles are separate: {circle1}, {circle2}")
        if d < abs(r1 - r2) + eps:
            print(
                "no solution, circles contained within each other: {circle1}, {circle2}",
            )
        if math.isclose(d, 0, abs_tol=eps) and math.isclose(
            r1, r2, rel_tol=0.0, abs_tol=eps
        ):
            print("Sans solution")
        return None, None
    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    xs1 = xm + h * dy / d
    xs2 = xm - h * dy / d
    ys1 = ym - h * dx / d
    ys2 = ym + h * dx / d
    if xs1 == xs2 and ys1 == ys2:
        return (xs1, ys1), None
    return (xs1, ys1), (xs2, ys2)


def get_placement_candidates(
    radius: float, c1: Circle, c2: Circle, margin: float
) -> Tuple[Circle, Circle]:
    """Generate placement candidates for 2 existing placed circles.
    Returns:
        A pair of candidate cirlces where one or both value can be None.

    """
    margin = radius * _eps * 10.0
    ic1 = _Circle(c1.x, c1.y, c1.r + (radius + margin))
    ic2 = _Circle(c2.x, c2.y, c2.r + (radius + margin))
    i1, i2 = get_intersection(ic1, ic2)
    if i1 is None:
        return None, None
    i1_x, i1_y = i1
    candidate1 = _Circle(i1_x, i1_y, radius)
    if i2 is None:
        return candidate1, None
    i2_x, i2_y = i2
    candidate2 = _Circle(i2_x, i2_y, radius)
    return candidate1, candidate2


def get_hole_degree_radius_w(candidate: Circle, circles: List[Circle]) -> float:
    """Calculate the hole degree of a candidate circle.

    Args:
        candidate: candidate circle.
        circles: sequence of circles.

    Returns:
        Squared euclidian distance of the candidate to the circles in argument.
        Each component of the distance is weighted by the inverse of the radius
        of the other circle to tilt the choice towards bigger circles.

    """
    return sum(distance(candidate, c) * c.r for c in circles)


def get_hole_degree_a1_0(candidate: Circle, circles: List[Circle]) -> float:
    """Calculate the hole degree of a candidate circle.

    Returns:
        minimum distance between the candidate and the circles in argument.

    In the paper, the hole degree defined as (1 - d_min / r_i) where d_min is
    a minimum disance between the candidate and the circles other than the one
    used to place the candidate and r_i the radius of the candidate.

    """
    return min(distance(candidate, c) for c in circles)


def get_hole_degree_density(candidate, circles) -> float:
    """Calculate the hole degree of a candidate circle.

    Returns:
        One minus the density of the configuration.

    """
    return 1.0 - density(circles + [candidate])


def place_new_A1_0(
    radius: float,
    const_placed_circles: List[Circle],
    get_hole_degree: Callable[[Circle, List[Circle]], float],
) -> float:
    """Place a new circle.

    Args:
        get_hole_degree: objective function to maximize.

    """
    placed_circles = const_placed_circles[
        :
    ]  # Make a new shallow copy TODO(Davidnet): Review
    n_circles = len(placed_circles)
    # If there are 2 or less, place circles on each side of (0, 0)
    if n_circles <= 1:
        x = radius if n_circles == 0 else -radius
        circle = _Circle(x, float(0.0), radius)
        placed_circles.append(circle)
        return placed_circles
    mhd = None
    lead_candidate = None
    for (c1, c2) in itertools.combinations(placed_circles, 2):
        margin = radius * _eps * 10.0
        # Placed circles other than the 2 circles used to find the
        # candidate placement.
        other_circles = [c for c in placed_circles if c not in (c1, c2)]
        for cand in get_placement_candidates(radius, c1, c2, margin):
            if cand is None:
                continue
            if not other_circles:
                lead_candidate = cand
                break
            # If overlaps with any, skip this candidate.
            if any(distance(c, cand) < 0.0 for c in other_circles):
                continue
            hd = get_hole_degree(cand, other_circles)
            assert hd is not None, "hole degree should not be None!"
            # If we were to use next_ we could use it here for look ahead.
            if mhd is None or hd < mhd:
                mhd = hd
                lead_candidate = cand
            if abs(mhd) < margin:
                break
    if lead_candidate is None:
        # The radius is set to sqrt(value) in pack_A1_0
        raise ValueError("cannot place circle for value " + str(radius ** 2))
    placed_circles.append(lead_candidate)
    return placed_circles


def pack_A1_0(data: List[float]) -> Union[float, List[float]]:
    """Pack circles whose area is proportional to the input data.

    This algorithm is based on the Huang et al. heuristic.

    """
    min_max_ratio = min(data) / max(data)
    if min_max_ratio < _eps:
        print(
            f"min to max ratio is too low at {min_max_ratio} and it could cause algorithm stability issues. Try to remove insignificant data",
        )
    assert data == sorted(data, reverse=True), "data must be sorted (desc)"
    placed_circles = []
    radiuses = [math.sqrt(value) for value in data]
    # TODO(Davidnet): next_ lookahead not used.
    for radius, next_ in look_ahead(radiuses):
        placed_circles = place_new_A1_0(
            radius, placed_circles, get_hole_degree_radius_w
        )
    return placed_circles


def extendBasis(B, p):
    """Extend basis to ...  """
    if enclosesWeakAll(p, B):
        return [p]

    # If we get here then B must have at least one element.
    for bel in B:
        if enclosesNot(p, bel) and enclosesWeakAll(encloseBasis2(bel, p), B):
            return [bel, p]

    # If we get here then B must have at least two elements.
    for i in range(len(B) - 1):
        for j in range(i + 1, len(B)):
            if (
                enclosesNot(encloseBasis2(B[i], B[j]), p)
                and enclosesNot(encloseBasis2(B[i], p), B[j])
                and enclosesNot(encloseBasis2(B[j], p), B[i])
                and enclosesWeakAll(encloseBasis3(B[i], B[j], p), B)
            ):
                return [B[i], B[j], p]
    raise ValueError("Invalid!")


def enclosesNot(a: Circle, b: Circle) -> bool:
    dr = a.r - b.r
    dx = b.x - a.x
    dy = b.y - a.y
    return dr < 0 or dr * dr < dx * dx + dy * dy


def enclosesWeak(a: Circle, b: Circle) -> bool:
    dr = a.r - b.r + 1e-6
    dx = b.x - a.x
    dy = b.y - a.y
    return dr > 0 and dr * dr > dx * dx + dy * dy


def enclosesWeakAll(a: Circle, B: List[Circle]) -> bool:
    for bel in B:
        if not enclosesWeak(a, bel):
            return False
    return True


def encloseBasis(B: List[Circle]) -> _Circle:
    if len(B) == 1:
        return B[0]
    elif len(B) == 2:
        return encloseBasis2(B[0], B[1])
    else:
        return encloseBasis3(B[0], B[1], B[2])


def encloseBasis2(a: Circle, b: Circle) -> _Circle:
    x1, y1, r1 = a.x, a.y, a.r
    x2, y2, r2 = b.x, b.y, b.r
    x21 = x2 - x1
    y21 = y2 - y1
    r21 = r2 - r1
    l21 = math.sqrt(x21 * x21 + y21 * y21)
    return _Circle(
        (x1 + x2 + x21 / l21 * r21) / 2,
        (y1 + y2 + y21 / l21 * r21) / 2,
        (l21 + r1 + r2) / 2,
    )


def encloseBasis3(a: Circle, b: Circle, c: Circle) -> _Circle:
    x1, y1, r1 = a.x, a.y, a.r
    x2, y2, r2 = b.x, b.y, b.r
    x3, y3, r3 = c.x, c.y, c.r
    a2 = x1 - x2
    a3 = x1 - x3
    b2 = y1 - y2
    b3 = y1 - y3
    c2 = r2 - r1
    c3 = r3 - r1
    d1 = x1 * x1 + y1 * y1 - r1 * r1
    d2 = d1 - x2 * x2 - y2 * y2 + r2 * r2
    d3 = d1 - x3 * x3 - y3 * y3 + r3 * r3
    ab = a3 * b2 - a2 * b3
    xa = (b2 * d3 - b3 * d2) / (ab * 2) - x1
    xb = (b3 * c2 - b2 * c3) / ab
    ya = (a3 * d2 - a2 * d3) / (ab * 2) - y1
    yb = (a2 * c3 - a3 * c2) / ab
    A = xb * xb + yb * yb - 1
    B = 2 * (r1 + xa * xb + ya * yb)
    C = xa * xa + ya * ya - r1 * r1
    if A != 0.0:
        r = -(B + math.sqrt(B * B - 4 * A * C)) / (2 * A)
    else:
        r = -C / B
    return _Circle(x1 + xa + xb * r, y1 + ya + yb * r, r)


def enclose(circles: List[Circle]) -> Optional[_Circle]:
    """Pastel from the adapted implementation from d3js.

    See https://github.com/d3/d3-hierarchy/blob/master/src/pack/enclose.js

    """
    B = []
    p, e = None, None
    # random.shuffle(circles)

    n = len(circles)
    i = 0
    while i < n:
        p = circles[i]
        if e is not None and enclosesWeak(e, p):
            i = i + 1
        else:
            B = extendBasis(B, p)
            e = encloseBasis(B)
            i = 0
    return e


def scale(circle: Circle, target: Circle, enclosure: _Circle) -> _Circle:
    """Scale circle in enclosure to fit in the target circle.

    Args:
        circle: Circle to scale.
        target: target Circle to scale to.
        enclosure: allows one to specify the enclosure.

    Returns:
        scaled circle

    """
    r = target.r / enclosure.r
    t_x, t_y = target.x, target.y
    e_x, e_y = enclosure.x, enclosure.y
    c_x, c_y, c_r = circle
    return _Circle((c_x - e_x) * r + t_x, (c_y - e_y) * r + t_y, c_r * r)


def density(circles: List[Circle], enclosure: Optional[Circle] = None) -> float:
    """Computes the relative density of te packed circles.

    Args:
        circles: packed list of circles.

    Return:
        Compute the enclosure if not passed as argument and calculates the
        relative area of the sum of the inner cirlces to the area of the
        enclosure.

    """
    if not enclosure:
        enclosure = enclose(circles)
    return sum(c.r ** 2.0 for c in circles if c != enclosure) / enclosure.r ** 2.0


def look_ahead(iterable: Iterator, n_elems: int = 1) -> Iterator:
    """Fetch look ahead elements of data

    From https://stackoverflow.com/questions/4197805/python-for-loop-look-ahead

    """
    items, nexts = itertools.tee(iterable, 2)
    nexts = itertools.islice(nexts, n_elems, None)
    return itertools.zip_longest(items, nexts)


def _handle(data: Union[dict, float], level: int, fields=None) -> List[Circle]:
    """Converts possibly heterogeneous list of float or dict in list of Output."""
    if fields is None:
        fields = FieldNames(None, None, None)
    datum_field = fields.datum if fields.datum else "datum"
    elements = []
    for datum in data:
        if isinstance(datum, dict):
            value = datum[datum_field]
            elements.append(Circle(r=value + 0, level=level, ex=datum))
            continue
        if datum <= 0.0:
            raise ValueError("input data must be positive. Found " + str(datum))
        if datum <= _eps:
            print(
                "input data {datum} is small and could cause stability issues. Can you scale the data set up or drop insignificant elements?"
            )
        try:
            elements.append(Circle(r=datum + 0, level=level, ex={"datum": datum}))
        except TypeError:  # if it fails, assume dict.
            raise TypeError("dict or numeric value expected")
    return sorted(elements, reverse=True)


def _circlify_level(
    data: Union[Dict, float], target_enclosure: Circle, fields, level: int = 1
) -> List:
    all_circles = []
    if not data:
        return all_circles
    circles = _handle(data, 1, fields)
    packed = pack_A1_0([circle.r for circle in circles])
    enclosure = enclose(packed)
    assert enclosure is not None
    for circle, inner_circle in zip(circles, packed):
        circle.level = level
        circle.circle = scale(inner_circle, target_enclosure, enclosure)
        if circle.ex and fields.children in circle.ex:
            all_circles += _circlify_level(
                circle.ex[fields.children], circle.circle, fields, level + 1
            )
        all_circles.append(circle)
    return all_circles


def _flatten(elements, flattened):
    """Flattens elems."""
    # TODO(Davidnet): Functional?
    if elements is None:
        return
    for elem in elements:
        _flatten(elem.children, flattened)
        elem.children = None
        flattened.append(elem)
    return flattened


def compute_retazo(
    data: List,
    target_enclosure: Optional[Circle] = None,
    show_enclosure=False,
    datum_field="datum",
    id_field="id",
    children_field="children",
):
    """Pack and enclose circles.

    Args:
        data: sorted (descending) array of values.
        show_enclosure: insert the target enclosure to the output if True.
        datum_field: metadata field specification
        id_field: field name that contains the id when the element is a dict.
        children_field: dict field name

    Returns:
        list of Circle whose *area* is proportional to the
        corresponding input value.  The list is sorted by ascending level
        (root to leaf) and descending value (biggest circles first).

    """
    fields = FieldNames(id=id_field, datum=datum_field, children=children_field)
    if target_enclosure is None:
        target_enclosure = Circle(level=0, x=0.0, y=0.0, r=1.0)
    all_circles = _circlify_level(data, target_enclosure, fields)
    if show_enclosure:
        all_circles.append(target_enclosure)
    return sorted(all_circles)