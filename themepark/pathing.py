from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot
from typing import Iterable

from themepark.config import GATE_X, GATE_Y


Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class PathNode:
    id: str
    x: float
    y: float


PATH_NODES: tuple[PathNode, ...] = (
    PathNode("gate", GATE_X, GATE_Y),
    PathNode("south_junction", 50.0, 90.0),
    PathNode("lower_left_junction", 25.0, 75.0),
    PathNode("lower_center_junction", 50.0, 75.0),
    PathNode("lower_right_junction", 75.0, 75.0),
    PathNode("middle_left_junction", 25.0, 50.0),
    PathNode("middle_center_junction", 50.0, 50.0),
    PathNode("middle_right_junction", 75.0, 50.0),
    PathNode("upper_left_junction", 25.0, 25.0),
    PathNode("upper_center_junction", 50.0, 25.0),
    PathNode("upper_right_junction", 75.0, 25.0),
    PathNode("attraction_0", 12.0, 15.0),
    PathNode("attraction_1", 50.0, 10.0),
    PathNode("attraction_2", 88.0, 15.0),
    PathNode("attraction_3", 12.0, 46.0),
    PathNode("attraction_4", 88.0, 46.0),
    PathNode("attraction_5", 12.0, 84.0),
    PathNode("attraction_6", 88.0, 84.0),
)


PATH_EDGES: tuple[tuple[str, str], ...] = (
    ("gate", "south_junction"),
    ("south_junction", "lower_center_junction"),
    ("lower_center_junction", "middle_center_junction"),
    ("middle_center_junction", "upper_center_junction"),
    ("lower_left_junction", "lower_center_junction"),
    ("lower_center_junction", "lower_right_junction"),
    ("middle_left_junction", "middle_center_junction"),
    ("middle_center_junction", "middle_right_junction"),
    ("upper_left_junction", "upper_center_junction"),
    ("upper_center_junction", "upper_right_junction"),
    ("upper_left_junction", "attraction_0"),
    ("upper_center_junction", "attraction_1"),
    ("upper_right_junction", "attraction_2"),
    ("middle_left_junction", "attraction_3"),
    ("middle_right_junction", "attraction_4"),
    ("lower_left_junction", "attraction_5"),
    ("lower_right_junction", "attraction_6"),
)


ATTRACTION_ENTRANCE_NODE_IDS: dict[int, str] = {
    0: "attraction_0",
    1: "attraction_1",
    2: "attraction_2",
    3: "attraction_3",
    4: "attraction_4",
    5: "attraction_5",
    6: "attraction_6",
}


class PathNetwork:
    def __init__(self) -> None:
        self.nodes: dict[str, Point] = {
            node.id: (node.x, node.y)
            for node in PATH_NODES
        }
        self.edges = PATH_EDGES
        self._neighbors: dict[str, list[tuple[str, float]]] = {
            node.id: []
            for node in PATH_NODES
        }
        for left, right in PATH_EDGES:
            distance = _distance(self.nodes[left], self.nodes[right])
            self._neighbors[left].append((right, distance))
            self._neighbors[right].append((left, distance))

    def route_to_attraction(self, start: Point, attraction_id: int) -> list[Point]:
        return self.route_to_node(start, ATTRACTION_ENTRANCE_NODE_IDS[attraction_id])

    def route_to_gate(self, start: Point) -> list[Point]:
        return self.route_to_node(start, "gate")

    def route_to_node(self, start: Point, end_node_id: str) -> list[Point]:
        start_node_id = self.nearest_node_id(start)
        node_path = self.shortest_node_path(start_node_id, end_node_id)
        points = [self.nodes[node_id] for node_id in node_path]
        if points and _distance(start, points[0]) < 1e-9:
            return points[1:]
        return points

    def shortest_node_path(self, start_node_id: str, end_node_id: str) -> list[str]:
        if start_node_id == end_node_id:
            return [start_node_id]
        distances = {start_node_id: 0.0}
        previous: dict[str, str] = {}
        queue: list[tuple[float, str]] = [(0.0, start_node_id)]
        visited: set[str] = set()
        while queue:
            current_distance, current = heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            if current == end_node_id:
                break
            for neighbor, edge_distance in self._neighbors[current]:
                candidate = current_distance + edge_distance
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    previous[neighbor] = current
                    heappush(queue, (candidate, neighbor))
        if end_node_id not in distances:
            raise ValueError(f"No path from {start_node_id} to {end_node_id}")
        path = [end_node_id]
        while path[-1] != start_node_id:
            path.append(previous[path[-1]])
        path.reverse()
        return path

    def nearest_node_id(self, point: Point) -> str:
        return min(self.nodes, key=lambda node_id: _distance(point, self.nodes[node_id]))

    def entrance_point(self, attraction_id: int) -> Point:
        return self.nodes[ATTRACTION_ENTRANCE_NODE_IDS[attraction_id]]

    def serializable_edges(self) -> list[dict[str, Point]]:
        return [
            {"from": self.nodes[left], "to": self.nodes[right]}
            for left, right in self.edges
        ]


def _distance(left: Point, right: Point) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


DEFAULT_PATH_NETWORK = PathNetwork()


def path_edges_payload() -> list[dict[str, dict[str, float]]]:
    return [
        {
            "from": {"x": start[0], "y": start[1]},
            "to": {"x": end[0], "y": end[1]},
        }
        for start, end in (
            (DEFAULT_PATH_NETWORK.nodes[left], DEFAULT_PATH_NETWORK.nodes[right])
            for left, right in DEFAULT_PATH_NETWORK.edges
        )
    ]


def attraction_entrances_payload() -> list[dict[str, float | int]]:
    return [
        {
            "attractionId": attraction_id,
            "x": DEFAULT_PATH_NETWORK.entrance_point(attraction_id)[0],
            "y": DEFAULT_PATH_NETWORK.entrance_point(attraction_id)[1],
        }
        for attraction_id in sorted(ATTRACTION_ENTRANCE_NODE_IDS)
    ]


def route_length(points: Iterable[Point]) -> float:
    total = 0.0
    previous: Point | None = None
    for point in points:
        if previous is not None:
            total += _distance(previous, point)
        previous = point
    return total
