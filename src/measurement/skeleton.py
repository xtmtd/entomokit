from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Skeletonization backed by scikit-image."""
    return sk_skeletonize(mask > 0).astype(np.uint8)


def _neighbors(y: int, x: int) -> list[tuple[int, int, float]]:
    return [
        (y - 1, x - 1, math.sqrt(2.0)),
        (y - 1, x, 1.0),
        (y - 1, x + 1, math.sqrt(2.0)),
        (y, x - 1, 1.0),
        (y, x + 1, 1.0),
        (y + 1, x - 1, math.sqrt(2.0)),
        (y + 1, x, 1.0),
        (y + 1, x + 1, math.sqrt(2.0)),
    ]


def skeleton_graph(
    skel: np.ndarray,
) -> dict[tuple[int, int], dict[tuple[int, int], float]]:
    h, w = skel.shape
    graph: dict[tuple[int, int], dict[tuple[int, int], float]] = defaultdict(dict)
    ys, xs = np.where(skel > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        node = (y, x)
        for ny, nx, weight in _neighbors(y, x):
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] > 0:
                graph[node][(ny, nx)] = weight
    return graph


def _degree(graph, node: tuple[int, int]) -> int:
    return len(graph.get(node, {}))


def _remove_node(graph, node: tuple[int, int]) -> None:
    for n in list(graph.get(node, {}).keys()):
        graph[n].pop(node, None)
    graph.pop(node, None)


def prune_short_branches(
    graph: dict[tuple[int, int], dict[tuple[int, int], float]],
    min_len: float,
) -> dict[tuple[int, int], dict[tuple[int, int], float]]:
    """Prune endpoint branches shorter than min_len up to nearest junction."""
    changed = True
    while changed:
        changed = False
        endpoints = [n for n in graph if _degree(graph, n) == 1]
        for start in endpoints:
            if start not in graph or _degree(graph, start) != 1:
                continue

            path = [start]
            length = 0.0
            prev = None
            cur = start
            while True:
                nbrs = [n for n in graph[cur] if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                length += graph[cur][nxt]
                path.append(nxt)
                prev, cur = cur, nxt
                deg = _degree(graph, cur)
                if deg != 2:
                    break

            if _degree(graph, cur) >= 3 and length < min_len:
                for node in path[:-1]:
                    if _degree(graph, node) <= 2:
                        _remove_node(graph, node)
                        changed = True
    return graph


def _connected_components(graph) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for node in graph:
        if node in seen:
            continue
        q = deque([node])
        seen.add(node)
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def _dijkstra(graph, start: tuple[int, int], nodes: set[tuple[int, int]]):
    import heapq

    dist = {start: 0.0}
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]

    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, float("inf")):
            continue
        for nxt, w in graph[node].items():
            if nxt not in nodes:
                continue
            nd = d + w
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                prev[nxt] = node
                heapq.heappush(heap, (nd, nxt))
    return dist, prev


def longest_backbone_path(
    graph: dict[tuple[int, int], dict[tuple[int, int], float]],
) -> tuple[list[tuple[int, int]], float]:
    if not graph:
        return [], 0.0

    best_path: list[tuple[int, int]] = []
    best_len = 0.0

    for comp in _connected_components(graph):
        nodes = set(comp)
        endpoints = [n for n in comp if len([x for x in graph[n] if x in nodes]) <= 1]
        seeds = endpoints if endpoints else [comp[0]]
        seed = seeds[0]

        dist1, _ = _dijkstra(graph, seed, nodes)
        far_a = max(dist1, key=dist1.get)

        dist2, prev = _dijkstra(graph, far_a, nodes)
        far_b = max(dist2, key=dist2.get)
        plen = dist2.get(far_b, 0.0)

        path = []
        cur = far_b
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()

        if plen > best_len:
            best_len = plen
            best_path = path

    return best_path, best_len


def backbone_mask(shape: tuple[int, int], path: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    for y, x in path:
        out[y, x] = 1
    return out
