from __future__ import annotations
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator
from heapq import heappop, heappush
from typing import Callable
from sys import maxsize
from math import gcd, isqrt, prod
from functools import reduce
from itertools import compress, count


# Dijkstra
def dijkstra(graph: list[list[tuple[int, int]]]) -> list[int]:
    n = len(graph)
    queue: list[tuple[int, int]] = [(0, 0)]
    dp: list[int] = [maxsize] * n
    dp[0] = 0

    while queue:
        cost, current = heappop(queue)
        if cost != dp[current]:
            continue
        for edge, c in graph[current]:
            new_cost = cost + c
            if new_cost >= dp[edge]:
                continue
            heappush(queue, (new_cost, edge))
            dp[edge] = new_cost
    return dp


# Kruskal
def kruskal(edges: list[tuple[int, int, int]], n: int) -> list[tuple[int, int, int]]:
    edges.sort(key=lambda x: x[2])
    p = [*range(n)]
    result: list[tuple[int, int, int]] = []
    for a, b, c in edges:
        while a != p[a]:  # type: ignore
            p[a], a = p[p[a]], p[a]
        while b != p[b]:  # type: ignore
            p[b], b = p[p[b]], p[b]
        if a != b:
            p[a] = b
            result.append((a, b, c))
    return result


# Floyd-Warshall
def floydwarshall(adj: list[list[int]]) -> list[list[int]]:
    n = len(adj)
    distance = [
        [0 if i == j else adj[i][j] if adj[i][j] else maxsize for j in range(n)]
        for i in range(n)
    ]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])

    return distance


# Segment Trees
def segmenttree(
    data: list[int], default: int, func: Callable[[int, int], int]
) -> tuple[Callable[[int, int], int], Callable[[int, int], None]]:
    k = 2 ** (len(data) - 1).bit_length()
    st = [default] * (2 * k)
    st[k : k + len(data)] = data
    for i in reversed(range(k)):
        st[i] = func(st[i * 2], st[i * 2 + 1])

    def query(a: int, b: int) -> int:
        # [a,b)
        a += k
        b += k
        res = default
        while a < b:
            if a & 1:
                res = func(res, st[a])
                a += 1
            if b & 1:
                b -= 1
                res = func(res, st[b])
            a //= 2
            b //= 2
        return res

    def set(i: int, v: int) -> None:
        i += k
        st[i] = v
        i //= 2
        while i:
            st[i] = func(st[i * 2], st[i * 2 + 1])
            i //= 2

    return query, set


# Union Find
def unionfind(n: int) -> list[int]:
    return [*range(n)]


def union(p: list[int], a: int, b: int) -> None:
    p[a] = b


def find(p: list[int], a: int) -> int:
    while a != p[a]:
        p[a], a = p[p[a]], p[a]
    return a


# Graph to Tree
def graph_to_tree(graph: list[list[int]]) -> list[int]:
    dfs = [0]
    for current in dfs:
        for edge in graph[current]:
            dfs.append(edge)
            graph[edge].remove(current)
    return dfs


# Overlapping ranges


def overlapping_ranges(
    ranges: list[tuple[int, int]], start: int, func: Callable[[int, int], int]
) -> int:
    # [a,b)
    values: list[tuple[int, int]] = []
    for a, b in ranges:
        values.append((a, 1))
        values.append((b, -1))

    current = 0
    result = start
    values.sort()
    for i in range(len(values)):
        t, v = values[i]
        current += v
        if i != len(values) and t == values[i + 1][0]:
            continue
        result = func(result, current)
    return result


# Topological Sort
def topological_sort(graph: list[list[int]]) -> list[int]:
    n = len(graph)
    degrees = [0] * n

    for node in range(n):
        for edge in graph[node]:
            degrees[edge] += 1

    zeros = [node for node in range(n) if degrees[node] == 0]

    result: list[int] = []
    while zeros:
        current = zeros.pop()
        result.append(current)
        for edge in graph[current]:
            degrees[edge] -= 1
            if degrees[edge] == 0:
                zeros.append(edge)

    return result


# Kosaraju


def kosaraju(graph: list[list[int]], graphi: list[list[int]]) -> list[list[int]]:
    n = len(graph)
    path: list[int] = []
    visited = [0] * n
    for node in range(n):
        if visited[node]:
            continue
        stack = [node]
        while stack:
            current = stack.pop()
            if visited[current] == 0:
                stack.append(current)
                stack.extend(edge for edge in graph[current] if not visited[edge])
            if visited[current] == 1:
                path.append(current)
            if visited[current] == 2:
                continue
            visited[current] += 1

    path.reverse()
    groups: list[list[int]] = []
    visited = [False] * n

    for node in path:
        if visited[node]:
            continue
        stack = [node]
        groups.append([])
        while stack:
            current = stack.pop()
            if visited[current]:
                continue
            visited[current] = True
            groups[-1].append(current)
            stack += graphi[current]
    return groups


# BFS


def bfs(graph: list[list[int]], start: int, f: Callable[[int], None]) -> None:
    n = len(graph)
    queue = [start]
    visited = [False] * n
    for current in queue:
        if visited[current]:
            continue
        visited[current] = True
        f(current)
        queue += graph[current]


# DFS


def dfs(
    graph: list[list[int]],
    start: int,
    pre: Callable[[int], None],
    pos: Callable[[int], None],
) -> None:
    n = len(graph)
    stack = [start]
    visited = [0] * n
    while stack:
        current = stack.pop()
        if visited[current] == 0:
            pre(current)
            stack.append(current)
            stack.extend(e for e in graph[current] if not visited[e])
        if visited[current] == 1:
            pos(current)
        if visited[current] == 2:
            continue
        visited[current] += 1


# Ford-Fulkenson


def fordfulkenson(graph: list[list[int]]) -> int:
    n = len(graph)
    result = 0
    while True:
        queue = [0]
        parents = [-1] * n
        parents[0] = 0
        for current in queue:
            for edge, cost in enumerate(graph[current]):
                if parents[edge] != -1 or cost == 0:
                    continue
                parents[edge] = current
                queue.append(edge)
        if parents[n - 1] == -1:
            break
        current = n - 1
        path: list[tuple[int, int]] = []
        while current != 0:
            path.append((parents[current], current))  # type: ignore
            current = parents[current]  # type: ignore
        minflow = min(graph[parent][current] for parent, current in path)
        result += minflow
        for parent, current in path:
            graph[parent][current] -= minflow
            graph[current][parent] += minflow
    return result


def euclidean(a: int, b: int) -> tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = euclidean(b % a, a)
    return gcd, (y1 - (b // a) * x1), x1


def invmod(a: int, m: int) -> int:
    _, x, _ = euclidean(a, m)
    return x % m


def chinese_remainder_gauss(n: list[int], a: list[int]) -> int:
    N = prod(n)
    result = 0
    for i in range(len(n)):
        ai = a[i]
        ni = n[i]
        bi = N // ni
        result += ai * bi * pow(bi, -1, ni)
    return result % N


def eratosthenes(n: int) -> list[int]:
    a = [False, True] * (n // 2)
    a[:3] = False, False, False
    for i in compress(range(isqrt(n) + 1), a):
        for n in range(i**2, n, i * 2):
            a[n] = False
    a[2] = n > 2
    return [*compress(count(), a)]


def diophantine(a: int, b: int, c: int) -> tuple[int, int]:
    x = c * pow(a, -1, b)
    return x, (c - a * x) // b


def gcd_n(numbers: list[int]) -> int:
    return reduce(gcd, numbers, 0)


def lcm(a: int, b: int) -> int:
    return abs(a) // gcd(a, b) * abs(b)


def skips(parents: list[int | None], heights: list[int]) -> list[list[int | None]]:
    jump = 2
    skips = [parents]
    max_height = max(heights)
    while jump <= max_height:
        skip = skips[-1]
        skips.append([skip[parent] if parent is not None else None for parent in skip])
        jump *= 2
    return skips


def common_ancestor(
    parents: list[int | None], heights: list[int], a: int, b: int
) -> int:
    delta = heights[a] - heights[b]
    if delta < 0:
        a, b = b, a
        delta *= -1
    for skip in skips(parents, heights):
        if delta & 1:
            s = skip[a]
            assert s is not None
            a = s
        delta >>= 1
    if a == b:
        return a
    for skip in reversed(skips(parents, heights)):
        skip_a = skip[a]
        skip_b = skip[b]
        if skip_a == skip_b:
            continue
        assert skip_a is not None
        assert skip_b is not None
        a, b = skip_a, skip_b
    assert parents[a] == parents[b]
    result = parents[a]
    return result if result is not None else 0


def heights(tree: list[list[int]]) -> list[int]:
    n = len(tree)
    dfs = [0]
    heights = [0] * n
    while dfs:
        current = dfs.pop()
        for edge in tree[current]:
            dfs.append(edge)
            heights[edge] = heights[current] + 1
    return heights


# Trie

Tree = dict[str | None, "Tree"]


def trie() -> Tree:
    return {}


def insert(trie: Tree, word: str) -> None:
    curr: Tree = trie
    for c in word:
        curr = curr.setdefault(c, {})
    curr[None] = {}


def search(trie: Tree, word: str) -> Iterator[int]:
    curr = trie
    if None in trie:
        yield 0
    for i, c in enumerate(word):
        if c not in curr:
            break
        curr = curr[c]
        if None in curr:
            yield i + 1


# Fenwick


def fenwick_tree(x: list[int]) -> list[int]:
    tree = x.copy()
    for i in range(len(x)):
        j = i | (i + 1)
        if j < len(x):
            tree[j] += tree[i]
    return tree


def increase(tree: list[int], index: int, x: int):
    while index < len(tree):
        tree[index] += x
        index |= index + 1


def query(tree: list[int], end: int) -> int:  # [0,end)
    x = 0
    while end:
        x += tree[end - 1]
        end &= end - 1
    return x


def find_kth(tree: list[int], k: int) -> tuple[int, int]:
    idx = -1
    for d in reversed(range(len(tree).bit_length())):
        right_idx = idx + (1 << d)
        if right_idx < len(tree) and tree[right_idx] <= k:
            idx = right_idx
            k -= tree[idx]
    return idx + 1, k


# Sorted pop list


def sorted_list(data: list[int]) -> list[list[int]]:
    SIZE = 2048
    buckets = (len(data) + SIZE - 1) // SIZE  # Ceil
    return [data[i * SIZE : min((i + 1) * SIZE, len(data))] for i in range(buckets)]


def pop(sl: list[list[int]], index: int) -> int:
    bucket = 0
    while index >= len(sl[bucket]):
        index -= len(sl[bucket])
        bucket += 1
    return sl[bucket].pop(index)


# Sorted list


class SortedList:
    BLOCK_SIZE = 700

    def __init__(self, iterable: Iterable[int] = ()):
        self.macro: list[int] = []
        self.micros: list[list[int]] = [[]]
        self.micro_size = [0]
        self.fenwick = fenwick_tree([0])
        self.size = 0
        for item in iterable:
            self.insert(item)

    def insert(self, x: int) -> None:
        i = bisect_left(self.macro, x)
        j = bisect_right(self.micros[i], x)
        self.micros[i].insert(j, x)
        self.size += 1
        self.micro_size[i] += 1
        increase(self.fenwick, i, 1)
        if len(self.micros[i]) >= SortedList.BLOCK_SIZE:
            self.micros[i : i + 1] = (
                self.micros[i][: SortedList.BLOCK_SIZE >> 1],
                self.micros[i][SortedList.BLOCK_SIZE >> 1 :],
            )
            self.micro_size[i : i + 1] = (
                SortedList.BLOCK_SIZE >> 1,
                SortedList.BLOCK_SIZE >> 1,
            )
            self.fenwick = fenwick_tree(self.micro_size)
            self.macro.insert(i, self.micros[i + 1][0])

    def pop(self, k: int = -1) -> int:
        i, j = self._find_kth(k)
        self.size -= 1
        self.micro_size[i] -= 1
        increase(self.fenwick, i, -1)
        return self.micros[i].pop(j)

    def __getitem__(self, k: int) -> int:
        i, j = self._find_kth(k)
        return self.micros[i][j]

    def count(self, x: int) -> int:
        return self.upper_bound(x) - self.lower_bound(x)

    def __contains__(self, x: int) -> bool:
        return self.count(x) > 0

    def lower_bound(self, x: int) -> int:
        i = bisect_left(self.macro, x)
        return query(self.fenwick, i) + bisect_left(self.micros[i], x)

    def upper_bound(self, x: int) -> int:
        i = bisect_right(self.macro, x)
        return query(self.fenwick, i) + bisect_right(self.micros[i], x)

    def _find_kth(self, k: int) -> tuple[int, int]:
        return find_kth(self.fenwick, k + self.size if k < 0 else k)

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[int]:
        return (x for micro in self.micros for x in micro)

    def __repr__(self) -> str:
        return str(list(self))
