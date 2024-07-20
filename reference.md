# Team Reference Document

## Forgettable Python Imports

-   heapq: nlargest, priority queue
-   bisect: binary search
-   collections: dequeue
-   sys: maxsize
-   math: comb, perm, isqrt, prod
-   functools: reduce
-   itertoos: count, compress, takewhile, repeat

## Data structures

### Union-Find

Collection of non-overlapping sets

-   constructor: $\Theta(n)$
-   union: $\Theta(log(n))$
-   find: $\Theta(log(n))$

```python
def unionfind(n: int) -> list[int]:
    return [*range(n)]

def union(p: list[int], a: int, b: int) -> None:
    p[a] = b

def find(p: list[int], a: int) -> int:
    while a != p[a]:
        p[a], a = p[p[a]], p[a]
    return a
```

### Segment Tree

Information about intervals  
Index 1 is root, index 0 is unused

-   constructor: $\Theta(nlog(n))$
-   query: $\Theta(log(n))$
-   set: $\Theta(log(n))$

```python
def segmenttree(data: list[int], default: int, func: Callable[[int, int], int]):
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
```

### Trie

Tree used to check if a string has some of the inserted prefixes

-   constructor: $\Theta(1)$
-   insert: $\Theta(n)$
-   search: $\Theta(n)$

```python
Tree = dict["str | None", "Tree"]

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
```

### Fenwick tree

Information about intervals only on Habelian group operators (invertible,commutative,associative) (replace + with the operator)

-   constructor: $\Theta(n)$
-   query: $\Theta(log(n))$
-   change: $\Theta(log(n))$
-   find_kth: $\Theta(log(n))$

```python
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


def query(tree: list[int], end: int) -> int:
    # [0,end)
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
```

### Sorted pop

Sorted list with fast index pop  
$\text{SIZE} \propto \sqrt[3]n$

-   constructor: $\Theta(n)$
-   pop: $\Theta(\sqrt[3]n)$

```python
def sorted_list(data: list[int]) -> list[list[int]]:
    SIZE = 700
    buckets = (len(data) + SIZE - 1) // SIZE  # Ceil
    return [data[i * SIZE : min((i + 1) * SIZE, len(data))] for i in range(buckets)]


def pop(sl: list[list[int]], index: int) -> int:
    bucket = 0
    while index >= len(sl[bucket]):
        index -= len(sl[bucket])
        bucket += 1
    return sl[bucket].pop(index)
```

### Sorted list

Full sorted list implementation  
Requires a Fenwick tree implementation  
$\text{SIZE} \propto \sqrt[3]n$

-   constructor: $\Theta(n\sqrt[3]n)$
-   insert: $\Theta(\sqrt[3]n)$
-   pop: $\Theta(\sqrt[3]n)$
-   getitem: $\Theta(log(n))$
-   count: $\Theta(log(n))$
-   contains: $\Theta(log(n))$
-   lower_bound: $\Theta(log(n))$
-   upper_bound: $\Theta(log(n))$

```python
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
```

## Graph

### Breadth-First Search

$\Theta(V+E)$

```python
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
```

### Deep First Search

With support for preordering and postordering

$\Theta(V+E)$

```python
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
```

### Dijkstra

Shortest paths from the source to all other nodes in a graph with unbounded non-negative weights

$\Theta((V+E)log(V+E))$

```python
def dijkstra(graph: list[list[tuple[int, int]]]) -> list[int]:
    n = len(graph)
    queue = [(0, 0)]
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
```

### Kruskal

Minimum spanning forest

$\Theta(Elog(E))$

```python
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
```

### Floyd-Warshall

Shortest paths in a weighted graph with positive or negative edge weights with no negative cycles between all pairs of vertices

$\Theta(V^2)$

```python
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
```

### Topological Sorting

$\Theta(V+E)$

```python
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
```

### Kosaraju

Strongly connected components of a directed graph

$\Theta(E+V)$

```python
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
```

### Ford-Fulkenson

Maximum flow

$\Theta(E^2V)$

```python
def fordfulkenson(graph:list[list[int]])->int:
    n=len(graph)
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
```

### Graph to tree

```python
def graph_to_tree(graph: list[list[int]]) -> list[int]:
    bfs = [0]
    for current in bfs:
        for edge in graph[current]:
            bfs.append(edge)
            graph[edge].remove(current)
    return bfs
```

### Skips

$\Theta(Vlog(V))$

```python
def skips(parents: list[int | None], heights: list[int]) -> list[list[int | None]]:
    jump = 2
    skips = [parents]
    max_height = max(heights)
    while jump <= max_height:
        skip = skips[-1]
        skips.append([skip[parent] if parent is not None else None for parent in skip])
        jump *= 2
    return skips
```

### Common ancestor

$\Theta(log(V))$

```python
def common_ancestor(
    parents: list[int | None], heights: list[int], a: int, b: int
) -> int:
    delta = heights[a] - heights[b]
    if delta < 0:
        a, b = b, a
        delta *= -1
    for skip in skips:
        if delta & 1:
            s = skip[a]
            assert s is not None
            a = s
        delta >>= 1
    for skip in reversed(skips):
        skip_a = skip[a]
        skip_b = skip[b]
        if skip_a == skip_b:
            continue
        assert skip_a is not None
        assert skip_b is not None
        a, b = skip_a, skip_b
    if a == b:
        return a
    else:
        assert parents[a] == parents[b]
        result = parents[a]
        return result if result is not None else 0
```

### Heights of nodes

$\Theta(V+E)$

```python
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
```

---

## Sorting

### Overlapping ranges

$\Theta(nlog(n))$

```python
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
```

## Mathematics

### Euclidean

$\Theta(log(min(a, b)))$

```python
def euclidean(a: int, b: int) -> tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = euclidean(b % a, a)
    return gcd, (y1 - (b // a) * x1), x1
```

### Greatest common divisor with n numbers

```python
def gcd_n(numbers: list[int]) -> int:
    return reduce(gcd, numbers, 0)
```

### Least common multiple

```python
def lcm(a: int, b: int) -> int:
    return abs(a) // gcd(a, b) * abs(b)
```

### Multiplicative inverse

$a$ and $m$ must be coprime

$\Theta(log(min(a, m)))$

```python
def invmod(a: int, m: int) -> int:
    _, x, _ = euclidean(a, m)
    return x % m
```

### Chinese Remainder

To check

$$
x=a_1mod(n_1)\\
...\\
x=a_kmod(n_k)
$$

```python
def chinese_remainder_gauss(n: list[int], a: list[int]) -> int:
    N = prod(n)
    result = 0
    for i in range(len(n)):
        ai = a[i]
        ni = n[i]
        bi = N // ni
        result += ai * bi * pow(bi, -1, ni)
    return result % N
```

### Eratosthenes

All prime numbers up to $n$

$\Theta(n\ log(log(n)))$

```python
def eratosthenes(n: int) -> list[int]:
    # n excluded
    a = [False, True] * (n // 2)
    a[1] = False
    for i in compress(range(isqrt(n) + 1), a):
        for n in range(i**2, n, i * 2):
            a[n] = False
    a[2] = True
    return [*compress(count(), a)]
```

### Diophantine

$a$ and $b$ must be coprime

$$
ax+by=c \\
ax=c-by \\
x=ca^{-1}\ mod(b)\\
\\
(x,y)\in\{(x_0+bk,y_0-ak)|k\in Z\}
$$

$\Theta(log(min(a,b)))$

```python
def diophantine(a: int, b: int, c: int) -> tuple[int, int]:
    x = c * pow(a, -1, b)
    return x, (c - a * x) // b
```

### Factors of a number

$\Theta(\sqrt{n}log(log(\sqrt{n})))$

```python
def factor(n: int) -> list[int]:
    result = [
        p
        for p in eratosthenes(isqrt(n) + 1)
        for _ in takewhile(lambda _: n % p == 0, repeat(p))
        if (n := n // p)
    ]
    if n >= 2:
        result.append(n)
    return result
```

### Polygon Area

$A=\frac{1}{2}|\sum_{i=1}^{n-1}(x_iy_{i+1}-x_{i+1}y_i)|$

### Divisors of number

$\Theta(\sqrt{n})$

```python
def divisors(n: int) -> list[int]:
    result: list[int] = []
    for i in range(1, isqrt(n) + 1):
        if n % i != 0:
            continue
        result.append(i)
        if i * i == n:
            continue
        result.append(n // i)
    return result
```
