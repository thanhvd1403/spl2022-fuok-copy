import sys
input = sys.stdin.readline
from collections import defaultdict
from copy import deepcopy

def check_planarity(G, counterexample=False):
    planarity_state = LRPlanarity(G)
    embedding = planarity_state.lr_planarity()
    if embedding is None:
        return False, None
    else:
        # graph is planar
        return True
    
def top_of_stack(l):
    """Returns the element on top of the stack."""
    if not l:
        return None
    return l[-1]

class LRPlanarity:
    """A class to maintain the state during planarity check."""

    __slots__ = [
        "G",
        "roots",
        "height",
        "lowpt",
        "lowpt2",
        "nesting_depth",
        "parent_edge",
        "DG",
        "adjs",
        "ordered_adjs",
        "ref",
        "side",
        "S",
        "stack_bottom",
        "lowpt_edge",
        "left_ref",
        "right_ref",
        "embedding",
    ]

    def __init__(self, G):
        # copy G without adding self-loops
        self.G = Graph()
        self.G.add_nodes_from(G.nodes)
        for e in G.edges:
            if e[0] != e[1]:
                self.G.add_edge(e[0], e[1])

        self.roots = []

        # distance from tree root
        self.height = defaultdict(lambda: None)

        self.lowpt = {}  # height of lowest return point of an edge
        self.lowpt2 = {}  # height of second lowest return point
        self.nesting_depth = {}  # for nesting order

        # None -> missing edge
        self.parent_edge = defaultdict(lambda: None)

        # oriented DFS graph
        self.DG = DiGraph()
        self.DG.add_nodes_from(G.nodes)

        self.adjs = {}
        self.ordered_adjs = {}

        self.ref = defaultdict(lambda: None)
        self.side = defaultdict(lambda: 1)

        # stack of conflict pairs
        self.S = []
        self.stack_bottom = {}
        self.lowpt_edge = {}

        self.left_ref = {}
        self.right_ref = {}

        self.embedding = PlanarEmbedding()

    def lr_planarity(self):
        """Execute the LR planarity test.

        Returns
        -------
        embedding : dict
            If the graph is planar an embedding is returned. Otherwise None.
        """
        if self.G.order() > 2 and self.G.size() > 3 * self.G.order() - 6:
            # graph is not planar
            return None

        # make adjacency lists for dfs
        for v in self.G:
            self.adjs[v] = list(self.G[v])

        # orientation of the graph by depth first search traversal
        for v in self.G:
            if self.height[v] is None:
                self.height[v] = 0
                self.roots.append(v)
                self.dfs_orientation(v)

        # Free no longer used variables
        self.G = None
        self.lowpt2 = None
        self.adjs = None

        # testing
        for v in self.DG:  # sort the adjacency lists by nesting depth
            # note: this sorting leads to non linear time
            self.ordered_adjs[v] = sorted(
                self.DG[v], key=lambda x: self.nesting_depth[(v, x)]
            )
        for v in self.roots:
            if not self.dfs_testing(v):
                return None

        # Free no longer used variables
        self.height = None
        self.lowpt = None
        self.S = None
        self.stack_bottom = None
        self.lowpt_edge = None

        for e in self.DG.edges:
            self.nesting_depth[e] = self.sign(e) * self.nesting_depth[e]

        self.embedding.add_nodes_from(self.DG.nodes)
        for v in self.DG:
            # sort the adjacency lists again
            self.ordered_adjs[v] = sorted(
                self.DG[v], key=lambda x: self.nesting_depth[(v, x)]
            )
            # initialize the embedding
            previous_node = None
            for w in self.ordered_adjs[v]:
                self.embedding.add_half_edge_cw(v, w, previous_node)
                previous_node = w

        # Free no longer used variables
        self.DG = None
        self.nesting_depth = None
        self.ref = None

        # compute the complete embedding
        for v in self.roots:
            self.dfs_embedding(v)

        # Free no longer used variables
        self.roots = None
        self.parent_edge = None
        self.ordered_adjs = None
        self.left_ref = None
        self.right_ref = None
        self.side = None

        return self.embedding

    def lr_planarity_recursive(self):
        """Recursive version of :meth:`lr_planarity`."""
        if self.G.order() > 2 and self.G.size() > 3 * self.G.order() - 6:
            # graph is not planar
            return None

        # orientation of the graph by depth first search traversal
        for v in self.G:
            if self.height[v] is None:
                self.height[v] = 0
                self.roots.append(v)
                self.dfs_orientation_recursive(v)

        # Free no longer used variable
        self.G = None

        # testing
        for v in self.DG:  # sort the adjacency lists by nesting depth
            # note: this sorting leads to non linear time
            self.ordered_adjs[v] = sorted(
                self.DG[v], key=lambda x: self.nesting_depth[(v, x)]
            )
        for v in self.roots:
            if not self.dfs_testing_recursive(v):
                return None

        for e in self.DG.edges:
            self.nesting_depth[e] = self.sign_recursive(e) * self.nesting_depth[e]

        self.embedding.add_nodes_from(self.DG.nodes)
        for v in self.DG:
            # sort the adjacency lists again
            self.ordered_adjs[v] = sorted(
                self.DG[v], key=lambda x: self.nesting_depth[(v, x)]
            )
            # initialize the embedding
            previous_node = None
            for w in self.ordered_adjs[v]:
                self.embedding.add_half_edge_cw(v, w, previous_node)
                previous_node = w

        # compute the complete embedding
        for v in self.roots:
            self.dfs_embedding_recursive(v)

        return self.embedding

    def dfs_orientation(self, v):
        """Orient the graph by DFS, compute lowpoints and nesting order."""
        # the recursion stack
        dfs_stack = [v]
        # index of next edge to handle in adjacency list of each node
        ind = defaultdict(lambda: 0)
        # boolean to indicate whether to skip the initial work for an edge
        skip_init = defaultdict(lambda: False)

        while dfs_stack:
            v = dfs_stack.pop()
            e = self.parent_edge[v]

            for w in self.adjs[v][ind[v] :]:
                vw = (v, w)

                if not skip_init[vw]:
                    if (v, w) in self.DG.edges or (w, v) in self.DG.edges:
                        ind[v] += 1
                        continue  # the edge was already oriented

                    self.DG.add_edge(v, w)  # orient the edge

                    self.lowpt[vw] = self.height[v]
                    self.lowpt2[vw] = self.height[v]
                    if self.height[w] is None:  # (v, w) is a tree edge
                        self.parent_edge[w] = vw
                        self.height[w] = self.height[v] + 1

                        dfs_stack.append(v)  # revisit v after finishing w
                        dfs_stack.append(w)  # visit w next
                        skip_init[vw] = True  # don't redo this block
                        break  # handle next node in dfs_stack (i.e. w)
                    else:  # (v, w) is a back edge
                        self.lowpt[vw] = self.height[w]

                # determine nesting graph
                self.nesting_depth[vw] = 2 * self.lowpt[vw]
                if self.lowpt2[vw] < self.height[v]:  # chordal
                    self.nesting_depth[vw] += 1

                # update lowpoints of parent edge e
                if e is not None:
                    if self.lowpt[vw] < self.lowpt[e]:
                        self.lowpt2[e] = min(self.lowpt[e], self.lowpt2[vw])
                        self.lowpt[e] = self.lowpt[vw]
                    elif self.lowpt[vw] > self.lowpt[e]:
                        self.lowpt2[e] = min(self.lowpt2[e], self.lowpt[vw])
                    else:
                        self.lowpt2[e] = min(self.lowpt2[e], self.lowpt2[vw])

                ind[v] += 1

    def dfs_orientation_recursive(self, v):
        """Recursive version of :meth:`dfs_orientation`."""
        e = self.parent_edge[v]
        for w in self.G[v]:
            if (v, w) in self.DG.edges or (w, v) in self.DG.edges:
                continue  # the edge was already oriented
            vw = (v, w)
            self.DG.add_edge(v, w)  # orient the edge

            self.lowpt[vw] = self.height[v]
            self.lowpt2[vw] = self.height[v]
            if self.height[w] is None:  # (v, w) is a tree edge
                self.parent_edge[w] = vw
                self.height[w] = self.height[v] + 1
                self.dfs_orientation_recursive(w)
            else:  # (v, w) is a back edge
                self.lowpt[vw] = self.height[w]

            # determine nesting graph
            self.nesting_depth[vw] = 2 * self.lowpt[vw]
            if self.lowpt2[vw] < self.height[v]:  # chordal
                self.nesting_depth[vw] += 1

            # update lowpoints of parent edge e
            if e is not None:
                if self.lowpt[vw] < self.lowpt[e]:
                    self.lowpt2[e] = min(self.lowpt[e], self.lowpt2[vw])
                    self.lowpt[e] = self.lowpt[vw]
                elif self.lowpt[vw] > self.lowpt[e]:
                    self.lowpt2[e] = min(self.lowpt2[e], self.lowpt[vw])
                else:
                    self.lowpt2[e] = min(self.lowpt2[e], self.lowpt2[vw])

    def dfs_testing(self, v):
        """Test for LR partition."""
        # the recursion stack
        dfs_stack = [v]
        # index of next edge to handle in adjacency list of each node
        ind = defaultdict(lambda: 0)
        # boolean to indicate whether to skip the initial work for an edge
        skip_init = defaultdict(lambda: False)

        while dfs_stack:
            v = dfs_stack.pop()
            e = self.parent_edge[v]
            # to indicate whether to skip the final block after the for loop
            skip_final = False

            for w in self.ordered_adjs[v][ind[v] :]:
                ei = (v, w)

                if not skip_init[ei]:
                    self.stack_bottom[ei] = top_of_stack(self.S)

                    if ei == self.parent_edge[w]:  # tree edge
                        dfs_stack.append(v)  # revisit v after finishing w
                        dfs_stack.append(w)  # visit w next
                        skip_init[ei] = True  # don't redo this block
                        skip_final = True  # skip final work after breaking
                        break  # handle next node in dfs_stack (i.e. w)
                    else:  # back edge
                        self.lowpt_edge[ei] = ei
                        self.S.append(ConflictPair(right=Interval(ei, ei)))

                # integrate new return edges
                if self.lowpt[ei] < self.height[v]:
                    if w == self.ordered_adjs[v][0]:  # e_i has return edge
                        self.lowpt_edge[e] = self.lowpt_edge[ei]
                    else:  # add constraints of e_i
                        if not self.add_constraints(ei, e):
                            # graph is not planar
                            return False

                ind[v] += 1

            if not skip_final:
                # remove back edges returning to parent
                if e is not None:  # v isn't root
                    self.remove_back_edges(e)

        return True

    def dfs_testing_recursive(self, v):
        """Recursive version of :meth:`dfs_testing`."""
        e = self.parent_edge[v]
        for w in self.ordered_adjs[v]:
            ei = (v, w)
            self.stack_bottom[ei] = top_of_stack(self.S)
            if ei == self.parent_edge[w]:  # tree edge
                if not self.dfs_testing_recursive(w):
                    return False
            else:  # back edge
                self.lowpt_edge[ei] = ei
                self.S.append(ConflictPair(right=Interval(ei, ei)))

            # integrate new return edges
            if self.lowpt[ei] < self.height[v]:
                if w == self.ordered_adjs[v][0]:  # e_i has return edge
                    self.lowpt_edge[e] = self.lowpt_edge[ei]
                else:  # add constraints of e_i
                    if not self.add_constraints(ei, e):
                        # graph is not planar
                        return False

        # remove back edges returning to parent
        if e is not None:  # v isn't root
            self.remove_back_edges(e)
        return True

    def add_constraints(self, ei, e):
        P = ConflictPair()
        # merge return edges of e_i into P.right
        while True:
            Q = self.S.pop()
            if not Q.left.empty():
                Q.swap()
            if not Q.left.empty():  # not planar
                return False
            if self.lowpt[Q.right.low] > self.lowpt[e]:
                # merge intervals
                if P.right.empty():  # topmost interval
                    P.right = Q.right.copy()
                else:
                    self.ref[P.right.low] = Q.right.high
                P.right.low = Q.right.low
            else:  # align
                self.ref[Q.right.low] = self.lowpt_edge[e]
            if top_of_stack(self.S) == self.stack_bottom[ei]:
                break
        # merge conflicting return edges of e_1,...,e_i-1 into P.L
        while top_of_stack(self.S).left.conflicting(ei, self) or top_of_stack(
            self.S
        ).right.conflicting(ei, self):
            Q = self.S.pop()
            if Q.right.conflicting(ei, self):
                Q.swap()
            if Q.right.conflicting(ei, self):  # not planar
                return False
            # merge interval below lowpt(e_i) into P.R
            self.ref[P.right.low] = Q.right.high
            if Q.right.low is not None:
                P.right.low = Q.right.low

            if P.left.empty():  # topmost interval
                P.left = Q.left.copy()
            else:
                self.ref[P.left.low] = Q.left.high
            P.left.low = Q.left.low

        if not (P.left.empty() and P.right.empty()):
            self.S.append(P)
        return True

    def remove_back_edges(self, e):
        u = e[0]
        # trim back edges ending at parent u
        # drop entire conflict pairs
        while self.S and top_of_stack(self.S).lowest(self) == self.height[u]:
            P = self.S.pop()
            if P.left.low is not None:
                self.side[P.left.low] = -1

        if self.S:  # one more conflict pair to consider
            P = self.S.pop()
            # trim left interval
            while P.left.high is not None and P.left.high[1] == u:
                P.left.high = self.ref[P.left.high]
            if P.left.high is None and P.left.low is not None:
                # just emptied
                self.ref[P.left.low] = P.right.low
                self.side[P.left.low] = -1
                P.left.low = None
            # trim right interval
            while P.right.high is not None and P.right.high[1] == u:
                P.right.high = self.ref[P.right.high]
            if P.right.high is None and P.right.low is not None:
                # just emptied
                self.ref[P.right.low] = P.left.low
                self.side[P.right.low] = -1
                P.right.low = None
            self.S.append(P)

        # side of e is side of a highest return edge
        if self.lowpt[e] < self.height[u]:  # e has return edge
            hl = top_of_stack(self.S).left.high
            hr = top_of_stack(self.S).right.high

            if hl is not None and (hr is None or self.lowpt[hl] > self.lowpt[hr]):
                self.ref[e] = hl
            else:
                self.ref[e] = hr

    def dfs_embedding(self, v):
        """Completes the embedding."""
        # the recursion stack
        dfs_stack = [v]
        # index of next edge to handle in adjacency list of each node
        ind = defaultdict(lambda: 0)

        while dfs_stack:
            v = dfs_stack.pop()

            for w in self.ordered_adjs[v][ind[v] :]:
                ind[v] += 1
                ei = (v, w)

                if ei == self.parent_edge[w]:  # tree edge
                    self.embedding.add_half_edge_first(w, v)
                    self.left_ref[v] = w
                    self.right_ref[v] = w

                    dfs_stack.append(v)  # revisit v after finishing w
                    dfs_stack.append(w)  # visit w next
                    break  # handle next node in dfs_stack (i.e. w)
                else:  # back edge
                    if self.side[ei] == 1:
                        self.embedding.add_half_edge_cw(w, v, self.right_ref[w])
                    else:
                        self.embedding.add_half_edge_ccw(w, v, self.left_ref[w])
                        self.left_ref[w] = v

    def dfs_embedding_recursive(self, v):
        """Recursive version of :meth:`dfs_embedding`."""
        for w in self.ordered_adjs[v]:
            ei = (v, w)
            if ei == self.parent_edge[w]:  # tree edge
                self.embedding.add_half_edge_first(w, v)
                self.left_ref[v] = w
                self.right_ref[v] = w
                self.dfs_embedding_recursive(w)
            else:  # back edge
                if self.side[ei] == 1:
                    # place v directly after right_ref[w] in embed. list of w
                    self.embedding.add_half_edge_cw(w, v, self.right_ref[w])
                else:
                    # place v directly before left_ref[w] in embed. list of w
                    self.embedding.add_half_edge_ccw(w, v, self.left_ref[w])
                    self.left_ref[w] = v

    def sign(self, e):
        """Resolve the relative side of an edge to the absolute side."""
        # the recursion stack
        dfs_stack = [e]
        # dict to remember reference edges
        old_ref = defaultdict(lambda: None)

        while dfs_stack:
            e = dfs_stack.pop()

            if self.ref[e] is not None:
                dfs_stack.append(e)  # revisit e after finishing self.ref[e]
                dfs_stack.append(self.ref[e])  # visit self.ref[e] next
                old_ref[e] = self.ref[e]  # remember value of self.ref[e]
                self.ref[e] = None
            else:
                self.side[e] *= self.side[old_ref[e]]

        return self.side[e]

    def sign_recursive(self, e):
        """Recursive version of :meth:`sign`."""
        if self.ref[e] is not None:
            self.side[e] = self.side[e] * self.sign_recursive(self.ref[e])
            self.ref[e] = None
        return self.side[e]
    
class Interval:
    def __init__(self, low=None, high=None):
        self.low = low
        self.high = high

    def empty(self):
        """Check if the interval is empty"""
        return self.low is None and self.high is None

    def copy(self):
        """Returns a copy of this interval"""
        return Interval(self.low, self.high)

    def conflicting(self, b, planarity_state):
        """Returns True if interval I conflicts with edge b"""
        return ( not self.empty() and planarity_state.lowpt[self.high] > planarity_state.lowpt[b])


    
class ConflictPair:
    def __init__(self, left=Interval(), right=Interval()):
        self.left = left
        self.right = right

    def swap(self):
        """Swap left and right intervals"""
        temp = self.left
        self.left = self.right
        self.right = temp

    def lowest(self, planarity_state):
        """Returns the lowest lowpoint of a conflict pair"""
        if self.left.empty():
            return planarity_state.lowpt[self.right.low]
        if self.right.empty():
            return planarity_state.lowpt[self.left.low]
        return min(
            planarity_state.lowpt[self.left.low], planarity_state.lowpt[self.right.low]
        )

class Graph:
    node_dict_factory = dict
    node_attr_dict_factory = dict
    adjlist_outer_dict_factory = dict
    adjlist_inner_dict_factory = dict
    edge_attr_dict_factory = dict
    graph_attr_dict_factory = dict

    def to_directed_class(self):
        return DiGraph

    def to_undirected_class(self):
        return Graph

    def __init__(self, incoming_graph_data=None, **attr):
        self.graph_attr_dict_factory = self.graph_attr_dict_factory
        self.node_dict_factory = self.node_dict_factory
        self.node_attr_dict_factory = self.node_attr_dict_factory
        self.adjlist_outer_dict_factory = self.adjlist_outer_dict_factory
        self.adjlist_inner_dict_factory = self.adjlist_inner_dict_factory
        self.edge_attr_dict_factory = self.edge_attr_dict_factory

        self.graph = self.graph_attr_dict_factory()  # dictionary for graph attributes
        self._node = self.node_dict_factory()  # empty node attribute dict
        self._adj = self.adjlist_outer_dict_factory()  # empty adjacency dict
        # attempt to load graph with data
        self.graph.update(attr)

    @property
    def adj(self):
        """Graph adjacency object holding the neighbors of each node.
        This object is a read-only dict-like structure with node keys
        and neighbor-dict values.  The neighbor-dict is keyed by neighbor
        to the edge-data-dict.  So `G.adj[3][2]['color'] = 'blue'` sets
        the color of the edge `(3, 2)` to `"blue"`.
        Iterating over G.adj behaves like a dict. Useful idioms include
        `for nbr, datadict in G.adj[n].items():`.
        The neighbor information is also provided by subscripting the graph.
        So `for nbr, foovalue in G[node].data('foo', default=1):` works.
        For directed graphs, `G.adj` holds outgoing (successor) info.
        """
        return AdjacencyView(self._adj)

    @property
    def name(self):
        """String identifier of the graph.
        This graph attribute appears in the attribute dict G.graph
        keyed by the string `"name"`. as well as an attribute (technically
        a property) `G.name`. This is entirely user controlled.
        """
        return self.graph.get("name", "")

    @name.setter
    def name(self, s):
        self.graph["name"] = s

    def __str__(self):
        """Returns a short summary of the graph.
        Returns
        -------
        info : string
            Graph information as provided by `nx.info`
        Examples
        --------
        >>> G = nx.Graph(name="foo")
        >>> str(G)
        "Graph named 'foo' with 0 nodes and 0 edges"
        >>> G = nx.path_graph(3)
        >>> str(G)
        'Graph with 3 nodes and 2 edges'
        """
        return "".join(
            [
                type(self).__name__,
                f" named {self.name!r}" if self.name else "",
                f" with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges",
            ]
        )

    def __iter__(self):
        """Iterate over the nodes. Use: 'for n in G'.
        Returns
        -------
        niter : iterator
            An iterator over all nodes in the graph.
        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> [n for n in G]
        [0, 1, 2, 3]
        >>> list(G)
        [0, 1, 2, 3]
        """
        return iter(self._node)

    def __contains__(self, n):
        """Returns True if n is a node, False otherwise. Use: 'n in G'.
        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> 1 in G
        True
        """
        try:
            return n in self._node
        except TypeError:
            return False

    def __len__(self):
        """Returns the number of nodes in the graph. Use: 'len(G)'.
        Returns
        -------
        nnodes : int
            The number of nodes in the graph.
        See Also
        --------
        number_of_nodes: identical method
        order: identical method
        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> len(G)
        4
        """
        return len(self._node)

    def __getitem__(self, n):
        return self.adj[n]

    def add_node(self, node_for_adding, **attr):
        if node_for_adding not in self._node:
            if node_for_adding is None:
                raise ValueError("None cannot be a node")
            self._adj[node_for_adding] = self.adjlist_inner_dict_factory()
            attr_dict = self._node[node_for_adding] = self.node_attr_dict_factory()
            attr_dict.update(attr)
        else:  # update attr even if node already exists
            self._node[node_for_adding].update(attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        for n in nodes_for_adding:
            try:
                newnode = n not in self._node
                newdict = attr
            except TypeError:
                n, ndict = n
                newnode = n not in self._node
                newdict = attr.copy()
                newdict.update(ndict)
            if newnode:
                if n is None:
                    raise ValueError("None cannot be a node")
                self._adj[n] = self.adjlist_inner_dict_factory()
                self._node[n] = self.node_attr_dict_factory()
            self._node[n].update(newdict)

    def remove_node(self, n):
        adj = self._adj
        try:
            nbrs = list(adj[n])  # list handles self-loops (allows mutation)
            del self._node[n]
        except KeyError as err:  # NetworkXError if n not in self
            pass
        for u in nbrs:
            del adj[u][n]  # remove all edges n-u in graph
        del adj[n]  # now remove node

    def remove_nodes_from(self, nodes):
        adj = self._adj
        for n in nodes:
            try:
                del self._node[n]
                for u in list(adj[n]):  # list handles self-loops
                    del adj[u][n]  # (allows mutation of dict in loop)
                del adj[n]
            except KeyError:
                pass

    @property
    def nodes(self):
        nodes = NodeView(self)
        # Lazy View creation: overload the (class) property on the instance
        # Then future G.nodes use the existing View
        # setattr doesn't work because attribute already exists
        self.__dict__["nodes"] = nodes
        return nodes

    def number_of_nodes(self):
        return len(self._node)

    def order(self):
        return len(self._node)

    def has_node(self, n):
        try:
            return n in self._node
        except TypeError:
            return False

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        u, v = u_of_edge, v_of_edge
        # add nodes
        if u not in self._node:
            if u is None:
                raise ValueError("None cannot be a node")
            self._adj[u] = self.adjlist_inner_dict_factory()
            self._node[u] = self.node_attr_dict_factory()
        if v not in self._node:
            if v is None:
                raise ValueError("None cannot be a node")
            self._adj[v] = self.adjlist_inner_dict_factory()
            self._node[v] = self.node_attr_dict_factory()
        # add the edge
        datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
        datadict.update(attr)
        self._adj[u][v] = datadict
        self._adj[v][u] = datadict

    def add_edges_from(self, ebunch_to_add, **attr):
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}  # doesn't need edge_attr_dict_factory
            
            if u not in self._node:
                if u is None:
                    raise ValueError("None cannot be a node")
                self._adj[u] = self.adjlist_inner_dict_factory()
                self._node[u] = self.node_attr_dict_factory()
            if v not in self._node:
                if v is None:
                    raise ValueError("None cannot be a node")
                self._adj[v] = self.adjlist_inner_dict_factory()
                self._node[v] = self.node_attr_dict_factory()
            datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
            datadict.update(attr)
            datadict.update(dd)
            self._adj[u][v] = datadict
            self._adj[v][u] = datadict

    def add_weighted_edges_from(self, ebunch_to_add, weight="weight", **attr):
        self.add_edges_from(((u, v, {weight: d}) for u, v, d in ebunch_to_add), **attr)

    def remove_edge(self, u, v):
        try:
            del self._adj[u][v]
            if u != v:  # self-loop needs only one entry removed
                del self._adj[v][u]
        except KeyError as err:
            pass

    def remove_edges_from(self, ebunch):
        adj = self._adj
        for e in ebunch:
            u, v = e[:2]  # ignore edge data if present
            if u in adj and v in adj[u]:
                del adj[u][v]
                if u != v:  # self loop needs only one entry removed
                    del adj[v][u]

    def update(self, edges=None, nodes=None):
        if edges is not None:
            if nodes is not None:
                self.add_nodes_from(nodes)
                self.add_edges_from(edges)
            else:
                # check if edges is a Graph object
                try:
                    graph_nodes = edges.nodes
                    graph_edges = edges.edges
                except AttributeError:
                    # edge not Graph-like
                    self.add_edges_from(edges)
                else:  # edges is Graph-like
                    self.add_nodes_from(graph_nodes.data())
                    self.add_edges_from(graph_edges.data())
                    self.graph.update(edges.graph)
        elif nodes is not None:
            self.add_nodes_from(nodes)
        else:
            pass

    def has_edge(self, u, v):
        try:
            return v in self._adj[u]
        except KeyError:
            return False

    def neighbors(self, n):
        try:
            return iter(self._adj[n])
        except KeyError as err:
            pass

    def edges(self):
        return EdgeView(self)

    def get_edge_data(self, u, v, default=None):
        try:
            return self._adj[u][v]
        except KeyError:
            return default

    def adjacency(self):
        return iter(self._adj.items())

    @property
    def degree(self):
        return DegreeView(self)

    def clear(self):
        self._adj.clear()
        self._node.clear()
        self.graph.clear()

    def clear_edges(self):
        for neighbours_dict in self._adj.values():
            neighbours_dict.clear()

    def is_multigraph(self):
        """Returns True if graph is a multigraph, False otherwise."""
        return False

    def is_directed(self):
        """Returns True if graph is directed, False otherwise."""
        return False

    def copy(self, as_view=False):
        G = self.__class__()
        G.graph.update(self.graph)
        G.add_nodes_from((n, d.copy()) for n, d in self._node.items())
        G.add_edges_from(
            (u, v, datadict.copy())
            for u, nbrs in self._adj.items()
            for v, datadict in nbrs.items()
        )
        return G

    def to_directed(self, as_view=False):
        graph_class = self.to_directed_class()
        # deepcopy when not a view
        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
        G.add_edges_from(
            (u, v, deepcopy(data))
            for u, nbrs in self._adj.items()
            for v, data in nbrs.items()
        )
        return G

    def to_undirected(self, as_view=False):
        graph_class = self.to_undirected_class()
        # deepcopy when not a view
        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
        G.add_edges_from(
            (u, v, deepcopy(d))
            for u, nbrs in self._adj.items()
            for v, d in nbrs.items()
        )
        return G

    def size(self, weight=None):
        """Returns the number of edges or total of all edge weights.
        Parameters
        ----------
        weight : string or None, optional (default=None)
            The edge attribute that holds the numerical value used
            as a weight. If None, then each edge has weight 1.
        Returns
        -------
        size : numeric
            The number of edges or
            (if weight keyword is provided) the total weight sum.
            If weight is None, returns an int. Otherwise a float
            (or more general numeric if the weights are more general).
        See Also
        --------
        number_of_edges
        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.size()
        3
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_edge("a", "b", weight=2)
        >>> G.add_edge("b", "c", weight=4)
        >>> G.size()
        2
        >>> G.size(weight="weight")
        6.0
        """
        s = sum(d for v, d in self.degree(weight=weight))
        # If `weight` is None, the sum of the degrees is guaranteed to be
        # even, so we can perform integer division and hence return an
        # integer. Otherwise, the sum of the weighted degrees is not
        # guaranteed to be an integer, so we perform "real" division.
        return s // 2 if weight is None else s / 2

    def number_of_edges(self, u=None, v=None):
        """Returns the number of edges between two nodes.
        Parameters
        ----------
        u, v : nodes, optional (default=all edges)
            If u and v are specified, return the number of edges between
            u and v. Otherwise return the total number of all edges.
        Returns
        -------
        nedges : int
            The number of edges in the graph.  If nodes `u` and `v` are
            specified return the number of edges between those nodes. If
            the graph is directed, this only returns the number of edges
            from `u` to `v`.
        See Also
        --------
        size
        Examples
        --------
        For undirected graphs, this method counts the total number of
        edges in the graph:
        >>> G = nx.path_graph(4)
        >>> G.number_of_edges()
        3
        If you specify two nodes, this counts the total number of edges
        joining the two nodes:
        >>> G.number_of_edges(0, 1)
        1
        For directed graphs, this method can count the total number of
        directed edges from `u` to `v`:
        >>> G = nx.DiGraph()
        >>> G.add_edge(0, 1)
        >>> G.add_edge(1, 0)
        >>> G.number_of_edges(0, 1)
        1
        """
        if u is None:
            return int(self.size())
        if v in self._adj[u]:
            return 1
        return 0

    def nbunch_iter(self, nbunch=None):
        """Returns an iterator over nodes contained in nbunch that are
        also in the graph.
        The nodes in nbunch are checked for membership in the graph
        and if not are silently ignored.
        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.
        Returns
        -------
        niter : iterator
            An iterator over nodes in nbunch that are also in the graph.
            If nbunch is None, iterate over all nodes in the graph.
        Raises
        ------
        NetworkXError
            If nbunch is not a node or sequence of nodes.
            If a node in nbunch is not hashable.
        See Also
        --------
        Graph.__iter__
        Notes
        -----
        When nbunch is an iterator, the returned iterator yields values
        directly from nbunch, becoming exhausted when nbunch is exhausted.
        To test whether nbunch is a single node, one can use
        "if nbunch in self:", even after processing with this routine.
        If nbunch is not a node or a (possibly empty) sequence/iterator
        or None, a :exc:`NetworkXError` is raised.  Also, if any object in
        nbunch is not hashable, a :exc:`NetworkXError` is raised.
        """
        if nbunch is None:  # include all nodes via iterator
            bunch = iter(self._adj)
        elif nbunch in self:  # if nbunch is a single node
            bunch = iter([nbunch])
        else:  # if nbunch is a sequence of nodes

            def bunch_iter(nlist, adj):
                try:
                    for n in nlist:
                        if n in adj:
                            yield n
                except TypeError as err:
                    pass

            bunch = bunch_iter(nbunch, self._adj)
        return bunch

class DiGraph(Graph):
    def __init__(self, incoming_graph_data=None, **attr):
        self.graph_attr_dict_factory = self.graph_attr_dict_factory
        self.node_dict_factory = self.node_dict_factory
        self.node_attr_dict_factory = self.node_attr_dict_factory
        self.adjlist_outer_dict_factory = self.adjlist_outer_dict_factory
        self.adjlist_inner_dict_factory = self.adjlist_inner_dict_factory
        self.edge_attr_dict_factory = self.edge_attr_dict_factory

        self.graph = self.graph_attr_dict_factory()  # dictionary for graph attributes
        self._node = self.node_dict_factory()  # dictionary for node attr
        # We store two adjacency lists:
        # the predecessors of node n are stored in the dict self._pred
        # the successors of node n are stored in the dict self._succ=self._adj
        self._adj = self.adjlist_outer_dict_factory()  # empty adjacency dict
        self._pred = self.adjlist_outer_dict_factory()  # predecessor
        self._succ = self._adj  # successor
        self.graph.update(attr)

    @property
    def adj(self):
        """Graph adjacency object holding the neighbors of each node.
        This object is a read-only dict-like structure with node keys
        and neighbor-dict values.  The neighbor-dict is keyed by neighbor
        to the edge-data-dict.  So `G.adj[3][2]['color'] = 'blue'` sets
        the color of the edge `(3, 2)` to `"blue"`.
        Iterating over G.adj behaves like a dict. Useful idioms include
        `for nbr, datadict in G.adj[n].items():`.
        The neighbor information is also provided by subscripting the graph.
        So `for nbr, foovalue in G[node].data('foo', default=1):` works.
        For directed graphs, `G.adj` holds outgoing (successor) info.
        """
        return AdjacencyView(self._succ)

    @property
    def succ(self):
        """Graph adjacency object holding the successors of each node.
        This object is a read-only dict-like structure with node keys
        and neighbor-dict values.  The neighbor-dict is keyed by neighbor
        to the edge-data-dict.  So `G.succ[3][2]['color'] = 'blue'` sets
        the color of the edge `(3, 2)` to `"blue"`.
        Iterating over G.succ behaves like a dict. Useful idioms include
        `for nbr, datadict in G.succ[n].items():`.  A data-view not provided
        by dicts also exists: `for nbr, foovalue in G.succ[node].data('foo'):`
        and a default can be set via a `default` argument to the `data` method.
        The neighbor information is also provided by subscripting the graph.
        So `for nbr, foovalue in G[node].data('foo', default=1):` works.
        For directed graphs, `G.adj` is identical to `G.succ`.
        """
        return AdjacencyView(self._succ)

    @property
    def pred(self):
        """Graph adjacency object holding the predecessors of each node.
        This object is a read-only dict-like structure with node keys
        and neighbor-dict values.  The neighbor-dict is keyed by neighbor
        to the edge-data-dict.  So `G.pred[2][3]['color'] = 'blue'` sets
        the color of the edge `(3, 2)` to `"blue"`.
        Iterating over G.pred behaves like a dict. Useful idioms include
        `for nbr, datadict in G.pred[n].items():`.  A data-view not provided
        by dicts also exists: `for nbr, foovalue in G.pred[node].data('foo'):`
        A default can be set via a `default` argument to the `data` method.
        """
        return AdjacencyView(self._pred)

    def add_node(self, node_for_adding, **attr):
        """Add a single node `node_for_adding` and update node attributes.
        Parameters
        ----------
        node_for_adding : node
            A node can be any hashable Python object except None.
        attr : keyword arguments, optional
            Set or change node attributes using key=value.
        See Also
        --------
        add_nodes_from
        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_node(1)
        >>> G.add_node("Hello")
        >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> G.add_node(K3)
        >>> G.number_of_nodes()
        3
        Use keywords set/change node attributes:
        >>> G.add_node(1, size=10)
        >>> G.add_node(3, weight=0.4, UTM=("13S", 382871, 3972649))
        Notes
        -----
        A hashable object is one that can be used as a key in a Python
        dictionary. This includes strings, numbers, tuples of strings
        and numbers, etc.
        On many platforms hashable items also include mutables such as
        NetworkX Graphs, though one should be careful that the hash
        doesn't change on mutables.
        """
        if node_for_adding not in self._succ:
            if node_for_adding is None:
                raise ValueError("None cannot be a node")
            self._succ[node_for_adding] = self.adjlist_inner_dict_factory()
            self._pred[node_for_adding] = self.adjlist_inner_dict_factory()
            attr_dict = self._node[node_for_adding] = self.node_attr_dict_factory()
            attr_dict.update(attr)
        else:  # update attr even if node already exists
            self._node[node_for_adding].update(attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        """Add multiple nodes.
        Parameters
        ----------
        nodes_for_adding : iterable container
            A container of nodes (list, dict, set, etc.).
            OR
            A container of (node, attribute dict) tuples.
            Node attributes are updated using the attribute dict.
        attr : keyword arguments, optional (default= no attributes)
            Update attributes for all nodes in nodes.
            Node attributes specified in nodes as a tuple take
            precedence over attributes specified via keyword arguments.
        See Also
        --------
        add_node
        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_nodes_from("Hello")
        >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> G.add_nodes_from(K3)
        >>> sorted(G.nodes(), key=str)
        [0, 1, 2, 'H', 'e', 'l', 'o']
        Use keywords to update specific node attributes for every node.
        >>> G.add_nodes_from([1, 2], size=10)
        >>> G.add_nodes_from([3, 4], weight=0.4)
        Use (node, attrdict) tuples to update attributes for specific nodes.
        >>> G.add_nodes_from([(1, dict(size=11)), (2, {"color": "blue"})])
        >>> G.nodes[1]["size"]
        11
        >>> H = nx.Graph()
        >>> H.add_nodes_from(G.nodes(data=True))
        >>> H.nodes[1]["size"]
        11
        """
        for n in nodes_for_adding:
            try:
                newnode = n not in self._node
                newdict = attr
            except TypeError:
                n, ndict = n
                newnode = n not in self._node
                newdict = attr.copy()
                newdict.update(ndict)
            if newnode:
                if n is None:
                    raise ValueError("None cannot be a node")
                self._succ[n] = self.adjlist_inner_dict_factory()
                self._pred[n] = self.adjlist_inner_dict_factory()
                self._node[n] = self.node_attr_dict_factory()
            self._node[n].update(newdict)

    def remove_node(self, n):
        """Remove node n.
        Removes the node n and all adjacent edges.
        Attempting to remove a non-existent node will raise an exception.
        Parameters
        ----------
        n : node
           A node in the graph
        Raises
        ------
        NetworkXError
           If n is not in the graph.
        See Also
        --------
        remove_nodes_from
        Examples
        --------
        >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> list(G.edges)
        [(0, 1), (1, 2)]
        >>> G.remove_node(1)
        >>> list(G.edges)
        []
        """
        try:
            nbrs = self._succ[n]
            del self._node[n]
        except KeyError as err:  # NetworkXError if n not in self
            pass
        for u in nbrs:
            del self._pred[u][n]  # remove all edges n-u in digraph
        del self._succ[n]  # remove node from succ
        for u in self._pred[n]:
            del self._succ[u][n]  # remove all edges n-u in digraph
        del self._pred[n]  # remove node from pred

    def remove_nodes_from(self, nodes):
        """Remove multiple nodes.
        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, set, etc.).  If a node
            in the container is not in the graph it is silently ignored.
        See Also
        --------
        remove_node
        Examples
        --------
        >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> e = list(G.nodes)
        >>> e
        [0, 1, 2]
        >>> G.remove_nodes_from(e)
        >>> list(G.nodes)
        []
        """
        for n in nodes:
            try:
                succs = self._succ[n]
                del self._node[n]
                for u in succs:
                    del self._pred[u][n]  # remove all edges n-u in digraph
                del self._succ[n]  # now remove node
                for u in self._pred[n]:
                    del self._succ[u][n]  # remove all edges n-u in digraph
                del self._pred[n]  # now remove node
            except KeyError:
                pass  # silent failure on remove

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """Add an edge between u and v.
        The nodes u and v will be automatically added if they are
        not already in the graph.
        Edge attributes can be specified with keywords or by directly
        accessing the edge's attribute dictionary. See examples below.
        Parameters
        ----------
        u_of_edge, v_of_edge : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.
        See Also
        --------
        add_edges_from : add a collection of edges
        Notes
        -----
        Adding an edge that already exists updates the edge data.
        Many NetworkX algorithms designed for weighted graphs use
        an edge attribute (by default `weight`) to hold a numerical value.
        Examples
        --------
        The following all add the edge e=(1, 2) to graph G:
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> e = (1, 2)
        >>> G.add_edge(1, 2)  # explicit two-node form
        >>> G.add_edge(*e)  # single edge as tuple of two nodes
        >>> G.add_edges_from([(1, 2)])  # add edges from iterable container
        Associate data to edges using keywords:
        >>> G.add_edge(1, 2, weight=3)
        >>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)
        For non-string attribute keys, use subscript notation.
        >>> G.add_edge(1, 2)
        >>> G[1][2].update({0: 5})
        >>> G.edges[1, 2].update({0: 5})
        """
        u, v = u_of_edge, v_of_edge
        # add nodes
        if u not in self._succ:
            if u is None:
                raise ValueError("None cannot be a node")
            self._succ[u] = self.adjlist_inner_dict_factory()
            self._pred[u] = self.adjlist_inner_dict_factory()
            self._node[u] = self.node_attr_dict_factory()
        if v not in self._succ:
            if v is None:
                raise ValueError("None cannot be a node")
            self._succ[v] = self.adjlist_inner_dict_factory()
            self._pred[v] = self.adjlist_inner_dict_factory()
            self._node[v] = self.node_attr_dict_factory()
        # add the edge
        datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
        datadict.update(attr)
        self._succ[u][v] = datadict
        self._pred[v][u] = datadict

    def add_edges_from(self, ebunch_to_add, **attr):
        """Add all the edges in ebunch_to_add.
        Parameters
        ----------
        ebunch_to_add : container of edges
            Each edge given in the container will be added to the
            graph. The edges must be given as 2-tuples (u, v) or
            3-tuples (u, v, d) where d is a dictionary containing edge data.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.
        See Also
        --------
        add_edge : add a single edge
        add_weighted_edges_from : convenient way to add weighted edges
        Notes
        -----
        Adding the same edge twice has no effect but any edge data
        will be updated when each duplicate edge is added.
        Edge attributes specified in an ebunch take precedence over
        attributes specified via keyword arguments.
        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_edges_from([(0, 1), (1, 2)])  # using a list of edge tuples
        >>> e = zip(range(0, 3), range(1, 4))
        >>> G.add_edges_from(e)  # Add the path graph 0-1-2-3
        Associate data to edges
        >>> G.add_edges_from([(1, 2), (2, 3)], weight=3)
        >>> G.add_edges_from([(3, 4), (1, 4)], label="WN2898")
        """
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}
            else:
                pass
            if u not in self._succ:
                if u is None:
                    raise ValueError("None cannot be a node")
                self._succ[u] = self.adjlist_inner_dict_factory()
                self._pred[u] = self.adjlist_inner_dict_factory()
                self._node[u] = self.node_attr_dict_factory()
            if v not in self._succ:
                if v is None:
                    raise ValueError("None cannot be a node")
                self._succ[v] = self.adjlist_inner_dict_factory()
                self._pred[v] = self.adjlist_inner_dict_factory()
                self._node[v] = self.node_attr_dict_factory()
            datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
            datadict.update(attr)
            datadict.update(dd)
            self._succ[u][v] = datadict
            self._pred[v][u] = datadict

    def remove_edge(self, u, v):
        """Remove the edge between u and v.
        Parameters
        ----------
        u, v : nodes
            Remove the edge between nodes u and v.
        Raises
        ------
        NetworkXError
            If there is not an edge between u and v.
        See Also
        --------
        remove_edges_from : remove a collection of edges
        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, etc
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.remove_edge(0, 1)
        >>> e = (1, 2)
        >>> G.remove_edge(*e)  # unpacks e from an edge tuple
        >>> e = (2, 3, {"weight": 7})  # an edge with attribute data
        >>> G.remove_edge(*e[:2])  # select first part of edge tuple
        """
        try:
            del self._succ[u][v]
            del self._pred[v][u]
        except KeyError as err:
            pass

    def remove_edges_from(self, ebunch):
        """Remove all edges specified in ebunch.
        Parameters
        ----------
        ebunch: list or container of edge tuples
            Each edge given in the list or container will be removed
            from the graph. The edges can be:
                - 2-tuples (u, v) edge between u and v.
                - 3-tuples (u, v, k) where k is ignored.
        See Also
        --------
        remove_edge : remove a single edge
        Notes
        -----
        Will fail silently if an edge in ebunch is not in the graph.
        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> ebunch = [(1, 2), (2, 3)]
        >>> G.remove_edges_from(ebunch)
        """
        for e in ebunch:
            u, v = e[:2]  # ignore edge data
            if u in self._succ and v in self._succ[u]:
                del self._succ[u][v]
                del self._pred[v][u]

    def has_successor(self, u, v):
        """Returns True if node u has successor v.
        This is true if graph has the edge u->v.
        """
        return u in self._succ and v in self._succ[u]

    def has_predecessor(self, u, v):
        """Returns True if node u has predecessor v.
        This is true if graph has the edge u<-v.
        """
        return u in self._pred and v in self._pred[u]

    def successors(self, n):
        """Returns an iterator over successor nodes of n.
        A successor of n is a node m such that there exists a directed
        edge from n to m.
        Parameters
        ----------
        n : node
           A node in the graph
        Raises
        ------
        NetworkXError
           If n is not in the graph.
        See Also
        --------
        predecessors
        Notes
        -----
        neighbors() and successors() are the same.
        """
        try:
            return iter(self._succ[n])
        except KeyError as err:
            pass

    # digraph definitions
    neighbors = successors

    def predecessors(self, n):
        """Returns an iterator over predecessor nodes of n.
        A predecessor of n is a node m such that there exists a directed
        edge from m to n.
        Parameters
        ----------
        n : node
           A node in the graph
        Raises
        ------
        NetworkXError
           If n is not in the graph.
        See Also
        --------
        successors
        """
        try:
            return iter(self._pred[n])
        except KeyError as err:
            pass

    @property
    def edges(self):
        """An OutEdgeView of the DiGraph as G.edges or G.edges().
        edges(self, nbunch=None, data=False, default=None)
        The OutEdgeView provides set-like operations on the edge-tuples
        as well as edge attribute lookup. When called, it also provides
        an EdgeDataView object which allows control of access to edge
        attributes (but does not provide set-like operations).
        Hence, `G.edges[u, v]['color']` provides the value of the color
        attribute for edge `(u, v)` while
        `for (u, v, c) in G.edges.data('color', default='red'):`
        iterates through all the edges yielding the color attribute
        with default `'red'` if no color attribute exists.
        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges from these nodes.
        data : string or bool, optional (default=False)
            The edge attribute returned in 3-tuple (u, v, ddict[data]).
            If True, return edge attribute dict in 3-tuple (u, v, ddict).
            If False, return 2-tuple (u, v).
        default : value, optional (default=None)
            Value used for edges that don't have the requested attribute.
            Only relevant if data is not True or False.
        Returns
        -------
        edges : OutEdgeView
            A view of edge attributes, usually it iterates over (u, v)
            or (u, v, d) tuples of edges, but can also be used for
            attribute lookup as `edges[u, v]['foo']`.
        See Also
        --------
        in_edges, out_edges
        Notes
        -----
        Nodes in nbunch that are not in the graph will be (quietly) ignored.
        For directed graphs this returns the out-edges.
        Examples
        --------
        >>> G = nx.DiGraph()  # or MultiDiGraph, etc
        >>> nx.add_path(G, [0, 1, 2])
        >>> G.add_edge(2, 3, weight=5)
        >>> [e for e in G.edges]
        [(0, 1), (1, 2), (2, 3)]
        >>> G.edges.data()  # default data is {} (empty dict)
        OutEdgeDataView([(0, 1, {}), (1, 2, {}), (2, 3, {'weight': 5})])
        >>> G.edges.data("weight", default=1)
        OutEdgeDataView([(0, 1, 1), (1, 2, 1), (2, 3, 5)])
        >>> G.edges([0, 2])  # only edges originating from these nodes
        OutEdgeDataView([(0, 1), (2, 3)])
        >>> G.edges(0)  # only edges from node 0
        OutEdgeDataView([(0, 1)])
        """
        return OutEdgeView(self)

    # alias out_edges to edges
    out_edges = edges

    @property
    def in_edges(self):
        """An InEdgeView of the Graph as G.in_edges or G.in_edges().
        in_edges(self, nbunch=None, data=False, default=None):
        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.
        data : string or bool, optional (default=False)
            The edge attribute returned in 3-tuple (u, v, ddict[data]).
            If True, return edge attribute dict in 3-tuple (u, v, ddict).
            If False, return 2-tuple (u, v).
        default : value, optional (default=None)
            Value used for edges that don't have the requested attribute.
            Only relevant if data is not True or False.
        Returns
        -------
        in_edges : InEdgeView
            A view of edge attributes, usually it iterates over (u, v)
            or (u, v, d) tuples of edges, but can also be used for
            attribute lookup as `edges[u, v]['foo']`.
        See Also
        --------
        edges
        """
        return InEdgeView(self)

    @property
    def degree(self):
        """A DegreeView for the Graph as G.degree or G.degree().
        The node degree is the number of edges adjacent to the node.
        The weighted node degree is the sum of the edge weights for
        edges incident to that node.
        This object provides an iterator for (node, degree) as well as
        lookup for the degree for a single node.
        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.
        weight : string or None, optional (default=None)
           The name of an edge attribute that holds the numerical value used
           as a weight.  If None, then each edge has weight 1.
           The degree is the sum of the edge weights adjacent to the node.
        Returns
        -------
        If a single node is requested
        deg : int
            Degree of the node
        OR if multiple nodes are requested
        nd_iter : iterator
            The iterator returns two-tuples of (node, degree).
        See Also
        --------
        in_degree, out_degree
        Examples
        --------
        >>> G = nx.DiGraph()  # or MultiDiGraph
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.degree(0)  # node 0 with degree 1
        1
        >>> list(G.degree([0, 1, 2]))
        [(0, 1), (1, 2), (2, 2)]
        """
        return DiDegreeView(self)

    @property
    def in_degree(self):
        """An InDegreeView for (node, in_degree) or in_degree for single node.
        The node in_degree is the number of edges pointing to the node.
        The weighted node degree is the sum of the edge weights for
        edges incident to that node.
        This object provides an iteration over (node, in_degree) as well as
        lookup for the degree for a single node.
        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.
        weight : string or None, optional (default=None)
           The name of an edge attribute that holds the numerical value used
           as a weight.  If None, then each edge has weight 1.
           The degree is the sum of the edge weights adjacent to the node.
        Returns
        -------
        If a single node is requested
        deg : int
            In-degree of the node
        OR if multiple nodes are requested
        nd_iter : iterator
            The iterator returns two-tuples of (node, in-degree).
        See Also
        --------
        degree, out_degree
        Examples
        --------
        >>> G = nx.DiGraph()
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.in_degree(0)  # node 0 with degree 0
        0
        >>> list(G.in_degree([0, 1, 2]))
        [(0, 0), (1, 1), (2, 1)]
        """
        return InDegreeView(self)

    @property
    def out_degree(self):
        """An OutDegreeView for (node, out_degree)
        The node out_degree is the number of edges pointing out of the node.
        The weighted node degree is the sum of the edge weights for
        edges incident to that node.
        This object provides an iterator over (node, out_degree) as well as
        lookup for the degree for a single node.
        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.
        weight : string or None, optional (default=None)
           The name of an edge attribute that holds the numerical value used
           as a weight.  If None, then each edge has weight 1.
           The degree is the sum of the edge weights adjacent to the node.
        Returns
        -------
        If a single node is requested
        deg : int
            Out-degree of the node
        OR if multiple nodes are requested
        nd_iter : iterator
            The iterator returns two-tuples of (node, out-degree).
        See Also
        --------
        degree, in_degree
        Examples
        --------
        >>> G = nx.DiGraph()
        >>> nx.add_path(G, [0, 1, 2, 3])
        >>> G.out_degree(0)  # node 0 with degree 1
        1
        >>> list(G.out_degree([0, 1, 2]))
        [(0, 1), (1, 1), (2, 1)]
        """
        return OutDegreeView(self)

    def clear(self):
        """Remove all nodes and edges from the graph.
        This also removes the name, and all graph, node, and edge attributes.
        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.clear()
        >>> list(G.nodes)
        []
        >>> list(G.edges)
        []
        """
        self._succ.clear()
        self._pred.clear()
        self._node.clear()
        self.graph.clear()

    def clear_edges(self):
        """Remove all edges from the graph without altering nodes.
        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.clear_edges()
        >>> list(G.nodes)
        [0, 1, 2, 3]
        >>> list(G.edges)
        []
        """
        for predecessor_dict in self._pred.values():
            predecessor_dict.clear()
        for successor_dict in self._succ.values():
            successor_dict.clear()

    def is_multigraph(self):
        """Returns True if graph is a multigraph, False otherwise."""
        return False

    def is_directed(self):
        """Returns True if graph is directed, False otherwise."""
        return True

    def to_undirected(self, reciprocal=False, as_view=False):
        """Returns an undirected representation of the digraph.
        Parameters
        ----------
        reciprocal : bool (optional)
          If True only keep edges that appear in both directions
          in the original digraph.
        as_view : bool (optional, default=False)
          If True return an undirected view of the original directed graph.
        Returns
        -------
        G : Graph
            An undirected graph with the same name and nodes and
            with edge (u, v, data) if either (u, v, data) or (v, u, data)
            is in the digraph.  If both edges exist in digraph and
            their edge data is different, only one edge is created
            with an arbitrary choice of which edge data to use.
            You must check and correct for this manually if desired.
        See Also
        --------
        Graph, copy, add_edge, add_edges_from
        Notes
        -----
        If edges in both directions (u, v) and (v, u) exist in the
        graph, attributes for the new undirected edge will be a combination of
        the attributes of the directed edges.  The edge data is updated
        in the (arbitrary) order that the edges are encountered.  For
        more customized control of the edge attributes use add_edge().
        This returns a "deepcopy" of the edge, node, and
        graph attributes which attempts to completely copy
        all of the data and references.
        This is in contrast to the similar G=DiGraph(D) which returns a
        shallow copy of the data.
        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.
        Warning: If you have subclassed DiGraph to use dict-like objects
        in the data structure, those changes do not transfer to the
        Graph created by this method.
        Examples
        --------
        >>> G = nx.path_graph(2)  # or MultiGraph, etc
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1), (1, 0)]
        >>> G2 = H.to_undirected()
        >>> list(G2.edges)
        [(0, 1)]
        """
        graph_class = self.to_undirected_class()
        # deepcopy when not a view
        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
        if reciprocal is True:
            G.add_edges_from(
                (u, v, deepcopy(d))
                for u, nbrs in self._adj.items()
                for v, d in nbrs.items()
                if v in self._pred[u]
            )
        else:
            G.add_edges_from(
                (u, v, deepcopy(d))
                for u, nbrs in self._adj.items()
                for v, d in nbrs.items()
            )
        return G
import warnings
from collections.abc import Mapping
class AtlasView(Mapping):
    """An AtlasView is a Read-only Mapping of Mappings.
    It is a View into a dict-of-dict data structure.
    The inner level of dict is read-write. But the
    outer level is read-only.
    See Also
    ========
    AdjacencyView: View into dict-of-dict-of-dict
    MultiAdjacencyView: View into dict-of-dict-of-dict-of-dict
    """

    __slots__ = ("_atlas",)

    def __getstate__(self):
        return {"_atlas": self._atlas}

    def __setstate__(self, state):
        self._atlas = state["_atlas"]

    def __init__(self, d):
        self._atlas = d

    def __len__(self):
        return len(self._atlas)

    def __iter__(self):
        return iter(self._atlas)

    def __getitem__(self, key):
        return self._atlas[key]

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}

    def __str__(self):
        return str(self._atlas)  # {nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return f"{self.__class__.__name__}({self._atlas!r})"


class AdjacencyView(AtlasView):
    """An AdjacencyView is a Read-only Map of Maps of Maps.
    It is a View into a dict-of-dict-of-dict data structure.
    The inner level of dict is read-write. But the
    outer levels are read-only.
    See Also
    ========
    AtlasView: View into dict-of-dict
    MultiAdjacencyView: View into dict-of-dict-of-dict-of-dict
    """

    __slots__ = ()  # Still uses AtlasView slots names _atlas

    def __getitem__(self, name):
        return AtlasView(self._atlas[name])

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}


class MultiAdjacencyView(AdjacencyView):
    """An MultiAdjacencyView is a Read-only Map of Maps of Maps of Maps.
    It is a View into a dict-of-dict-of-dict-of-dict data structure.
    The inner level of dict is read-write. But the
    outer levels are read-only.
    See Also
    ========
    AtlasView: View into dict-of-dict
    AdjacencyView: View into dict-of-dict-of-dict
    """

    __slots__ = ()  # Still uses AtlasView slots names _atlas

    def __getitem__(self, name):
        return AdjacencyView(self._atlas[name])

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}


class UnionAtlas(Mapping):
    """A read-only union of two atlases (dict-of-dict).
    The two dict-of-dicts represent the inner dict of
    an Adjacency:  `G.succ[node]` and `G.pred[node]`.
    The inner level of dict of both hold attribute key:value
    pairs and is read-write. But the outer level is read-only.
    See Also
    ========
    UnionAdjacency: View into dict-of-dict-of-dict
    UnionMultiAdjacency: View into dict-of-dict-of-dict-of-dict
    """

    __slots__ = ("_succ", "_pred")

    def __getstate__(self):
        return {"_succ": self._succ, "_pred": self._pred}

    def __setstate__(self, state):
        self._succ = state["_succ"]
        self._pred = state["_pred"]

    def __init__(self, succ, pred):
        self._succ = succ
        self._pred = pred

    def __len__(self):
        return len(self._succ) + len(self._pred)

    def __iter__(self):
        return iter(set(self._succ.keys()) | set(self._pred.keys()))

    def __getitem__(self, key):
        try:
            return self._succ[key]
        except KeyError:
            return self._pred[key]

    def copy(self):
        result = {nbr: dd.copy() for nbr, dd in self._succ.items()}
        for nbr, dd in self._pred.items():
            if nbr in result:
                result[nbr].update(dd)
            else:
                result[nbr] = dd.copy()
        return result

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return f"{self.__class__.__name__}({self._succ!r}, {self._pred!r})"


class UnionAdjacency(Mapping):
    """A read-only union of dict Adjacencies as a Map of Maps of Maps.
    The two input dict-of-dict-of-dicts represent the union of
    `G.succ` and `G.pred`. Return values are UnionAtlas
    The inner level of dict is read-write. But the
    middle and outer levels are read-only.
    succ : a dict-of-dict-of-dict {node: nbrdict}
    pred : a dict-of-dict-of-dict {node: nbrdict}
    The keys for the two dicts should be the same
    See Also
    ========
    UnionAtlas: View into dict-of-dict
    UnionMultiAdjacency: View into dict-of-dict-of-dict-of-dict
    """

    __slots__ = ("_succ", "_pred")

    def __getstate__(self):
        return {"_succ": self._succ, "_pred": self._pred}

    def __setstate__(self, state):
        self._succ = state["_succ"]
        self._pred = state["_pred"]

    def __init__(self, succ, pred):
        # keys must be the same for two input dicts
        assert len(set(succ.keys()) ^ set(pred.keys())) == 0
        self._succ = succ
        self._pred = pred

    def __len__(self):
        return len(self._succ)  # length of each dict should be the same

    def __iter__(self):
        return iter(self._succ)

    def __getitem__(self, nbr):
        return UnionAtlas(self._succ[nbr], self._pred[nbr])

    def copy(self):
        return {n: self[n].copy() for n in self._succ}

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return f"{self.__class__.__name__}({self._succ!r}, {self._pred!r})"


class UnionMultiInner(UnionAtlas):
    """A read-only union of two inner dicts of MultiAdjacencies.
    The two input dict-of-dict-of-dicts represent the union of
    `G.succ[node]` and `G.pred[node]` for MultiDiGraphs.
    Return values are UnionAtlas.
    The inner level of dict is read-write. But the outer levels are read-only.
    See Also
    ========
    UnionAtlas: View into dict-of-dict
    UnionAdjacency:  View into dict-of-dict-of-dict
    UnionMultiAdjacency:  View into dict-of-dict-of-dict-of-dict
    """

    __slots__ = ()  # Still uses UnionAtlas slots names _succ, _pred

    def __getitem__(self, node):
        in_succ = node in self._succ
        in_pred = node in self._pred
        if in_succ:
            if in_pred:
                return UnionAtlas(self._succ[node], self._pred[node])
            return UnionAtlas(self._succ[node], {})
        return UnionAtlas({}, self._pred[node])

    def copy(self):
        nodes = set(self._succ.keys()) | set(self._pred.keys())
        return {n: self[n].copy() for n in nodes}


class UnionMultiAdjacency(UnionAdjacency):
    """A read-only union of two dict MultiAdjacencies.
    The two input dict-of-dict-of-dict-of-dicts represent the union of
    `G.succ` and `G.pred` for MultiDiGraphs. Return values are UnionAdjacency.
    The inner level of dict is read-write. But the outer levels are read-only.
    See Also
    ========
    UnionAtlas:  View into dict-of-dict
    UnionMultiInner:  View into dict-of-dict-of-dict
    """

    __slots__ = ()  # Still uses UnionAdjacency slots names _succ, _pred

    def __getitem__(self, node):
        return UnionMultiInner(self._succ[node], self._pred[node])


class FilterAtlas(Mapping):  # nodedict, nbrdict, keydict
    def __init__(self, d, NODE_OK):
        self._atlas = d
        self.NODE_OK = NODE_OK

    def __len__(self):
        return sum(1 for n in self)

    def __iter__(self):
        try:  # check that NODE_OK has attr 'nodes'
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            return (n for n in self.NODE_OK.nodes if n in self._atlas)
        return (n for n in self._atlas if self.NODE_OK(n))

    def __getitem__(self, key):
        if key in self._atlas and self.NODE_OK(key):
            return self._atlas[key]
        raise KeyError(f"Key {key} not found")

    # FIXME should this just be removed? we don't use it, but someone might
    def copy(self):
        warnings.warn(
            (
                "FilterAtlas.copy is deprecated.\n"
                "It will be removed in NetworkX 3.0.\n"
                "Please open an Issue on https://github.com/networkx/networkx/issues\n"
                "if you use this feature. We think that no one does use it."
            ),
            DeprecationWarning,
        )
        try:  # check that NODE_OK has attr 'nodes'
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            return {u: self._atlas[u] for u in self.NODE_OK.nodes if u in self._atlas}
        return {u: d for u, d in self._atlas.items() if self.NODE_OK(u)}

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return f"{self.__class__.__name__}({self._atlas!r}, {self.NODE_OK!r})"


class FilterAdjacency(Mapping):  # edgedict
    def __init__(self, d, NODE_OK, EDGE_OK):
        self._atlas = d
        self.NODE_OK = NODE_OK
        self.EDGE_OK = EDGE_OK

    def __len__(self):
        return sum(1 for n in self)

    def __iter__(self):
        try:  # check that NODE_OK has attr 'nodes'
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            return (n for n in self.NODE_OK.nodes if n in self._atlas)
        return (n for n in self._atlas if self.NODE_OK(n))

    def __getitem__(self, node):
        if node in self._atlas and self.NODE_OK(node):

            def new_node_ok(nbr):
                return self.NODE_OK(nbr) and self.EDGE_OK(node, nbr)

            return FilterAtlas(self._atlas[node], new_node_ok)
        raise KeyError(f"Key {node} not found")

    # FIXME should this just be removed? we don't use it, but someone might
    def copy(self):
        warnings.warn(
            (
                "FilterAdjacency.copy is deprecated.\n"
                "It will be removed in NetworkX 3.0.\n"
                "Please open an Issue on https://github.com/networkx/networkx/issues\n"
                "if you use this feature. We think that no one does use it."
            ),
            DeprecationWarning,
        )
        try:  # check that NODE_OK has attr 'nodes'
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            return {
                u: {
                    v: d
                    for v, d in self._atlas[u].items()
                    if self.NODE_OK(v)
                    if self.EDGE_OK(u, v)
                }
                for u in self.NODE_OK.nodes
                if u in self._atlas
            }
        return {
            u: {v: d for v, d in nbrs.items() if self.NODE_OK(v) if self.EDGE_OK(u, v)}
            for u, nbrs in self._atlas.items()
            if self.NODE_OK(u)
        }

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({self._atlas!r}, {self.NODE_OK!r}, {self.EDGE_OK!r})"


class FilterMultiInner(FilterAdjacency):  # muliedge_seconddict
    def __iter__(self):
        try:  # check that NODE_OK has attr 'nodes'
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            my_nodes = (n for n in self.NODE_OK.nodes if n in self._atlas)
        else:
            my_nodes = (n for n in self._atlas if self.NODE_OK(n))
        for n in my_nodes:
            some_keys_ok = False
            for key in self._atlas[n]:
                if self.EDGE_OK(n, key):
                    some_keys_ok = True
                    break
            if some_keys_ok is True:
                yield n

    def __getitem__(self, nbr):
        if nbr in self._atlas and self.NODE_OK(nbr):

            def new_node_ok(key):
                return self.EDGE_OK(nbr, key)

            return FilterAtlas(self._atlas[nbr], new_node_ok)
        raise KeyError(f"Key {nbr} not found")

    # FIXME should this just be removed? we don't use it, but someone might
    def copy(self):
        warnings.warn(
            (
                "FilterMultiInner.copy is deprecated.\n"
                "It will be removed in NetworkX 3.0.\n"
                "Please open an Issue on https://github.com/networkx/networkx/issues\n"
                "if you use this feature. We think that no one does use it."
            ),
            DeprecationWarning,
        )
        try:  # check that NODE_OK has attr 'nodes'
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            return {
                v: {k: d for k, d in self._atlas[v].items() if self.EDGE_OK(v, k)}
                for v in self.NODE_OK.nodes
                if v in self._atlas
            }
        return {
            v: {k: d for k, d in nbrs.items() if self.EDGE_OK(v, k)}
            for v, nbrs in self._atlas.items()
            if self.NODE_OK(v)
        }


class FilterMultiAdjacency(FilterAdjacency):  # multiedgedict
    def __getitem__(self, node):
        if node in self._atlas and self.NODE_OK(node):

            def edge_ok(nbr, key):
                return self.NODE_OK(nbr) and self.EDGE_OK(node, nbr, key)

            return FilterMultiInner(self._atlas[node], self.NODE_OK, edge_ok)
        raise KeyError(f"Key {node} not found")

    # FIXME should this just be removed? we don't use it, but someone might
    def copy(self):
        warnings.warn(
            (
                "FilterMultiAdjacency.copy is deprecated.\n"
                "It will be removed in NetworkX 3.0.\n"
                "Please open an Issue on https://github.com/networkx/networkx/issues\n"
                "if you use this feature. We think that no one does use it."
            ),
            DeprecationWarning,
        )
        try:  # check that NODE_OK has attr 'nodes'
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            my_nodes = self.NODE_OK.nodes
            return {
                u: {
                    v: {k: d for k, d in kd.items() if self.EDGE_OK(u, v, k)}
                    for v, kd in self._atlas[u].items()
                    if v in my_nodes
                }
                for u in my_nodes
                if u in self._atlas
            }
        return {
            u: {
                v: {k: d for k, d in kd.items() if self.EDGE_OK(u, v, k)}
                for v, kd in nbrs.items()
                if self.NODE_OK(v)
            }
            for u, nbrs in self._atlas.items()
            if self.NODE_OK(u)
        }

from collections.abc import Mapping, Set
# NodeViews
class NodeView(Mapping, Set):
    """A NodeView class to act as G.nodes for a NetworkX Graph
    Set operations act on the nodes without considering data.
    Iteration is over nodes. Node data can be looked up like a dict.
    Use NodeDataView to iterate over node data or to specify a data
    attribute for lookup. NodeDataView is created by calling the NodeView.
    Parameters
    ----------
    graph : NetworkX graph-like class
    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> NV = G.nodes()
    >>> 2 in NV
    True
    >>> for n in NV:
    ...     print(n)
    0
    1
    2
    >>> assert NV & {1, 2, 3} == {1, 2}
    >>> G.add_node(2, color="blue")
    >>> NV[2]
    {'color': 'blue'}
    >>> G.add_node(8, color="red")
    >>> NDV = G.nodes(data=True)
    >>> (2, NV[2]) in NDV
    True
    >>> for n, dd in NDV:
    ...     print((n, dd.get("color", "aqua")))
    (0, 'aqua')
    (1, 'aqua')
    (2, 'blue')
    (8, 'red')
    >>> NDV[2] == NV[2]
    True
    >>> NVdata = G.nodes(data="color", default="aqua")
    >>> (2, NVdata[2]) in NVdata
    True
    >>> for n, dd in NVdata:
    ...     print((n, dd))
    (0, 'aqua')
    (1, 'aqua')
    (2, 'blue')
    (8, 'red')
    >>> NVdata[2] == NV[2]  # NVdata gets 'color', NV gets datadict
    False
    """

    __slots__ = ("_nodes",)

    def __getstate__(self):
        return {"_nodes": self._nodes}

    def __setstate__(self, state):
        self._nodes = state["_nodes"]

    def __init__(self, graph):
        self._nodes = graph._node

    # Mapping methods
    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __getitem__(self, n):
        return self._nodes[n]

    # Set methods
    def __contains__(self, n):
        return n in self._nodes

    @classmethod
    def _from_iterable(cls, it):
        return set(it)

    # DataView method
    def __call__(self, data=False, default=None):
        if data is False:
            return self
        return NodeDataView(self._nodes, data, default)

    def data(self, data=True, default=None):
        """
        Return a read-only view of node data.
        Parameters
        ----------
        data : bool or node data key, default=True
            If ``data=True`` (the default), return a `NodeDataView` object that
            maps each node to *all* of its attributes. `data` may also be an
            arbitrary key, in which case the `NodeDataView` maps each node to
            the value for the keyed attribute. In this case, if a node does
            not have the `data` attribute, the `default` value is used.
        default : object, default=None
            The value used when a node does not have a specific attribute.
        Returns
        -------
        NodeDataView
            The layout of the returned NodeDataView depends on the value of the
            `data` parameter.
        Notes
        -----
        If ``data=False``, returns a `NodeView` object without data.
        See Also
        --------
        NodeDataView
        Examples
        --------
        >>> G = nx.Graph()
        >>> G.add_nodes_from([
        ...     (0, {"color": "red", "weight": 10}),
        ...     (1, {"color": "blue"}),
        ...     (2, {"color": "yellow", "weight": 2})
        ... ])
        Accessing node data with ``data=True`` (the default) returns a
        NodeDataView mapping each node to all of its attributes:
        >>> G.nodes.data()
        NodeDataView({0: {'color': 'red', 'weight': 10}, 1: {'color': 'blue'}, 2: {'color': 'yellow', 'weight': 2}})
        If `data` represents  a key in the node attribute dict, a NodeDataView mapping
        the nodes to the value for that specific key is returned:
        >>> G.nodes.data("color")
        NodeDataView({0: 'red', 1: 'blue', 2: 'yellow'}, data='color')
        If a specific key is not found in an attribute dict, the value specified
        by `default` is returned:
        >>> G.nodes.data("weight", default=-999)
        NodeDataView({0: 10, 1: -999, 2: 2}, data='weight')
        Note that there is no check that the `data` key is in any of the
        node attribute dictionaries:
        >>> G.nodes.data("height")
        NodeDataView({0: None, 1: None, 2: None}, data='height')
        """
        if data is False:
            return self
        return NodeDataView(self._nodes, data, default)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return f"{self.__class__.__name__}({tuple(self)})"


class NodeDataView(Set):
    """A DataView class for nodes of a NetworkX Graph
    The main use for this class is to iterate through node-data pairs.
    The data can be the entire data-dictionary for each node, or it
    can be a specific attribute (with default) for each node.
    Set operations are enabled with NodeDataView, but don't work in
    cases where the data is not hashable. Use with caution.
    Typically, set operations on nodes use NodeView, not NodeDataView.
    That is, they use `G.nodes` instead of `G.nodes(data='foo')`.
    Parameters
    ==========
    graph : NetworkX graph-like class
    data : bool or string (default=False)
    default : object (default=None)
    """

    __slots__ = ("_nodes", "_data", "_default")

    def __getstate__(self):
        return {"_nodes": self._nodes, "_data": self._data, "_default": self._default}

    def __setstate__(self, state):
        self._nodes = state["_nodes"]
        self._data = state["_data"]
        self._default = state["_default"]

    def __init__(self, nodedict, data=False, default=None):
        self._nodes = nodedict
        self._data = data
        self._default = default

    @classmethod
    def _from_iterable(cls, it):
        try:
            return set(it)
        except TypeError as err:
            if "unhashable" in str(err):
                msg = " : Could be b/c data=True or your values are unhashable"
                raise TypeError(str(err) + msg) from err
            raise

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        data = self._data
        if data is False:
            return iter(self._nodes)
        if data is True:
            return iter(self._nodes.items())
        return (
            (n, dd[data] if data in dd else self._default)
            for n, dd in self._nodes.items()
        )

    def __contains__(self, n):
        try:
            node_in = n in self._nodes
        except TypeError:
            n, d = n
            return n in self._nodes and self[n] == d
        if node_in is True:
            return node_in
        try:
            n, d = n
        except (TypeError, ValueError):
            return False
        return n in self._nodes and self[n] == d

    def __getitem__(self, n):
        ddict = self._nodes[n]
        data = self._data
        if data is False or data is True:
            return ddict
        return ddict[data] if data in ddict else self._default

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        name = self.__class__.__name__
        if self._data is False:
            return f"{name}({tuple(self)})"
        if self._data is True:
            return f"{name}({dict(self)})"
        return f"{name}({dict(self)}, data={self._data!r})"


# DegreeViews
class DiDegreeView:
    """A View class for degree of nodes in a NetworkX Graph
    The functionality is like dict.items() with (node, degree) pairs.
    Additional functionality includes read-only lookup of node degree,
    and calling with optional features nbunch (for only a subset of nodes)
    and weight (use edge weights to compute degree).
    Parameters
    ==========
    graph : NetworkX graph-like class
    nbunch : node, container of nodes, or None meaning all nodes (default=None)
    weight : bool or string (default=None)
    Notes
    -----
    DegreeView can still lookup any node even if nbunch is specified.
    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> DV = G.degree()
    >>> assert DV[2] == 1
    >>> assert sum(deg for n, deg in DV) == 4
    >>> DVweight = G.degree(weight="span")
    >>> G.add_edge(1, 2, span=34)
    >>> DVweight[2]
    34
    >>> DVweight[0]  #  default edge weight is 1
    1
    >>> sum(span for n, span in DVweight)  # sum weighted degrees
    70
    >>> DVnbunch = G.degree(nbunch=(1, 2))
    >>> assert len(list(DVnbunch)) == 2  # iteration over nbunch only
    """

    def __init__(self, G, nbunch=None, weight=None):
        self._graph = G
        self._succ = G._succ if hasattr(G, "_succ") else G._adj
        self._pred = G._pred if hasattr(G, "_pred") else G._adj
        self._nodes = self._succ if nbunch is None else list(G.nbunch_iter(nbunch))
        self._weight = weight

    def __call__(self, nbunch=None, weight=None):
        if nbunch is None:
            if weight == self._weight:
                return self
            return self.__class__(self._graph, None, weight)
        try:
            if nbunch in self._nodes:
                if weight == self._weight:
                    return self[nbunch]
                return self.__class__(self._graph, None, weight)[nbunch]
        except TypeError:
            pass
        return self.__class__(self._graph, nbunch, weight)

    def __getitem__(self, n):
        weight = self._weight
        succs = self._succ[n]
        preds = self._pred[n]
        if weight is None:
            return len(succs) + len(preds)
        return sum(dd.get(weight, 1) for dd in succs.values()) + sum(
            dd.get(weight, 1) for dd in preds.values()
        )

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                succs = self._succ[n]
                preds = self._pred[n]
                yield (n, len(succs) + len(preds))
        else:
            for n in self._nodes:
                succs = self._succ[n]
                preds = self._pred[n]
                deg = sum(dd.get(weight, 1) for dd in succs.values()) + sum(
                    dd.get(weight, 1) for dd in preds.values()
                )
                yield (n, deg)

    def __len__(self):
        return len(self._nodes)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return f"{self.__class__.__name__}({dict(self)})"


class DegreeView(DiDegreeView):
    """A DegreeView class to act as G.degree for a NetworkX Graph
    Typical usage focuses on iteration over `(node, degree)` pairs.
    The degree is by default the number of edges incident to the node.
    Optional argument `weight` enables weighted degree using the edge
    attribute named in the `weight` argument.  Reporting and iteration
    can also be restricted to a subset of nodes using `nbunch`.
    Additional functionality include node lookup so that `G.degree[n]`
    reported the (possibly weighted) degree of node `n`. Calling the
    view creates a view with different arguments `nbunch` or `weight`.
    Parameters
    ==========
    graph : NetworkX graph-like class
    nbunch : node, container of nodes, or None meaning all nodes (default=None)
    weight : string or None (default=None)
    Notes
    -----
    DegreeView can still lookup any node even if nbunch is specified.
    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> DV = G.degree()
    >>> assert DV[2] == 1
    >>> assert G.degree[2] == 1
    >>> assert sum(deg for n, deg in DV) == 4
    >>> DVweight = G.degree(weight="span")
    >>> G.add_edge(1, 2, span=34)
    >>> DVweight[2]
    34
    >>> DVweight[0]  #  default edge weight is 1
    1
    >>> sum(span for n, span in DVweight)  # sum weighted degrees
    70
    >>> DVnbunch = G.degree(nbunch=(1, 2))
    >>> assert len(list(DVnbunch)) == 2  # iteration over nbunch only
    """

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._succ[n]
        if weight is None:
            return len(nbrs) + (n in nbrs)
        return sum(dd.get(weight, 1) for dd in nbrs.values()) + (
            n in nbrs and nbrs[n].get(weight, 1)
        )

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                nbrs = self._succ[n]
                yield (n, len(nbrs) + (n in nbrs))
        else:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum(dd.get(weight, 1) for dd in nbrs.values()) + (
                    n in nbrs and nbrs[n].get(weight, 1)
                )
                yield (n, deg)


class OutDegreeView(DiDegreeView):
    """A DegreeView class to report out_degree for a DiGraph; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._succ[n]
        if self._weight is None:
            return len(nbrs)
        return sum(dd.get(self._weight, 1) for dd in nbrs.values())

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                succs = self._succ[n]
                yield (n, len(succs))
        else:
            for n in self._nodes:
                succs = self._succ[n]
                deg = sum(dd.get(weight, 1) for dd in succs.values())
                yield (n, deg)


class InDegreeView(DiDegreeView):
    """A DegreeView class to report in_degree for a DiGraph; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._pred[n]
        if weight is None:
            return len(nbrs)
        return sum(dd.get(weight, 1) for dd in nbrs.values())

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                preds = self._pred[n]
                yield (n, len(preds))
        else:
            for n in self._nodes:
                preds = self._pred[n]
                deg = sum(dd.get(weight, 1) for dd in preds.values())
                yield (n, deg)


class MultiDegreeView(DiDegreeView):
    """A DegreeView class for undirected multigraphs; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._succ[n]
        if weight is None:
            return sum(len(keys) for keys in nbrs.values()) + (
                n in nbrs and len(nbrs[n])
            )
        # edge weighted graph - degree is sum of nbr edge weights
        deg = sum(
            d.get(weight, 1) for key_dict in nbrs.values() for d in key_dict.values()
        )
        if n in nbrs:
            deg += sum(d.get(weight, 1) for d in nbrs[n].values())
        return deg

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum(len(keys) for keys in nbrs.values()) + (
                    n in nbrs and len(nbrs[n])
                )
                yield (n, deg)
        else:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum(
                    d.get(weight, 1)
                    for key_dict in nbrs.values()
                    for d in key_dict.values()
                )
                if n in nbrs:
                    deg += sum(d.get(weight, 1) for d in nbrs[n].values())
                yield (n, deg)


class DiMultiDegreeView(DiDegreeView):
    """A DegreeView class for MultiDiGraph; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        succs = self._succ[n]
        preds = self._pred[n]
        if weight is None:
            return sum(len(keys) for keys in succs.values()) + sum(
                len(keys) for keys in preds.values()
            )
        # edge weighted graph - degree is sum of nbr edge weights
        deg = sum(
            d.get(weight, 1) for key_dict in succs.values() for d in key_dict.values()
        ) + sum(
            d.get(weight, 1) for key_dict in preds.values() for d in key_dict.values()
        )
        return deg

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                succs = self._succ[n]
                preds = self._pred[n]
                deg = sum(len(keys) for keys in succs.values()) + sum(
                    len(keys) for keys in preds.values()
                )
                yield (n, deg)
        else:
            for n in self._nodes:
                succs = self._succ[n]
                preds = self._pred[n]
                deg = sum(
                    d.get(weight, 1)
                    for key_dict in succs.values()
                    for d in key_dict.values()
                ) + sum(
                    d.get(weight, 1)
                    for key_dict in preds.values()
                    for d in key_dict.values()
                )
                yield (n, deg)


class InMultiDegreeView(DiDegreeView):
    """A DegreeView class for inward degree of MultiDiGraph; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._pred[n]
        if weight is None:
            return sum(len(data) for data in nbrs.values())
        # edge weighted graph - degree is sum of nbr edge weights
        return sum(
            d.get(weight, 1) for key_dict in nbrs.values() for d in key_dict.values()
        )

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                nbrs = self._pred[n]
                deg = sum(len(data) for data in nbrs.values())
                yield (n, deg)
        else:
            for n in self._nodes:
                nbrs = self._pred[n]
                deg = sum(
                    d.get(weight, 1)
                    for key_dict in nbrs.values()
                    for d in key_dict.values()
                )
                yield (n, deg)


class OutMultiDegreeView(DiDegreeView):
    """A DegreeView class for outward degree of MultiDiGraph; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._succ[n]
        if weight is None:
            return sum(len(data) for data in nbrs.values())
        # edge weighted graph - degree is sum of nbr edge weights
        return sum(
            d.get(weight, 1) for key_dict in nbrs.values() for d in key_dict.values()
        )

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum(len(data) for data in nbrs.values())
                yield (n, deg)
        else:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum(
                    d.get(weight, 1)
                    for key_dict in nbrs.values()
                    for d in key_dict.values()
                )
                yield (n, deg)


# EdgeDataViews
class OutEdgeDataView:
    """EdgeDataView for outward edges of DiGraph; See EdgeDataView"""

    __slots__ = (
        "_viewer",
        "_nbunch",
        "_data",
        "_default",
        "_adjdict",
        "_nodes_nbrs",
        "_report",
    )

    def __getstate__(self):
        return {
            "viewer": self._viewer,
            "nbunch": self._nbunch,
            "data": self._data,
            "default": self._default,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    def __init__(self, viewer, nbunch=None, data=False, default=None):
        self._viewer = viewer
        adjdict = self._adjdict = viewer._adjdict
        if nbunch is None:
            self._nodes_nbrs = adjdict.items
        else:
            # dict retains order of nodes but acts like a set
            nbunch = dict.fromkeys(viewer._graph.nbunch_iter(nbunch))
            self._nodes_nbrs = lambda: [(n, adjdict[n]) for n in nbunch]
        self._nbunch = nbunch
        self._data = data
        self._default = default
        # Set _report based on data and default
        if data is True:
            self._report = lambda n, nbr, dd: (n, nbr, dd)
        elif data is False:
            self._report = lambda n, nbr, dd: (n, nbr)
        else:  # data is attribute name
            self._report = (
                lambda n, nbr, dd: (n, nbr, dd[data])
                if data in dd
                else (n, nbr, default)
            )

    def __len__(self):
        return sum(len(nbrs) for n, nbrs in self._nodes_nbrs())

    def __iter__(self):
        return (
            self._report(n, nbr, dd)
            for n, nbrs in self._nodes_nbrs()
            for nbr, dd in nbrs.items()
        )

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and u not in self._nbunch:
            return False  # this edge doesn't start in nbunch
        try:
            ddict = self._adjdict[u][v]
        except KeyError:
            return False
        return e == self._report(u, v, ddict)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)})"


class EdgeDataView(OutEdgeDataView):
    __slots__ = ()

    def __len__(self):
        return sum(1 for e in self)

    def __iter__(self):
        seen = {}
        for n, nbrs in self._nodes_nbrs():
            for nbr, dd in nbrs.items():
                if nbr not in seen:
                    yield self._report(n, nbr, dd)
            seen[n] = 1
        del seen

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and u not in self._nbunch and v not in self._nbunch:
            return False  # this edge doesn't start and it doesn't end in nbunch
        try:
            ddict = self._adjdict[u][v]
        except KeyError:
            return False
        return e == self._report(u, v, ddict)


class InEdgeDataView(OutEdgeDataView):
    """An EdgeDataView class for outward edges of DiGraph; See EdgeDataView"""

    __slots__ = ()

    def __iter__(self):
        return (
            self._report(nbr, n, dd)
            for n, nbrs in self._nodes_nbrs()
            for nbr, dd in nbrs.items()
        )

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and v not in self._nbunch:
            return False  # this edge doesn't end in nbunch
        try:
            ddict = self._adjdict[v][u]
        except KeyError:
            return False
        return e == self._report(u, v, ddict)


class OutMultiEdgeDataView(OutEdgeDataView):
    """An EdgeDataView for outward edges of MultiDiGraph; See EdgeDataView"""

    __slots__ = ("keys",)

    def __getstate__(self):
        return {
            "viewer": self._viewer,
            "nbunch": self._nbunch,
            "keys": self.keys,
            "data": self._data,
            "default": self._default,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    def __init__(self, viewer, nbunch=None, data=False, keys=False, default=None):
        self._viewer = viewer
        adjdict = self._adjdict = viewer._adjdict
        self.keys = keys
        if nbunch is None:
            self._nodes_nbrs = adjdict.items
        else:
            # dict retains order of nodes but acts like a set
            nbunch = dict.fromkeys(viewer._graph.nbunch_iter(nbunch))
            self._nodes_nbrs = lambda: [(n, adjdict[n]) for n in nbunch]
        self._nbunch = nbunch
        self._data = data
        self._default = default
        # Set _report based on data and default
        if data is True:
            if keys is True:
                self._report = lambda n, nbr, k, dd: (n, nbr, k, dd)
            else:
                self._report = lambda n, nbr, k, dd: (n, nbr, dd)
        elif data is False:
            if keys is True:
                self._report = lambda n, nbr, k, dd: (n, nbr, k)
            else:
                self._report = lambda n, nbr, k, dd: (n, nbr)
        else:  # data is attribute name
            if keys is True:
                self._report = (
                    lambda n, nbr, k, dd: (n, nbr, k, dd[data])
                    if data in dd
                    else (n, nbr, k, default)
                )
            else:
                self._report = (
                    lambda n, nbr, k, dd: (n, nbr, dd[data])
                    if data in dd
                    else (n, nbr, default)
                )

    def __len__(self):
        return sum(1 for e in self)

    def __iter__(self):
        return (
            self._report(n, nbr, k, dd)
            for n, nbrs in self._nodes_nbrs()
            for nbr, kd in nbrs.items()
            for k, dd in kd.items()
        )

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and u not in self._nbunch:
            return False  # this edge doesn't start in nbunch
        try:
            kdict = self._adjdict[u][v]
        except KeyError:
            return False
        if self.keys is True:
            k = e[2]
            try:
                dd = kdict[k]
            except KeyError:
                return False
            return e == self._report(u, v, k, dd)
        for k, dd in kdict.items():
            if e == self._report(u, v, k, dd):
                return True
        return False


class MultiEdgeDataView(OutMultiEdgeDataView):
    """An EdgeDataView class for edges of MultiGraph; See EdgeDataView"""

    __slots__ = ()

    def __iter__(self):
        seen = {}
        for n, nbrs in self._nodes_nbrs():
            for nbr, kd in nbrs.items():
                if nbr not in seen:
                    for k, dd in kd.items():
                        yield self._report(n, nbr, k, dd)
            seen[n] = 1
        del seen

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and u not in self._nbunch and v not in self._nbunch:
            return False  # this edge doesn't start and doesn't end in nbunch
        try:
            kdict = self._adjdict[u][v]
        except KeyError:
            try:
                kdict = self._adjdict[v][u]
            except KeyError:
                return False
        if self.keys is True:
            k = e[2]
            try:
                dd = kdict[k]
            except KeyError:
                return False
            return e == self._report(u, v, k, dd)
        for k, dd in kdict.items():
            if e == self._report(u, v, k, dd):
                return True
        return False


class InMultiEdgeDataView(OutMultiEdgeDataView):
    """An EdgeDataView for inward edges of MultiDiGraph; See EdgeDataView"""

    __slots__ = ()

    def __iter__(self):
        return (
            self._report(nbr, n, k, dd)
            for n, nbrs in self._nodes_nbrs()
            for nbr, kd in nbrs.items()
            for k, dd in kd.items()
        )

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and v not in self._nbunch:
            return False  # this edge doesn't end in nbunch
        try:
            kdict = self._adjdict[v][u]
        except KeyError:
            return False
        if self.keys is True:
            k = e[2]
            dd = kdict[k]
            return e == self._report(u, v, k, dd)
        for k, dd in kdict.items():
            if e == self._report(u, v, k, dd):
                return True
        return False


# EdgeViews    have set operations and no data reported
class OutEdgeView(Set, Mapping):
    """A EdgeView class for outward edges of a DiGraph"""

    __slots__ = ("_adjdict", "_graph", "_nodes_nbrs")

    def __getstate__(self):
        return {"_graph": self._graph}

    def __setstate__(self, state):
        self._graph = G = state["_graph"]
        self._adjdict = G._succ if hasattr(G, "succ") else G._adj
        self._nodes_nbrs = self._adjdict.items

    @classmethod
    def _from_iterable(cls, it):
        return set(it)

    dataview = OutEdgeDataView

    def __init__(self, G):
        self._graph = G
        self._adjdict = G._succ if hasattr(G, "succ") else G._adj
        self._nodes_nbrs = self._adjdict.items

    # Set methods
    def __len__(self):
        return sum(len(nbrs) for n, nbrs in self._nodes_nbrs())

    def __iter__(self):
        for n, nbrs in self._nodes_nbrs():
            for nbr in nbrs:
                yield (n, nbr)

    def __contains__(self, e):
        try:
            u, v = e
            return v in self._adjdict[u]
        except KeyError:
            return False

    # Mapping Methods
    def __getitem__(self, e):
        u, v = e
        return self._adjdict[u][v]

    # EdgeDataView methods
    def __call__(self, nbunch=None, data=False, default=None):
        if nbunch is None and data is False:
            return self
        return self.dataview(self, nbunch, data, default)

    def data(self, data=True, default=None, nbunch=None):
        """
        Return a read-only view of edge data.
        Parameters
        ----------
        data : bool or edge attribute key
            If ``data=True``, then the data view maps each edge to a dictionary
            containing all of its attributes. If `data` is a key in the edge
            dictionary, then the data view maps each edge to its value for
            the keyed attribute. In this case, if the edge doesn't have the
            attribute, the `default` value is returned.
        default : object, default=None
            The value used when an edge does not have a specific attribute
        nbunch : container of nodes, optional (default=None)
            Allows restriction to edges only involving certain nodes. All edges
            are considered by default.
        Returns
        -------
        dataview
            Returns an `EdgeDataView` for undirected Graphs, `OutEdgeDataView`
            for DiGraphs, `MultiEdgeDataView` for MultiGraphs and
            `OutMultiEdgeDataView` for MultiDiGraphs.
        Notes
        -----
        If ``data=False``, returns an `EdgeView` without any edge data.
        See Also
        --------
        EdgeDataView
        OutEdgeDataView
        MultiEdgeDataView
        OutMultiEdgeDataView
        Examples
        --------
        >>> G = nx.Graph()
        >>> G.add_edges_from([
        ...     (0, 1, {"dist": 3, "capacity": 20}),
        ...     (1, 2, {"dist": 4}),
        ...     (2, 0, {"dist": 5})
        ... ])
        Accessing edge data with ``data=True`` (the default) returns an
        edge data view object listing each edge with all of its attributes:
        >>> G.edges.data()
        EdgeDataView([(0, 1, {'dist': 3, 'capacity': 20}), (0, 2, {'dist': 5}), (1, 2, {'dist': 4})])
        If `data` represents a key in the edge attribute dict, a dataview listing
        each edge with its value for that specific key is returned:
        >>> G.edges.data("dist")
        EdgeDataView([(0, 1, 3), (0, 2, 5), (1, 2, 4)])
        `nbunch` can be used to limit the edges:
        >>> G.edges.data("dist", nbunch=[0])
        EdgeDataView([(0, 1, 3), (0, 2, 5)])
        If a specific key is not found in an edge attribute dict, the value
        specified by `default` is used:
        >>> G.edges.data("capacity")
        EdgeDataView([(0, 1, 20), (0, 2, None), (1, 2, None)])
        Note that there is no check that the `data` key is present in any of
        the edge attribute dictionaries:
        >>> G.edges.data("speed")
        EdgeDataView([(0, 1, None), (0, 2, None), (1, 2, None)])
        """
        if nbunch is None and data is False:
            return self
        return self.dataview(self, nbunch, data, default)

    # String Methods
    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)})"


class EdgeView(OutEdgeView):
    __slots__ = ()

    dataview = EdgeDataView

    def __len__(self):
        num_nbrs = (len(nbrs) + (n in nbrs) for n, nbrs in self._nodes_nbrs())
        return sum(num_nbrs) // 2

    def __iter__(self):
        seen = {}
        for n, nbrs in self._nodes_nbrs():
            for nbr in list(nbrs):
                if nbr not in seen:
                    yield (n, nbr)
            seen[n] = 1
        del seen

    def __contains__(self, e):
        try:
            u, v = e[:2]
            return v in self._adjdict[u] or u in self._adjdict[v]
        except (KeyError, ValueError):
            return False


class InEdgeView(OutEdgeView):
    """A EdgeView class for inward edges of a DiGraph"""

    __slots__ = ()

    def __setstate__(self, state):
        self._graph = G = state["_graph"]
        self._adjdict = G._pred if hasattr(G, "pred") else G._adj
        self._nodes_nbrs = self._adjdict.items

    dataview = InEdgeDataView

    def __init__(self, G):
        self._graph = G
        self._adjdict = G._pred if hasattr(G, "pred") else G._adj
        self._nodes_nbrs = self._adjdict.items

    def __iter__(self):
        for n, nbrs in self._nodes_nbrs():
            for nbr in nbrs:
                yield (nbr, n)

    def __contains__(self, e):
        try:
            u, v = e
            return u in self._adjdict[v]
        except KeyError:
            return False

    def __getitem__(self, e):
        u, v = e
        return self._adjdict[v][u]


class OutMultiEdgeView(OutEdgeView):
    """A EdgeView class for outward edges of a MultiDiGraph"""

    __slots__ = ()

    dataview = OutMultiEdgeDataView

    def __len__(self):
        return sum(
            len(kdict) for n, nbrs in self._nodes_nbrs() for nbr, kdict in nbrs.items()
        )

    def __iter__(self):
        for n, nbrs in self._nodes_nbrs():
            for nbr, kdict in nbrs.items():
                for key in kdict:
                    yield (n, nbr, key)

    def __contains__(self, e):
        N = len(e)
        if N == 3:
            u, v, k = e
        elif N == 2:
            u, v = e
            k = 0
        else:
            raise ValueError("MultiEdge must have length 2 or 3")
        try:
            return k in self._adjdict[u][v]
        except KeyError:
            return False

    def __getitem__(self, e):
        u, v, k = e
        return self._adjdict[u][v][k]

    def __call__(self, nbunch=None, data=False, keys=False, default=None):
        if nbunch is None and data is False and keys is True:
            return self
        return self.dataview(self, nbunch, data, keys, default)

    def data(self, data=True, keys=False, default=None, nbunch=None):
        if nbunch is None and data is False and keys is True:
            return self
        return self.dataview(self, nbunch, data, keys, default)


class MultiEdgeView(OutMultiEdgeView):
    """A EdgeView class for edges of a MultiGraph"""

    __slots__ = ()

    dataview = MultiEdgeDataView

    def __len__(self):
        return sum(1 for e in self)

    def __iter__(self):
        seen = {}
        for n, nbrs in self._nodes_nbrs():
            for nbr, kd in nbrs.items():
                if nbr not in seen:
                    for k, dd in kd.items():
                        yield (n, nbr, k)
            seen[n] = 1
        del seen


class InMultiEdgeView(OutMultiEdgeView):
    """A EdgeView class for inward edges of a MultiDiGraph"""

    __slots__ = ()

    def __setstate__(self, state):
        self._graph = G = state["_graph"]
        self._adjdict = G._pred if hasattr(G, "pred") else G._adj
        self._nodes_nbrs = self._adjdict.items

    dataview = InMultiEdgeDataView

    def __init__(self, G):
        self._graph = G
        self._adjdict = G._pred if hasattr(G, "pred") else G._adj
        self._nodes_nbrs = self._adjdict.items

    def __iter__(self):
        for n, nbrs in self._nodes_nbrs():
            for nbr, kdict in nbrs.items():
                for key in kdict:
                    yield (nbr, n, key)

    def __contains__(self, e):
        N = len(e)
        if N == 3:
            u, v, k = e
        elif N == 2:
            u, v = e
            k = 0
        else:
            raise ValueError("MultiEdge must have length 2 or 3")
        try:
            return k in self._adjdict[v][u]
        except KeyError:
            return False

    def __getitem__(self, e):
        u, v, k = e

class PlanarEmbedding(DiGraph):
    def get_data(self):
        """Converts the adjacency structure into a better readable structure.

        Returns
        -------
        embedding : dict
            A dict mapping all nodes to a list of neighbors sorted in
            clockwise order.

        See Also
        --------
        set_data

        """
        embedding = dict()
        for v in self:
            embedding[v] = list(self.neighbors_cw_order(v))
        return embedding


    def set_data(self, data):
        """Inserts edges according to given sorted neighbor list.

        The input format is the same as the output format of get_data().

        Parameters
        ----------
        data : dict
            A dict mapping all nodes to a list of neighbors sorted in
            clockwise order.

        See Also
        --------
        get_data

        """
        for v in data:
            for w in reversed(data[v]):
                self.add_half_edge_first(v, w)


    def neighbors_cw_order(self, v):
        """Generator for the neighbors of v in clockwise order.

        Parameters
        ----------
        v : node

        Yields
        ------
        node

        """
        if len(self[v]) == 0:
            # v has no neighbors
            return
        start_node = self.nodes[v]["first_nbr"]
        yield start_node
        current_node = self[v][start_node]["cw"]
        while start_node != current_node:
            yield current_node
            current_node = self[v][current_node]["cw"]


    def add_half_edge_ccw(self, start_node, end_node, reference_neighbor):
        """Adds a half-edge from start_node to end_node.

        The half-edge is added counter clockwise next to the existing half-edge
        (start_node, reference_neighbor).

        Parameters
        ----------
        start_node : node
            Start node of inserted edge.
        end_node : node
            End node of inserted edge.
        reference_neighbor: node
            End node of reference edge.

        Raises
        ------
        NetworkXException
            If the reference_neighbor does not exist.

        See Also
        --------
        add_half_edge_cw
        connect_components
        add_half_edge_first

        """
        if reference_neighbor is None:
            # The start node has no neighbors
            self.add_edge(start_node, end_node)  # Add edge to graph
            self[start_node][end_node]["cw"] = end_node
            self[start_node][end_node]["ccw"] = end_node
            self.nodes[start_node]["first_nbr"] = end_node
        else:
            ccw_reference = self[start_node][reference_neighbor]["ccw"]
            self.add_half_edge_cw(start_node, end_node, ccw_reference)

            if reference_neighbor == self.nodes[start_node].get("first_nbr", None):
                # Update first neighbor
                self.nodes[start_node]["first_nbr"] = end_node


    def add_half_edge_cw(self, start_node, end_node, reference_neighbor):
        """Adds a half-edge from start_node to end_node.

        The half-edge is added clockwise next to the existing half-edge
        (start_node, reference_neighbor).

        Parameters
        ----------
        start_node : node
            Start node of inserted edge.
        end_node : node
            End node of inserted edge.
        reference_neighbor: node
            End node of reference edge.

        Raises
        ------
        NetworkXException
            If the reference_neighbor does not exist.

        See Also
        --------
        add_half_edge_ccw
        connect_components
        add_half_edge_first
        """
        self.add_edge(start_node, end_node)  # Add edge to graph

        if reference_neighbor is None:
            # The start node has no neighbors
            self[start_node][end_node]["cw"] = end_node
            self[start_node][end_node]["ccw"] = end_node
            self.nodes[start_node]["first_nbr"] = end_node
            return

        # Get half-edge at the other side
        cw_reference = self[start_node][reference_neighbor]["cw"]
        # Alter half-edge data structures
        self[start_node][reference_neighbor]["cw"] = end_node
        self[start_node][end_node]["cw"] = cw_reference
        self[start_node][cw_reference]["ccw"] = end_node
        self[start_node][end_node]["ccw"] = reference_neighbor


    def connect_components(self, v, w):
        """Adds half-edges for (v, w) and (w, v) at some position.

        This method should only be called if v and w are in different
        components, or it might break the embedding.
        This especially means that if `connect_components(v, w)`
        is called it is not allowed to call `connect_components(w, v)`
        afterwards. The neighbor orientations in both directions are
        all set correctly after the first call.

        Parameters
        ----------
        v : node
        w : node

        See Also
        --------
        add_half_edge_ccw
        add_half_edge_cw
        add_half_edge_first
        """
        self.add_half_edge_first(v, w)
        self.add_half_edge_first(w, v)


    def add_half_edge_first(self, start_node, end_node):
        """The added half-edge is inserted at the first position in the order.

        Parameters
        ----------
        start_node : node
        end_node : node

        See Also
        --------
        add_half_edge_ccw
        add_half_edge_cw
        connect_components
        """
        if start_node in self and "first_nbr" in self.nodes[start_node]:
            reference = self.nodes[start_node]["first_nbr"]
        else:
            reference = None
        self.add_half_edge_ccw(start_node, end_node, reference)


    def next_face_half_edge(self, v, w):
        """Returns the following half-edge left of a face.

        Parameters
        ----------
        v : node
        w : node

        Returns
        -------
        half-edge : tuple
        """
        new_node = self[w][v]["ccw"]
        return w, new_node


    def traverse_face(self, v, w, mark_half_edges=None):
        """Returns nodes on the face that belong to the half-edge (v, w).

        The face that is traversed lies to the right of the half-edge (in an
        orientation where v is below w).

        Optionally it is possible to pass a set to which all encountered half
        edges are added. Before calling this method, this set must not include
        any half-edges that belong to the face.

        Parameters
        ----------
        v : node
            Start node of half-edge.
        w : node
            End node of half-edge.
        mark_half_edges: set, optional
            Set to which all encountered half-edges are added.

        Returns
        -------
        face : list
            A list of nodes that lie on this face.
        """
        if mark_half_edges is None:
            mark_half_edges = set()

        face_nodes = [v]
        mark_half_edges.add((v, w))
        prev_node = v
        cur_node = w
        # Last half-edge is (incoming_node, v)
        incoming_node = self[v][w]["cw"]

        while cur_node != v or prev_node != incoming_node:
            face_nodes.append(cur_node)
            prev_node, cur_node = self.next_face_half_edge(prev_node, cur_node)
        return face_nodes


    def is_directed(self):
        return False
    
for _ in range(int(input())):
    n = int(input())
    arr = [int(x)-1 for x in input().split()]
    G = Graph()
    for i in range(2*n-1):
        G.add_edge(i, i+1)
    G.add_edge(2*n-1, 0)
    pair = [[] for i in range(n)]
    for i, num in enumerate(arr):
        pair[num].append(i)
    for a, b in pair:
        G.add_edge(a, b)
    
    if check_planarity(G):
        print('yes')
    else:
        print('no')