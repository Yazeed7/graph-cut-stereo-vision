import networkx as nx

class MinCut(object):
    def __init__(self,G,s,t):
        """
        Determines the mincut partition between two terminals 's' and 't' using
        the Boykov Kolmogorov Algorithm cited in the report
        :G: Networkx DiGraph object with edge weights specified with a 'capacity' attribute
        :s: First terminal name
        :t: Second terminal name
        """

        self.s = s
        self.t = t
        self.S = {s}
        self.T = {t}
        self.A = {s,t}
        self.O = set()
        self.parent = {}
        self.G = G

        # Initialize flows as zero
        for u,v,a in G.edges(data=True):
            a['flow'] = 0
            a['backwards'] = False

    def compute(self):
        """Refer to Boykov Kolmogorov Algorithm cited in the report"""
        while True:
            P = self.grow()
            if P is None:
                break
            self.augment(P)
            self.adopt()
        return self.partition()

    def grow(self):
        """Refer to Boykov Kolmogorov Algorithm cited in the report"""
        while len(self.A) != 0:
            p = self.A.pop()
            self.A.add(p)
            qs = self.G.successors(p) if p in self.S else self.G.predecessors(p)
            for q in qs:
                e = self.G[p][q] if p in self.S else self.G[q][p]
                if e['capacity']-e['flow'] > 0:
                    if q not in self.S and q not in self.T:
                        if p in self.S:
                            self.S.add(q)
                        else:
                            self.T.add(q)
                        self.parent[q] = p
                        self.A.add(q)
                    else:
                        if (p in self.S and q in self.T) or (p in self.T and q in self.S):
                            return self.get_path(p,q)
            self.A.remove(p)
        return None

    def augment(self, P):
        """Refer to Boykov Kolmogorov Algorithm cited in the report"""
        edges = list(zip(P[:-1], P[1:]))
        bottleneck = min([self.G[p][q]['capacity']-self.G[p][q]['flow'] for p,q in edges])
        for p,q in edges:
            if not self.G[p][q]['backwards']:
                self.G[p][q]['flow'] += bottleneck
                # turn bottlenecks into orphans
                if self.G[p][q]['flow'] == self.G[p][q]['capacity']:
                    if (p in self.S) and (q in self.S):
                        self.parent.pop(q)
                        self.O.add(q)
                    elif (p in self.T) and (q in self.T):
                        self.parent.pop(p)
                        self.O.add(p)
                # add reverse edges
                if not self.G.has_edge(q,p):
                    self.G.add_edge(q,p)
                    self.G[q][p]['backwards'] = True
                    self.G[q][p]['capacity'] = 0
                self.G[q][p]['flow'] = -self.G[p][q]['flow']
            else:
                self.G[p][q]['flow'] += bottleneck
                if self.G[p][q]['flow']==0:
                    self.G.remove_edge(p,q)
                self.G[q][p]['flow'] -= bottleneck

    def adopt(self):
        """Refer to Boykov Kolmogorov Algorithm cited in the report"""
        while len(self.O) != 0:
            p = self.O.pop()
            qs = list(self.G.predecessors(p) if p in self.S else self.G.successors(p))
            found_parent = False
            for q in qs:
                same_tree_condition = (p in self.S and q in self.S) or (p in self.T and q in self.T)
                e = self.G[q][p] if p in self.S else self.G[p][q]
                capacity_condition = e['capacity']-e['flow']>0
                origin_condition = self.get_origin(q) in {self.s, self.t}
                if all([same_tree_condition,capacity_condition,origin_condition]):
                    self.parent[p] = q
                    found_parent = True
                    break
            if not found_parent:
#                 qs = list(self.G.predecessors(p))+list(self.G.predecessors(p))
                qs = list(self.G.successors(p) if p in self.S else self.G.predecessors(p))
                for q in qs:
                    if (q in self.S and p in self.S) or (q in self.T and p in self.T):
#                         e = self.G[q][p] if p in self.S else self.G[p][q]
                        e = self.G[p][q] if p in self.S else self.G[q][p]
                        if e['capacity']-e['flow']>0:
                            self.A.add(q)
                        try:
                            if self.parent[q]==p:
                                self.O.add(q)
                                self.parent.pop(q)
                        except KeyError:
                            pass
                if p in self.A: self.A.remove(p)
                if p in self.S: self.S.remove(p)
                elif p in self.T: self.T.remove(p)

    def get_path(self, p, q):
        """
        Given a set of connected points p and q. This function
        traces back to the s and t origin nodes and creates an s-t path
        """
        if p in self.T:
            p,q = q,p
        path_p = []
        path_q = []
        while p != self.s:
            path_p.append(p)
            p = self.parent[p]
        while q != self.t:
            path_q.append(q)
            q = self.parent[q]
        path_p.reverse()

        return [self.s]+path_p+path_q+[self.t]

    def get_origin(self, p):
        """
        Lord forgive me for the sin i'm about to commit.
        Returns the highest parent of a node p.
        """
        while True:
            try:
                p = self.parent[p]
            except KeyError:
                break
        return p

    def partition(self):
        for u,v,a in list(self.G.edges(data=True)):
            if a['flow']==a['capacity'] or a['backwards']:
                if self.G.has_edge(u,v):
                    self.G.remove_edge(u,v)
                if self.G.has_edge(v,u):
                    self.G.remove_edge(v,u)
        self.G = self.G.to_undirected()
        s_nodes = nx.node_connected_component(self.G, self.s)
        return s_nodes, set(self.G.nodes).difference(s_nodes)

# make parent dict
# mincut using iters like nx
