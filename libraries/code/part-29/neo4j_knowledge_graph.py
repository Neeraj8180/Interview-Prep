# Neo4j knowledge graphs — Part 29 code examples
# Demonstrates graph concepts without Neo4j server
# For server-based examples: pip install neo4j

class PropertyGraph:
    """
    In-memory property graph — demonstrates Neo4j concepts.
    Replace with Neo4j GraphDatabase.driver() for production.
    """
    def __init__(self):
        self.nodes = {}   # id → {"labels": set, "props": dict}
        self.rels  = []   # {"from_id", "to_id", "type", "props"}
        self._id   = 0

    def merge_node(self, labels, **props):
        """Like MERGE in Cypher — find or create."""
        for nid, node in self.nodes.items():
            if set(labels) == node["labels"] and all(
                node["props"].get(k) == v for k, v in props.items()
            ):
                return nid
        nid = self._id; self._id += 1
        self.nodes[nid] = {"labels": set(labels), "props": props}
        return nid

    def create_rel(self, from_id, to_id, rel_type, **props):
        self.rels.append({"from_id": from_id, "to_id": to_id,
                          "type": rel_type, "props": props})

    def match_pattern(self, rel_type=None, from_labels=None, to_labels=None):
        """MATCH (a)-[:type]->(b)"""
        results = []
        for r in self.rels:
            if rel_type and r["type"] != rel_type: continue
            fn = self.nodes.get(r["from_id"])
            tn = self.nodes.get(r["to_id"])
            if fn is None or tn is None: continue
            if from_labels and not set(from_labels).issubset(fn["labels"]): continue
            if to_labels   and not set(to_labels).issubset(tn["labels"]):   continue
            results.append((fn["props"], r["type"], tn["props"], r["props"]))
        return results

    def traverse(self, start_id, rel_type, hops=1):
        """Multi-hop traversal: start_id → (hop 1) → (hop 2) → ..."""
        current = {start_id}
        for _ in range(hops):
            next_level = set()
            for r in self.rels:
                if r["from_id"] in current and r["type"] == rel_type:
                    next_level.add(r["to_id"])
            current = next_level
        return [self.nodes[nid]["props"] for nid in current if nid in self.nodes]

    def shortest_path(self, from_id, to_id, rel_type=None):
        """BFS shortest path."""
        from collections import deque
        queue   = deque([(from_id, [from_id])])
        visited = {from_id}
        while queue:
            cur, path = queue.popleft()
            if cur == to_id:
                return [self.nodes[nid]["props"] for nid in path]
            for r in self.rels:
                if r["from_id"] == cur:
                    nxt = r["to_id"]
                    if nxt not in visited:
                        if rel_type is None or r["type"] == rel_type:
                            visited.add(nxt)
                            queue.append((nxt, path + [nxt]))
        return None


def demo_knowledge_graph():
    """Build and query a research knowledge graph."""
    print("=" * 60)
    print("Knowledge Graph: Research Network")
    print("=" * 60)

    g = PropertyGraph()

    # Create researchers
    alice = g.merge_node(["Person"], name="Alice",  org="Google")
    bob   = g.merge_node(["Person"], name="Bob",    org="Meta")
    carol = g.merge_node(["Person"], name="Carol",  org="OpenAI")
    dave  = g.merge_node(["Person"], name="Dave",   org="DeepMind")

    # Create papers
    attention = g.merge_node(["Paper"], title="Attention Is All You Need", year=2017)
    bert      = g.merge_node(["Paper"], title="BERT",                      year=2019)
    gpt3      = g.merge_node(["Paper"], title="GPT-3",                     year=2020)
    llama     = g.merge_node(["Paper"], title="LLaMA",                     year=2023)

    # Create institutions
    google  = g.merge_node(["Institution"], name="Google")
    meta    = g.merge_node(["Institution"], name="Meta")

    # Authorship
    g.create_rel(alice, attention, "AUTHORED")
    g.create_rel(bob,   attention, "AUTHORED")
    g.create_rel(carol, bert,      "AUTHORED")
    g.create_rel(carol, gpt3,      "AUTHORED")
    g.create_rel(dave,  llama,     "AUTHORED")

    # Citations
    g.create_rel(bert,  attention, "CITES")
    g.create_rel(gpt3,  bert,      "CITES")
    g.create_rel(llama, gpt3,      "CITES")

    # Professional network
    g.create_rel(alice, bob,   "KNOWS")
    g.create_rel(bob,   carol, "KNOWS")
    g.create_rel(carol, dave,  "KNOWS")

    # Affiliations
    g.create_rel(alice, google, "WORKS_AT")
    g.create_rel(bob,   meta,   "WORKS_AT")

    print("\n1. Who authored each paper? (MATCH (p:Person)-[:AUTHORED]->(paper:Paper))")
    for author, rel, paper, _ in g.match_pattern("AUTHORED", ["Person"], ["Paper"]):
        print(f"   {author['name']:8} → '{paper['title']}'")

    print("\n2. Citation chain (MATCH (a:Paper)-[:CITES]->(b:Paper))")
    for paper, rel, cited, _ in g.match_pattern("CITES", ["Paper"], ["Paper"]):
        print(f"   '{paper['title']}' → '{cited['title']}'")

    print("\n3. 2-hop KNOWS from Alice (friends-of-friends)")
    fof = g.traverse(alice, "KNOWS", hops=2)
    print(f"   Alice's friends-of-friends: {[p['name'] for p in fof]}")

    print("\n4. Shortest path: Alice → Dave")
    path = g.shortest_path(alice, dave, rel_type="KNOWS")
    if path:
        print(f"   Path: {' → '.join(p['name'] for p in path)}")

    print("\n5. Equivalent Cypher queries:")
    print("   MATCH (p:Person)-[:AUTHORED]->(paper:Paper) RETURN p.name, paper.title")
    print("   MATCH (a:Paper)-[:CITES]->(b:Paper) RETURN a.title, b.title")
    print("   MATCH (alice:Person {name:'Alice'})-[:KNOWS*2]->(fof) RETURN fof.name")
    print("   MATCH path=shortestPath((alice)-[:KNOWS*]-(dave)) RETURN path")


def demo_neo4j_driver():
    """Show Neo4j driver API (requires running Neo4j)."""
    print("\n" + "=" * 60)
    print("Neo4j Driver API (requires: pip install neo4j)")
    print("=" * 60)

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        with driver.session() as session:
            count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            print(f"  Connected! Total nodes: {count}")
        driver.close()

    except ImportError:
        print("  neo4j not installed: pip install neo4j")
    except Exception as e:
        print(f"  Neo4j not running: {e}")

    print("""
  Neo4j driver pattern:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "pwd"))

    # Write
    def create_person(tx, name):
        tx.run("MERGE (:Person {name: $n})", n=name)

    with driver.session() as s:
        s.execute_write(create_person, "Alice")
        rows = s.execute_read(lambda tx: list(
            tx.run("MATCH (p:Person)-[:AUTHORED]->(paper:Paper) RETURN p.name, paper.title")
        ))
    driver.close()

  LangChain integration:
    from langchain_community.graphs import Neo4jGraph
    graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="pwd")
    graph.query("MATCH (n:Person) RETURN count(n)")
""")


if __name__ == "__main__":
    demo_knowledge_graph()
    demo_neo4j_driver()
