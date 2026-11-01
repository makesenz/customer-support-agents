"""(Re)load the Neo4j knowledge graph from knowledge_graph/graph_seed.yaml.

Run after editing the seed file:  python -m scripts.load_graph
"""
from neo4j import GraphDatabase

import config
from rag.graph import load_into_neo4j


def main() -> None:
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
                                  notifications_min_severity="OFF")
    driver.verify_connectivity()
    load_into_neo4j(driver)
    with driver.session() as s:
        for row in s.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label"):
            print(f"{row['label']:<14} {row['n']}")
        print("relationships:", s.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"])
    driver.close()


if __name__ == "__main__":
    main()
