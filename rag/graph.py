"""Knowledge graph (Neo4j): products, packages, concerns, payment methods, policies.

Vector and keyword search find passages; the graph answers *relationship*
questions exactly ("which pack lasts 3 months?", "what's free with 8 tins?",
"what helps with PCOD?"). If Neo4j is unreachable we fall back to the same
seed file in memory, so the app keeps working.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import yaml

import config

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_seed() -> dict:
    return yaml.safe_load(config.GRAPH_SEED.read_text(encoding="utf-8"))


def _inr(n: int | float) -> str:
    return f"₹{int(n):,}"


# --------------------------------------------------------------------------- loading
def _props(item: dict, drop: tuple[str, ...] = ()) -> dict:
    """Neo4j properties: lists must hold one type, so list values are stored as strings."""
    return {k: [str(x) for x in v] if isinstance(v, list) else v for k, v in item.items() if k not in drop}


def load_into_neo4j(driver) -> None:
    """Idempotently (re)create the graph from the seed file."""
    s = load_seed()
    with driver.session() as session:
        session.run("MATCH (n) WHERE n:Brand OR n:Product OR n:Package OR n:Concern OR n:Ingredient "
                    "OR n:PaymentMethod OR n:Policy OR n:Document DETACH DELETE n")
        b = s["brand"]
        session.run("MERGE (b:Brand {id:$id}) SET b += $props", id=b["id"], props=_props(b))
        for p in s["products"]:
            props = _props(p, drop=("ingredients", "concerns", "pairs_with"))
            session.run(
                "MATCH (b:Brand {id:$brand}) MERGE (p:Product {id:$id}) SET p += $props "
                "MERGE (b)-[:SELLS]->(p) MERGE (d:Document {name:$doc}) MERGE (p)-[:DOCUMENTED_IN]->(d)",
                brand=b["id"], id=p["id"], props=props, doc=p["doc"],
            )
            for ing in p["ingredients"]:
                session.run("MATCH (p:Product {id:$id}) MERGE (i:Ingredient {name:$ing}) MERGE (p)-[:CONTAINS]->(i)",
                            id=p["id"], ing=ing)
        for c in s["concerns"]:
            session.run("MERGE (c:Concern {id:$id}) SET c += $props", id=c["id"], props=_props(c))
        for p in s["products"]:
            for cid in p["concerns"]:
                session.run("MATCH (p:Product {id:$p}), (c:Concern {id:$c}) MERGE (p)-[:SUPPORTS]->(c)", p=p["id"], c=cid)
            if p.get("pairs_with"):
                session.run("MATCH (a:Product {id:$a}), (b:Product {id:$b}) MERGE (a)-[:PAIRS_WITH]->(b)",
                            a=p["id"], b=p["pairs_with"])
        for pk in s["packages"]:
            props = _props(pk, drop=("product",))
            session.run(
                "MATCH (p:Product {id:$product}) MERGE (k:Package {id:$id}) SET k += $props "
                "MERGE (p)-[:SOLD_AS]->(k) MERGE (d:Document {name:'pricing_and_offers.md'}) MERGE (k)-[:DOCUMENTED_IN]->(d)",
                product=pk["product"], id=pk["id"], props=props,
            )
            if pk.get("free_oils"):
                session.run("MATCH (k:Package {id:$id}), (o:Product {id:'herbal_oil'}) MERGE (k)-[:INCLUDES_FREE {qty:$q}]->(o)",
                            id=pk["id"], q=pk["free_oils"])
        for m in s["payment_methods"]:
            session.run("MATCH (b:Brand {id:$brand}) MERGE (m:PaymentMethod {id:$id}) SET m += $props "
                        "MERGE (b)-[:ACCEPTS]->(m) MERGE (d:Document {name:'payment_and_cod.md'}) MERGE (m)-[:DOCUMENTED_IN]->(d)",
                        brand=b["id"], id=m["id"], props=_props(m))
        for pol in s["policies"]:
            session.run("MATCH (b:Brand {id:$brand}) MERGE (x:Policy {id:$id}) SET x += $props "
                        "MERGE (b)-[:HAS_POLICY]->(x) MERGE (d:Document {name:$doc}) MERGE (x)-[:DOCUMENTED_IN]->(d)",
                        brand=b["id"], id=pol["id"], props=_props(pol), doc=pol["doc"])


# --------------------------------------------------------------------------- entity linking
def link_entities(text: str) -> dict[str, list[str]]:
    """Return seed ids mentioned in the text, grouped by node type."""
    s = load_seed()
    q = f" {re.sub(r'[-_/]', ' ', text.lower())} "  # "3-month" -> "3 month"
    found: dict[str, list[str]] = {}

    def hit(aliases: list[str]) -> bool:
        return any(re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", q) for a in aliases)

    for group in ("products", "packages", "concerns", "payment_methods", "policies"):
        for item in s[group]:
            aliases = [str(a).lower() for a in item.get("aliases", [])] + [item["name"].lower()]
            if hit(aliases):
                found.setdefault(group, []).append(item["id"])
    return found


# --------------------------------------------------------------------------- fact rendering
def _package_fact(pk: dict) -> str:
    saving = pk["mrp"] - pk["price"]
    save_txt = f", you save {_inr(saving)}" if saving > 0 else ""
    dur = f", lasts {pk['duration']}" if pk.get("duration") else ""
    return (f"{pk['name']} ({pk['product_name']}): {pk['contents']} — offer price {_inr(pk['price'])} "
            f"(MRP {_inr(pk['mrp'])}{save_txt}){dur}.")


def render_facts(concern_links: list[tuple[str, str]], products: list[dict], pairs: list[tuple[str, str]],
                 packages: list[dict], payments: list[dict], policies: list[dict]) -> list[str]:
    facts = [f"{prod} is designed to support {concern}." for prod, concern in concern_links]
    for p in products:
        facts.append(f"{p['name']}: {p['form']}, for {p['for']}. How to use: {p['dosage']}.")
    facts += [f"{a} pairs with {b} for combined use." for a, b in pairs]
    facts += [_package_fact(pk) for pk in packages]
    facts += [f"Payment – {m['name']}: {m['details']}. Rule: {m['rule']}" for m in payments]
    facts += [f"Policy – {x['name']}: {x['fact']} (see {x['doc']})" for x in policies]
    return list(dict.fromkeys(facts))


def facts_from_seed(entities: dict[str, list[str]]) -> list[str]:
    """In-memory fallback: same traversal as the Cypher queries, over the seed file."""
    s = load_seed()
    products = {p["id"]: p for p in s["products"]}
    concerns = {c["id"]: c for c in s["concerns"]}
    prod_ids = list(entities.get("products", []))
    concern_links = []
    for cid in entities.get("concerns", []):
        for p in s["products"]:
            if cid in p["concerns"]:
                concern_links.append((p["name"], concerns[cid]["name"]))
                prod_ids.append(p["id"])
    prod_ids = list(dict.fromkeys(prod_ids))
    pkg_ids = entities.get("packages") or [pk["id"] for pk in s["packages"] if pk["product"] in prod_ids]
    packages = [dict(pk, product_name=products[pk["product"]]["name"]) for pk in s["packages"] if pk["id"] in pkg_ids]
    pairs = [(products[i]["name"], products[products[i]["pairs_with"]]["name"])
             for i in prod_ids if products[i].get("pairs_with")]
    return render_facts(
        concern_links,
        [products[i] for i in prod_ids],
        pairs,
        packages,
        [m for m in s["payment_methods"] if m["id"] in entities.get("payment_methods", [])],
        [x for x in s["policies"] if x["id"] in entities.get("policies", [])],
    )


class KnowledgeGraph:
    def __init__(self):
        self.driver = None
        self.backend = "seed-file"
        if config.GRAPH_ENABLED and config.NEO4J_PASSWORD:
            try:
                from neo4j import GraphDatabase

                driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
                                              notifications_min_severity="OFF")
                driver.verify_connectivity()
                with driver.session() as session:
                    empty = session.run("MATCH (p:Product) RETURN count(p) AS n").single()["n"] == 0
                if empty:
                    load_into_neo4j(driver)
                self.driver = driver
                self.backend = "neo4j"
            except Exception as exc:  # noqa: BLE001 - any connection problem → fallback
                log.warning("Neo4j unavailable (%s); using seed-file graph.", exc)

    def query(self, text: str) -> list[str]:
        entities = link_entities(text)
        if not entities:
            return []
        if self.driver is None:
            return facts_from_seed(entities)
        try:
            return self._query_neo4j(entities)
        except Exception as exc:  # noqa: BLE001
            log.warning("Neo4j query failed (%s); using seed-file graph.", exc)
            return facts_from_seed(entities)

    def _query_neo4j(self, e: dict[str, list[str]]) -> list[str]:
        """One-hop traversal in Neo4j from the linked nodes."""
        with self.driver.session() as s:
            concern_links = [
                (r["p"], r["c"]) for r in s.run(
                    "MATCH (p:Product)-[:SUPPORTS]->(c:Concern) WHERE c.id IN $ids RETURN p.name AS p, c.name AS c",
                    ids=e.get("concerns", []))
            ]
            products = [dict(r["p"]) for r in s.run(
                "MATCH (p:Product) WHERE p.id IN $pids "
                "OR EXISTS { MATCH (p)-[:SUPPORTS]->(c:Concern) WHERE c.id IN $cids } RETURN p",
                pids=e.get("products", []), cids=e.get("concerns", []))]
            prod_ids = [p["id"] for p in products]
            pairs = [(r["a"], r["b"]) for r in s.run(
                "MATCH (a:Product)-[:PAIRS_WITH]->(b:Product) WHERE a.id IN $ids RETURN a.name AS a, b.name AS b",
                ids=prod_ids)]
            if e.get("packages"):
                pkg_rows = s.run("MATCH (p:Product)-[:SOLD_AS]->(k:Package) WHERE k.id IN $ids "
                                 "RETURN k, p.name AS product_name ORDER BY k.price", ids=e["packages"])
            else:
                pkg_rows = s.run("MATCH (p:Product)-[:SOLD_AS]->(k:Package) WHERE p.id IN $ids "
                                 "RETURN k, p.name AS product_name ORDER BY k.price", ids=prod_ids)
            packages = [dict(r["k"], product_name=r["product_name"]) for r in pkg_rows]
            payments = [dict(r["m"]) for r in s.run(
                "MATCH (m:PaymentMethod) WHERE m.id IN $ids RETURN m", ids=e.get("payment_methods", []))]
            policies = [dict(r["x"]) for r in s.run(
                "MATCH (x:Policy) WHERE x.id IN $ids RETURN x", ids=e.get("policies", []))]
        return render_facts(concern_links, products, pairs, packages, payments, policies)

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
