# HYBRID LANGGRAPH + LANGCHAIN PIPELINE (per-source retrievers + weighted fusion)
# ------------------------------------------------------------------------------
# - Separate vector stores for: FSD, T5, Mapping, Regression
# - Weighted retrieval for generation (regression weight=0) and review (regression > 0)
# - Generator grounded only on FSD/T5/Mapping; Reviewer can see regression as reference
# - Same output format as your earlier flow
# ------------------------------------------------------------------------------

import os, uuid, json, math, re
from typing import List, Dict, Tuple, TypedDict

import pandas as pd
from docx import Document as DocxDocument

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document

from langgraph.graph import StateGraph, END


# =========================
# CONFIG
# =========================
# Set one of these (OpenAI or Azure OpenAI). LangChain respects standard env vars.
# export OPENAI_API_KEY="sk-..."
# # For Azure:
# export AZURE_OPENAI_API_KEY="..."
# export AZURE_OPENAI_ENDPOINT="https://<your>.openai.azure.com/"
# export AZURE_OPENAI_API_VERSION="2024-02-15-preview"

EMBED_MODEL = "text-embedding-3-large"  # Azure: use your embedding deployment name via env if needed
CHAT_MODEL_GEN = "gpt-4o-mini"          # Azure: set model="azure/<deployment-name>"
CHAT_MODEL_REV = "gpt-4o-mini"

DB_DIR = "./vectordbs_hybrid"
os.makedirs(DB_DIR, exist_ok=True)

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

EMB = OpenAIEmbeddings(model=EMBED_MODEL)
llm_gen = ChatOpenAI(model=CHAT_MODEL_GEN, temperature=0)
llm_rev = ChatOpenAI(model=CHAT_MODEL_REV, temperature=0)


# =========================
# LOADERS → TEXT
# =========================

def load_docx_text(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def excel_to_summaries(path: str, source_tag: str) -> List[Dict]:
    """Turn each sheet into a concise text summary (schema + quick stats)."""
    xls = pd.ExcelFile(path)
    items = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        schema_lines = [f"- {c}: {str(df[c].dtype)}" for c in df.columns]
        # Small stats (avoid heavy ops)
        stats_lines = []
        for c in df.columns:
            s = df[c]
            stats_lines.append(f"- {c}: n={s.notna().sum()}, nulls={s.isna().sum()}")
        txt = (
            f"Source: {source_tag}\nSheet: {sheet}\n\n"
            f"Columns:\n" + "\n".join(schema_lines) + "\n\n"
            f"Quick Stats:\n" + "\n".join(stats_lines)
        )
        items.append({"text": txt, "metadata": {"source": source_tag, "sheet": sheet}})
    return items

def excel_to_records_preview(path: str, max_rows: int = 50) -> str:
    """If you want to show small record previews to the LLM."""
    xls = pd.ExcelFile(path)
    first = xls.sheet_names[0]
    df = xls.parse(first)
    return df.head(max_rows).to_json(orient="records")


# =========================
# BUILD SEPARATE VECTOR STORES
# =========================

def chunk_and_index(texts_with_meta: List[Dict], persist_dir: str, collection_name: str) -> Chroma:
    texts, metas = [], []
    for item in texts_with_meta:
        for chunk in SPLITTER.split_text(item["text"]):
            texts.append(chunk)
            md = dict(item["metadata"])
            md["chunk_id"] = str(uuid.uuid4())
            metas.append(md)
    db = Chroma.from_texts(
        texts=texts,
        embedding=EMB,
        metadatas=metas,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    db.persist()
    return db

def build_all_stores(
    fsd_docx_path: str,
    t5_excel_path: str,
    mapping_excel_path: str,
    regression_excel_path: str,
):
    # FSD
    fsd_text = load_docx_text(fsd_docx_path)
    fsd_items = [{"text": fsd_text, "metadata": {"source": "fsd", "path": fsd_docx_path}}]
    fsd_db = chunk_and_index(fsd_items, f"{DB_DIR}/fsd", "fsd")

    # T5 (EFRA)
    t5_items = excel_to_summaries(t5_excel_path, "t5")
    t5_db = chunk_and_index(t5_items, f"{DB_DIR}/t5", "t5")

    # Disclosure Mapping
    mapping_items = excel_to_summaries(mapping_excel_path, "mapping")
    mapping_db = chunk_and_index(mapping_items, f"{DB_DIR}/mapping", "mapping")

    # Regression (as reference only)
    reg_items = excel_to_summaries(regression_excel_path, "regression")
    regression_db = chunk_and_index(reg_items, f"{DB_DIR}/regression", "regression")

    return fsd_db, t5_db, mapping_db, regression_db


# =========================
# WEIGHTED RETRIEVAL (RRF)
# =========================

def retrieve_topk(db: Chroma, query: str, k: int = 6):
    return db.similarity_search_with_score(query, k=k)

def weighted_rrf(results_by_source: Dict[str, List[Tuple]], weights: Dict[str, float], k_fuse: int = 12):
    """
    results_by_source: {"fsd":[(doc,score),...], "t5":[...], ...}
    weights: e.g. {"fsd":0.5,"t5":0.3,"mapping":0.2,"regression":0.0}
    """
    ranks: Dict[str, Dict] = {}
    for src, results in results_by_source.items():
        for rank, (doc, _score) in enumerate(results, start=1):
            key = doc.metadata.get("chunk_id", id(doc))
            contrib = weights.get(src, 0.0) * (1.0 / (60 + rank))  # standard RRF with base 60
            if key not in ranks:
                ranks[key] = {"doc": doc, "score": 0.0}
            ranks[key]["score"] += contrib
    fused = sorted(ranks.values(), key=lambda x: x["score"], reverse=True)
    return [(x["doc"], x["score"]) for x in fused[:k_fuse]]

GEN_WEIGHTS = {"fsd": 0.5, "t5": 0.3, "mapping": 0.2, "regression": 0.0}
REVIEW_WEIGHTS = {"fsd": 0.4, "t5": 0.2, "mapping": 0.2, "regression": 0.2}

def build_context(query: str, stores, for_review: bool = False) -> str:
    fsd_db, t5_db, mapping_db, reg_db = stores
    results = {
        "fsd": retrieve_topk(fsd_db, query, k=6),
        "t5": retrieve_topk(t5_db, query, k=6),
        "mapping": retrieve_topk(mapping_db, query, k=6),
        "regression": retrieve_topk(reg_db, query, k=6),
    }
    weights = REVIEW_WEIGHTS if for_review else GEN_WEIGHTS
    fused = weighted_rrf(results, weights, k_fuse=12)

    # context string with chunk tags for citations
    ctx_lines = []
    for i, (doc, _sc) in enumerate(fused, start=1):
        md = doc.metadata
        ref = f"{md.get('source')}|{md.get('sheet', md.get('path',''))}|{md.get('chunk_id','')}"
        ctx_lines.append(f"<<<CTX-{i} [{ref}]>>>\n{doc.page_content}\n<<<END-CTX-{i}>>>")
    return "\n".join(ctx_lines)

def build_multi_aspect_context(aspects: List[str], stores, for_review: bool = False) -> str:
    # Fuse contexts from multiple sub-queries to improve coverage
    ctx_parts = []
    for a in aspects:
        ctx = build_context(a, stores, for_review=for_review)
        if ctx.strip():
            ctx_parts.append(f"### QUERY: {a}\n{ctx}")
    return "\n\n".join(ctx_parts)


# =========================
# GRAPH STATE
# =========================

class TestGenState(TypedDict):
    # Inputs
    user_query: str
    stores: tuple  # (fsd_db, t5_db, mapping_db, reg_db)
    # Intermediate contexts
    gen_context: str
    review_context: str
    # Outputs
    test_cases: str
    reviewed_cases: str


# =========================
# NODES
# =========================

def node_retrieve_generation(state: TestGenState) -> TestGenState:
    """Build generation context from FSD/T5/Mapping only (weighted)."""
    aspects = [
        state["user_query"],
        f"{state['user_query']} KPIs filters aggregations",
        f"{state['user_query']} data types column constraints",
        f"{state['user_query']} disclosure mapping joins drilldown",
        f"{state['user_query']} edge cases nulls RBAC date ranges performance"
    ]
    ctx = build_multi_aspect_context(aspects, state["stores"], for_review=False)
    return {**state, "gen_context": ctx}

GEN_PROMPT = """You are a QA test designer for Tableau disclosures.

PRIORITY OF SOURCES (strict):
1) FSD (functional spec) – authoritative facts
2) T5/EFRA extract – data shape, types, constraints
3) Disclosure mapping – field-to-report linkage
Regression examples are NOT available to you at this step.

Task:
- Generate MANUAL test cases grounded ONLY in the provided context.
- Output strict JSON with this schema:
{{
  "test_cases": [
    {{
      "id": "TC-<auto-number>",
      "title": "...",
      "area": "KPI|Filter|Aggregation|UI|Export|Security|Performance|DataQuality|Join",
      "preconditions": ["..."],
      "steps": ["step 1", "step 2", "..."],
      "test_data": ["..."],
      "expected_results": ["..."],
      "traceability": [
         {{"source": "<fsd|t5|mapping>", "ref": "<chunk_id/sheet/page>"}}
      ]
    }}
  ]
}}

Rules:
- Prefer exact field names, KPIs, filters from the context.
- Include negative & boundary cases (nulls, date ranges, RBAC).
- For each expected result, include a citation to a context tag (<<<CTX-i ...>>>).
- Do not invent report names that are not present in the context.
Context:
{context}

User request:
{query}
"""

def node_generate(state: TestGenState) -> TestGenState:
    prompt = GEN_PROMPT.format(context=state["gen_context"], query=state["user_query"])
    out = llm_gen.invoke(prompt)
    return {**state, "test_cases": out.content}

def node_retrieve_review(state: TestGenState) -> TestGenState:
    """Build review context incl. regression (lower weight than FSD/T5/Mapping)."""
    aspects = [
        state["user_query"],
        f"{state['user_query']} missing coverage risky scenarios",
        f"{state['user_query']} navigation drilldown filter carry-forward",
        f"{state['user_query']} SQL validation join rules"
    ]
    ctx = build_multi_aspect_context(aspects, state["stores"], for_review=True)
    return {**state, "review_context": ctx}

REVIEW_PROMPT = """You are a Senior QA reviewer.

You will receive:
1) Draft test cases (JSON).
2) Context fused from FSD/T5/Mapping/Regression.

Instructions:
- Ensure every assertion is grounded in FSD/T5/Mapping. If a claim conflicts with them, mark an issue.
- Use regression context ONLY to suggest missing categories or risky gaps; do NOT override FSD/T5/Mapping facts.
- Enforce: coverage of KPIs, filters, aggregations, edge cases, RBAC, joins, drilldowns, exports, performance.
- Output JSON:
{{
  "reviewed_test_cases": [... same schema as input but improved ...],
  "issues": ["..."],
  "ok": true|false
}}

Context:
{context}

Draft JSON:
{draft}
"""

def node_review(state: TestGenState) -> TestGenState:
    prompt = REVIEW_PROMPT.format(context=state["review_context"], draft=state["test_cases"])
    out = llm_rev.invoke(prompt)
    return {**state, "reviewed_cases": out.content}


# =========================
# GRAPH
# =========================

builder = StateGraph(TestGenState)
builder.add_node("retrieve_generation", node_retrieve_generation)
builder.add_node("generate", node_generate)
builder.add_node("retrieve_review", node_retrieve_review)
builder.add_node("review", node_review)

builder.set_entry_point("retrieve_generation")
builder.add_edge("retrieve_generation", "generate")
builder.add_edge("generate", "retrieve_review")
builder.add_edge("retrieve_review", "review")
builder.add_edge("review", END)

graph = builder.compile()


# =========================
# UTIL: CSV/Excel export
# =========================

def json_tests_to_dataframe(json_text: str) -> pd.DataFrame:
    """Accepts either generator JSON or reviewer JSON; extracts test cases list."""
    try:
        obj = json.loads(json_text)
    except Exception:
        # Try to extract JSON block
        m = re.search(r"\{[\s\S]*\}", json_text)
        obj = json.loads(m.group(0)) if m else {"test_cases": []}

    test_list = []
    if "test_cases" in obj:
        test_list = obj["test_cases"]
    elif "reviewed_test_cases" in obj:
        test_list = obj["reviewed_test_cases"]

    rows = []
    for tc in test_list:
        rows.append({
            "id": tc.get("id", ""),
            "area": tc.get("area", ""),
            "title": tc.get("title", ""),
            "preconditions": " | ".join(tc.get("preconditions", [])),
            "steps": " | ".join(tc.get("steps", [])),
            "test_data": " | ".join(tc.get("test_data", [])),
            "expected_results": " | ".join(tc.get("expected_results", [])),
            "traceability": " | ".join([f"{t.get('source','')}:{t.get('ref','')}" for t in tc.get("traceability", [])]),
        })
    return pd.DataFrame(rows)

def save_tests_to_csv(json_text: str, out_path: str = "./generated_test_cases.csv"):
    df = json_tests_to_dataframe(json_text)
    df.to_csv(out_path, index=False)
    return out_path


# =========================
# MAIN (example)
# =========================
if __name__ == "__main__":
    # --- Paths (replace with your actual files) ---
    fsd_docx_path = "fsd_docs/EFRA_Reporting_Layer_FSD.docx"
    t5_excel_path = "t5_extract/EFRA_T5_Table.xlsx"
    mapping_excel_path = "disclosure/derivative_screen.xlsx"
    regression_excel_path = "regression/Regression_test_cases.xlsx"

    # Build (or reuse) the 4 vector stores
    fsd_db, t5_db, mapping_db, reg_db = build_all_stores(
        fsd_docx_path, t5_excel_path, mapping_excel_path, regression_excel_path
    )

    # User request / scope
    user_query = "Generate comprehensive manual test cases for the new <Disclosure Name> Tableau report."

    # Run the graph
    state: TestGenState = {
        "user_query": user_query,
        "stores": (fsd_db, t5_db, mapping_db, reg_db),
        "gen_context": "",
        "review_context": "",
        "test_cases": "",
        "reviewed_cases": "",
    }

    final = graph.invoke(state)

    print("\n=== GENERATOR OUTPUT ===\n")
    print(final["test_cases"][:2000])  # preview

    print("\n=== REVIEWER OUTPUT ===\n")
    print(final["reviewed_cases"][:2000])  # preview

    # Save to CSV (from reviewer output if available, else generator)
    output_json = final["reviewed_cases"] or final["test_cases"]
    csv_path = save_tests_to_csv(output_json, "./generated_test_cases_hybrid.csv")
    print(f"\nCSV written to: {csv_path}")
