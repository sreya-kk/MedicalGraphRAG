"""
Streamlit UI for the Medical GraphRAG Q&A system.

Layout:
  Left (main): Chat interface with inline citations
  Right (sidebar): Graph stats + expandable citation cards
"""

import os
import sys
from pathlib import Path

# Allow running as `streamlit run src/app.py` from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.graph.neo4j_client import Neo4jClient
from src.qa.answerer import Answerer
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import VectorStore

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Medical GraphRAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Session state init
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []  # sources from last answer


# ------------------------------------------------------------------
# Resource initialization (cached)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to Neo4j...")
def get_neo4j_client() -> Neo4jClient:
    client = Neo4jClient()
    client.connect()
    return client


@st.cache_resource(show_spinner="Loading vector store...")
def get_vector_store() -> VectorStore:
    store = VectorStore()
    store.connect()
    return store


@st.cache_resource(show_spinner="Initializing retriever...")
def get_retriever() -> HybridRetriever:
    import anthropic as _anthropic

    neo4j = get_neo4j_client()
    vstore = get_vector_store()
    anth_client = _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    retriever = HybridRetriever(
        vector_store=vstore,
        neo4j_client=neo4j,
        anthropic_client=anth_client,
    )
    return retriever


@st.cache_resource(show_spinner="Loading answerer...")
def get_answerer() -> Answerer:
    import anthropic as _anthropic

    anth_client = _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return Answerer(anthropic_client=anth_client)


# ------------------------------------------------------------------
# Sidebar — Graph stats + citation panel
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🏥 Medical GraphRAG")
    st.caption("CMS Medicare Policy Knowledge Graph")

    st.divider()
    st.subheader("Graph Stats")

    try:
        neo4j = get_neo4j_client()
        stats = neo4j.get_stats()
        col1, col2 = st.columns(2)
        col1.metric("Nodes", f"{stats['nodes']:,}")
        col2.metric("Edges", f"{stats['relationships']:,}")
    except Exception as e:
        st.warning(f"Neo4j unavailable: {e}")

    try:
        vstore = get_vector_store()
        st.metric("Indexed Chunks", f"{vstore.count():,}")
    except Exception as e:
        st.warning(f"Chroma unavailable: {e}")

    st.divider()

    # Citation cards from last answer
    if st.session_state.sources:
        st.subheader("Sources")
        for src in st.session_state.sources:
            with st.expander(f"📄 {src['label']}"):
                if src.get("source_url"):
                    st.caption(f"[CMS PDF]({src['source_url']})")
                st.write(src["text"])

    st.divider()
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.rerun()

# ------------------------------------------------------------------
# Main panel — Tabs
# ------------------------------------------------------------------
tab_chat, tab_graph = st.tabs(["💬 Chat", "🕸 Graph Explorer"])

# ------------------------------------------------------------------
# Tab 1: Chat interface
# ------------------------------------------------------------------
with tab_chat:
    st.header("Ask about Medicare Policy")

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("e.g. Does Medicare cover physical therapy?"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching policy knowledge graph..."):
                try:
                    retriever = get_retriever()
                    answerer = get_answerer()

                    retrieval = retriever.retrieve(question)
                    result = answerer.answer(question, retrieval)

                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    # Show entity context as info box
                    if retrieval.get("entities_found"):
                        entities_str = ", ".join(retrieval["entities_found"][:6])
                        st.info(f"**Entities detected:** {entities_str}")

                    # Store for sidebar
                    st.session_state.sources = sources
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                    # Trigger sidebar refresh
                    st.rerun()

                except Exception as e:
                    error_msg = f"Error generating answer: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

# ------------------------------------------------------------------
# Tab 2: Graph Explorer
# ------------------------------------------------------------------
with tab_graph:
    from pyvis.network import Network
    import streamlit.components.v1 as components

    st.header("Knowledge Graph Explorer")

    ENTITY_TYPES = ["Service", "Condition", "Coverage", "Requirement", "Term", "Policy"]
    COLOR_MAP = {
        "Service": "#4e8ef7",
        "Condition": "#e74c3c",
        "Coverage": "#2ecc71",
        "Requirement": "#f39c12",
        "Term": "#9b59b6",
        "Policy": "#f1c40f",
    }

    col1, col2, col3 = st.columns(3)
    with col1:
        focus_entity = st.text_input("Focus entity (optional)", placeholder="e.g. Physical Therapy")
    with col2:
        selected_types = st.multiselect(
            "Filter by type",
            options=ENTITY_TYPES,
            default=ENTITY_TYPES,
        )
    with col3:
        max_nodes = st.slider("Max nodes", min_value=50, max_value=300, value=100, step=10)

    render_btn = st.button("Render Graph", type="primary")

    if render_btn:
        with st.spinner("Fetching subgraph from Neo4j..."):
            try:
                neo4j = get_neo4j_client()
                subgraph = neo4j.get_subgraph(
                    limit=max_nodes,
                    entity_types=selected_types if selected_types else None,
                    focus_entity=focus_entity.strip() if focus_entity.strip() else None,
                )

                nodes = subgraph["nodes"]
                edges = subgraph["edges"]

                if not nodes:
                    st.warning("No nodes found for the current filters. Try broadening your search.")
                else:
                    net = Network(
                        height="780px",
                        width="100%",
                        bgcolor="#0e1117",
                        font_color="#ffffff",
                    )
                    net.barnes_hut()
                    net.set_options("""{
                      "interaction": {
                        "zoomSpeed": 0.3,
                        "tooltipDelay": 100
                      }
                    }""")

                    for node in nodes:
                        color = COLOR_MAP.get(node["type"], "#aaaaaa")
                        desc = node["description"] or "No description available"
                        tooltip = (
                            f"Name: {node['name']}<br>"
                            f"Type: {node['type']}<br>"
                            f"Description: {desc}"
                        )
                        net.add_node(
                            node["id"],
                            label=node["name"],
                            color=color,
                            title=tooltip,
                        )

                    for edge in edges:
                        evidence = edge["evidence"] or "N/A"
                        edge_tooltip = (
                            f"Relation: {edge['relation']}<br>"
                            f"Evidence: {evidence}"
                        )
                        net.add_edge(
                            edge["source"],
                            edge["target"],
                            title=edge_tooltip,
                            label=edge["relation"],
                        )

                    html = net.generate_html()
                    components.html(html, height=800, scrolling=False)

                    st.caption(f"Showing {len(nodes)} nodes and {len(edges)} edges.")

                    # Legend
                    legend_rows = "".join(
                        f"| <span style='background:{color};padding:2px 10px;border-radius:3px;'>&nbsp;</span> | {etype} |"
                        for etype, color in COLOR_MAP.items()
                    )
                    st.markdown(
                        "**Legend:**\n\n"
                        "| Color | Type |\n"
                        "|-------|------|\n"
                        + "\n".join(
                            f"| <span style='background:{color};padding:2px 12px;border-radius:3px;color:{color}'>..</span> | {etype} |"
                            for etype, color in COLOR_MAP.items()
                        ),
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Error loading graph: {e}")
