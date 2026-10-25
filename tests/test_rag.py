"""
Unit tests for the Compliance RAG Engine and Policy Retriever.
"""

import sys
import os
import pytest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_DIR)

from rag.engine import ComplianceKnowledgeRetriever, ComplianceRAGPipeline


def test_knowledge_base_indexing():
    retriever = ComplianceKnowledgeRetriever()
    assert len(retriever.chunks) > 0, "RAG Knowledge base must load policy chunks"
    assert retriever.chunk_vectors is not None


def test_semantic_policy_retrieval():
    retriever = ComplianceKnowledgeRetriever()
    # Query related to wire transfer and large drain
    results = retriever.retrieve("large wire transfer draining account SAR FinCEN", top_k=2)
    assert len(results) == 2
    assert "BSA" in results[0]["doc_title"] or "AML" in results[0]["doc_title"]
    assert results[0]["relevance_score"] > 0.0


def test_rag_compliance_dossier_critical():
    pipeline = ComplianceRAGPipeline()
    dossier = pipeline.generate_compliance_dossier(
        transaction_id="TXN-UNITTEST-RAG",
        amount=6500.0,
        risk_score=95.0,
        risk_tier="CRITICAL",
        dist_from_home_km=2500.0,
        balance_drain_pct=92.0,
        failed_pins=2,
        tx_type="TRANSFER"
    )

    assert dossier["sar_filing_required"] is True
    assert "FinCEN" in dossier["sar_filing_justification"]
    assert len(dossier["retrieved_policies"]) > 0
    assert len(dossier["compliance_timelines"]) > 0
