"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { RatioWordmark } from "@/components/RatioWordmark";
import { ArrowLeft, Database, Cpu, Search, Zap, Layers, Server } from "lucide-react";

const RagArchitectureDiagram = dynamic(
  () => import("@/components/RagArchitectureDiagram").then((m) => m.RagArchitectureDiagram),
  { ssr: false, loading: () => (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--color-ash)", fontFamily: "var(--font-mono)", fontSize: "0.75rem", letterSpacing: "0.15em" }}>
      LOADING PIPELINE...
    </div>
  ) }
);

const specs = [
  {
    category: "Retrieval Engines",
    icon: <Search size={18} />,
    items: [
      { label: "Dense Vector Model", value: "BAAI/bge-small-en-v1.5" },
      { label: "Vector Dimensions", value: "384d" },
      { label: "Sparse Index", value: "BM25 (Lexical)" },
      { label: "Reranking Model", value: "BAAI/bge-reranker-base" },
      { label: "Retrieval Strategy", value: "Hybrid (Alpha=0.75) + Cross-Encoder" },
    ]
  },
  {
    category: "Agentic Reasoning",
    icon: <Cpu size={18} />,
    items: [
      { label: "Base LLM", value: "IFM/K2-Horizon-375B-A23B" },
      { label: "Inference Mode", value: "Streaming Server-Sent Events (SSE)" },
      { label: "Reasoning Loop", value: "Thought-Action-Observation (ReAct)" },
      { label: "Context Window", value: "128,000 tokens" },
      { label: "System Prompt", value: "Contextual Legal Analysis Constraints" },
    ]
  },
  {
    category: "Data Infrastructure",
    icon: <Database size={18} />,
    items: [
      { label: "Vector Database", value: "LanceDB (Memory-mapped columnar)" },
      { label: "Relational Database", value: "SQLite3 (Metadata & Lineage)" },
      { label: "Total Corpus Size", value: "580,939 processed chunks" },
      { label: "Data Sharding", value: "12 parallel shards" },
      { label: "Chunking Strategy", value: "Recursive Character (1024 / 200)" },
    ]
  },
  {
    category: "Compute & Deployment",
    icon: <Server size={18} />,
    items: [
      { label: "Backend API", value: "FastAPI + Uvicorn" },
      { label: "Runtime Environment", value: "Google Cloud Run (Serverless)" },
      { label: "Frontend Framework", value: "Next.js 14 App Router" },
      { label: "Edge Deployment", value: "Vercel Edge Network" },
      { label: "Containerization", value: "Docker (Multi-stage build)" },
    ]
  }
];

export default function ArchitecturePage() {
  return (
    <div style={{ background: "var(--color-void)", minHeight: "100vh", display: "flex", flexDirection: "column", color: "var(--color-ivory)" }}>
      {/* Header */}
      <header style={{ position: "fixed", top: 0, left: 0, right: 0, zIndex: 50, background: "rgba(7,7,10,0.92)", backdropFilter: "blur(16px) saturate(1.4)", WebkitBackdropFilter: "blur(16px) saturate(1.4)", borderBottom: "1px solid var(--color-graphite-border)" }}>
        <div className="ratio-container" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: "4.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
            <Link href="/" style={{ textDecoration: "none" }}><RatioWordmark size="sm" /></Link>
            <span style={{ width: "1px", height: "1.25rem", background: "var(--color-graphite-border)" }} />
            <span style={{ fontSize: "0.7rem", fontWeight: 600, letterSpacing: "0.18em", textTransform: "uppercase", color: "var(--color-ash)", fontFamily: "var(--font-mono)" }}>System Architecture</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            <Link href="/" style={{ fontSize: "0.75rem", color: "var(--color-fog)", textDecoration: "none", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <ArrowLeft size={14} /> Back
            </Link>
          </div>
        </div>
      </header>

      <main style={{ flex: 1, paddingTop: "4.5rem", paddingBottom: "8rem" }}>
        {/* Interactive Diagram Section */}
        <section style={{ borderBottom: "1px solid var(--color-graphite-border)", background: "var(--color-obsidian)" }}>
          <div style={{ padding: "4rem 2rem 2rem", maxWidth: "1400px", margin: "0 auto" }}>
            <div style={{ marginBottom: "2rem", maxWidth: "600px" }}>
              <h1 style={{ fontSize: "2rem", fontWeight: 300, letterSpacing: "-0.02em", color: "var(--color-ivory)", marginBottom: "1rem" }}>
                Interactive Pipeline
              </h1>
              <p style={{ color: "var(--color-parchment)", fontSize: "0.95rem", lineHeight: 1.6 }}>
                The core execution flow of the Ratio agent. Click any node in the graph below to inspect its operational logic and data transformations.
              </p>
            </div>
            <div style={{ height: "60vh", minHeight: "500px", borderRadius: "12px", border: "1px solid var(--color-graphite-border)", overflow: "hidden", background: "var(--color-void)" }}>
              <RagArchitectureDiagram />
            </div>
          </div>
        </section>

        {/* Technical Specifications Section */}
        <section style={{ paddingTop: "6rem" }}>
          <div className="ratio-container">
            <div style={{ marginBottom: "4rem", maxWidth: "600px" }}>
              <h2 style={{ fontSize: "1.75rem", fontWeight: 300, letterSpacing: "-0.01em", color: "var(--color-ivory)", marginBottom: "1rem" }}>
                Technical Specifications
              </h2>
              <p style={{ color: "var(--color-parchment)", fontSize: "0.95rem", lineHeight: 1.6 }}>
                A complete breakdown of the models, infrastructure, and parameters powering the production deployment. Built for scale, low latency, and deterministic reasoning.
              </p>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "2rem" }}>
              {specs.map((specGroup, idx) => (
                <div key={idx} style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--color-graphite-border)", borderRadius: "12px", padding: "2rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "2rem" }}>
                    <div style={{ width: "32px", height: "32px", borderRadius: "8px", background: "rgba(229,193,88,0.1)", color: "var(--color-gold)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      {specGroup.icon}
                    </div>
                    <h3 style={{ fontSize: "1.1rem", fontWeight: 500, color: "var(--color-ivory)" }}>{specGroup.category}</h3>
                  </div>
                  
                  <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
                    {specGroup.items.map((item, i) => (
                      <div key={i} style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                        <span style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--color-ash)", fontFamily: "var(--font-mono)" }}>
                          {item.label}
                        </span>
                        <span style={{ fontSize: "0.95rem", color: "var(--color-parchment)", fontWeight: 400 }}>
                          {item.value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
