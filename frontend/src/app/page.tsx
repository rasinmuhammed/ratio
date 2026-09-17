"use client";

import { useRef, useEffect, useState } from "react";
import { motion, useInView } from "framer-motion";
import Link from "next/link";
import dynamic from "next/dynamic";
import { RatioWordmark } from "@/components/RatioWordmark";

// WebGL Citation network (SSR disabled)
const CitationWebGL = dynamic(
  () => import("@/components/CitationWebGL").then((m) => m.CitationWebGL),
  { ssr: false, loading: () => <div className="w-full h-full" /> }
);

const RagArchitectureDiagram = dynamic(
  () => import("@/components/RagArchitectureDiagram").then((m) => m.RagArchitectureDiagram),
  { ssr: false, loading: () => <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-ash)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', letterSpacing: '0.15em' }}>LOADING...</div> }
);

const PIPELINE_STEPS = [
  { id: "01", name: "Query Router", type: "Deterministic", accent: "#E5C158", glow: "rgba(229,193,88,0.12)", summary: "A rule-based classifier inspects the incoming query before any retriever is invoked. Statutory citations ('AIR 1974 SC 224', 'Section 300 IPC') are dispatched directly to an exact match index, while natural language queries are sent to vector search.", why: "Dense vector embeddings collapse exact citations like 'AIR' into generic 'air', destroying legal precision. Measured on a 313-query labelled set: 0.925 recall and a perfect 1.000 MRR on exact-citation queries, up from 0.574 for BM25 alone and 0.049 for dense retrieval.", badge: "EXACT ROUTING" },
  { id: "02", name: "Hybrid Retrieval", type: "BM25 × BGE-Small", accent: "#5882C7", glow: "rgba(88,130,199,0.14)", summary: "Dual retrievers execute in parallel: a case-sensitive BM25 engine for statutory terms, and a bi-encoder dense model (BAAI/bge-small-en-v1.5) for semantic similarity. Candidate lists are combined via Reciprocal Rank Fusion (RRF).", why: "Legal arguments alternate between strict statutory language and doctrinal reasoning. Neither dense nor sparse retrieval alone reaches 90%+ recall.", badge: "RRF MERGE" },
  { id: "03", name: "Authority Weighting", type: "Court Hierarchy", accent: "#E5C158", glow: "rgba(229,193,88,0.12)", summary: "The Indian judicial hierarchy - Supreme Court → High Courts → Tribunals - is encoded as an algebraic multiplier in the re-ranking formula. A binding Supreme Court precedent outranks a single-judge order of equal semantic score.", why: "Relevance without judicial authority is legally useless. A matching passage from a reversed or lower court order is not binding law.", badge: "PRECEDENT SCORING" },
  { id: "04", name: "Stance Detection", type: "Holding vs Submission", accent: "#429B6C", glow: "rgba(66,155,108,0.12)", summary: "Every candidate passage is tagged during indexing: does it record the court's ratio decidendi (holding), or counsel's submission? The prompt generator filters out party arguments before synthesis.", why: "Judgments report what counsel submitted, what the court rejected, and what it held in the same judgment. Without stance tagging, models hallucinate advocate submissions as law.", badge: "RATIO ISOLATION" },
  { id: "05", name: "Corrective RAG", type: "CRAG Fallback", accent: "#8B61C7", glow: "rgba(139,97,199,0.14)", summary: "If initial context fails confidence checks or lacks definitive precedent, a self-reflection loop triggers: the query is decomposed into discrete legal sub-questions, retrieved independently, and synthesized.", why: "Complex legal issues span multiple statutory sections. Single-pass retrieval misses sub-holdings. CRAG treats low confidence as a trigger for query decomposition.", badge: "SELF-REFLECTION" },
];

const INNOVATIONS = [
  { tag: "SAC", title: "Summary-Augmented Chunking", body: "Every 450-token chunk is automatically prefixed with a machine-generated 3-sentence summary of its parent judgment - preserving the core issue, ruling, and context in every vector embedding. Chunk size was chosen for defensible reasons; it has not itself been swept against the label set, which is recorded rather than implied.", metric: "450 Tokens", metricLabel: "Tokens Per Chunk (SAC-Prefixed)" },
  { tag: "CS-BM25", title: "Case-Sensitive Tokenisation", body: "Standard LLM tokenizers lowercase all text, destroying citation sensitivity ('AIR' → 'air', 'CrPC' → 'crpc'). Ratio indexes each identifier twice: once atomic and case-preserved, once split into lowercase words, so exact and topical matching share one postings list.", metric: "580,939", metricLabel: "Chunks In The Full Corpus" },
  { tag: "SCO", title: "Enforced Citation Schema", body: "Generation prompts enforce a strict JSON output schema. A claim without a source ID cannot be emitted at all, measured at 0% uncited claims and 91.2% citation precision on a 15-query structured-output benchmark - the schema forces attribution, it does not by itself guarantee the citation is right.", metric: "0%", metricLabel: "Uncited Claims (Structured Output)" },
];

const STATS = [
  { value: "10,588", label: "Indian Court Judgments" },
  { value: "580,939", label: "Indexed Context Chunks" },
  { value: "IFM K2-Horizon", label: "Inference LLM (375B MoE)" },
  { value: "73.5%", label: "Recall@5 (Measured, 313 Labels)" },
];

const COURT_ROWS = [
  { label: "Supreme Court of India", sub: "Binding Precedent (Art. 141)", weight: "WEIGHT 1.0", color: "var(--color-gold)", dot: "var(--color-gold)", bg: "var(--color-gold-glow)" },
  { label: "High Courts (State Benches)", sub: "Persuasive / Regional Authority", weight: "WEIGHT 0.75", color: "var(--color-sapphire-bright)", dot: "var(--color-sapphire-bright)", bg: "var(--color-sapphire-glow)" },
  { label: "Specialised Tribunals (NCLT, NCLAT, ITAT)", sub: "Domain Specific Orders", weight: "WEIGHT 0.50", color: "var(--color-parchment)", dot: "var(--color-mist)", bg: "var(--color-graphite-light)" },
];

function FadeIn({ children, delay = 0, className = "", style = {} }: { children: React.ReactNode; delay?: number; className?: string; style?: React.CSSProperties }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-40px" });
  return (
    <motion.div ref={ref} initial={{ opacity: 0, y: 20 }} animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }} transition={{ duration: 0.7, delay, ease: [0.16, 1, 0.3, 1] }} className={className} style={style}>
      {children}
    </motion.div>
  );
}

function HeaderNav() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 30);
    window.addEventListener("scroll", h, { passive: true });
    return () => window.removeEventListener("scroll", h);
  }, []);

  return (
    <header style={{ position: "fixed", top: 0, left: 0, right: 0, zIndex: 50, transition: "all 0.3s ease", background: scrolled ? "rgba(7,7,10,0.97)" : "transparent", backdropFilter: scrolled ? "blur(16px) saturate(1.4)" : "none", WebkitBackdropFilter: scrolled ? "blur(16px) saturate(1.4)" : "none", borderBottom: scrolled ? "1px solid var(--color-graphite-border)" : "1px solid transparent" }}>
      <div className="ratio-container" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: "5rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
          <RatioWordmark size="sm" />
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", padding: "0.25rem 0.75rem", borderRadius: "9999px", border: "1px solid var(--color-graphite-border)", background: "var(--color-graphite-deep)" }}>
            <span style={{ width: "0.5rem", height: "0.5rem", borderRadius: "9999px", background: "var(--color-emerald-bright)", display: "inline-block" }} />
            <span className="label-tag" style={{ color: "var(--color-parchment)", fontSize: "10px" }}>INDEX LIVE · 580K CHUNKS</span>
          </div>
        </div>
        <nav style={{ display: "flex", alignItems: "center", gap: "2rem" }}>
          {["Architecture", "Innovations", "Corpus"].map((label) => (
            <a key={label} href={`#${label.toLowerCase()}`} style={{ fontSize: "0.7rem", textTransform: "uppercase", letterSpacing: "0.15em", color: "var(--color-fog)", textDecoration: "none", transition: "color 0.2s ease" }} onMouseEnter={(e) => (e.currentTarget.style.color = "var(--color-ivory)")} onMouseLeave={(e) => (e.currentTarget.style.color = "var(--color-fog)")}>{label}</a>
          ))}
        </nav>
        <Link href="/research" style={{ display: "flex", alignItems: "center", gap: "0.5rem", background: "var(--color-ivory)", color: "var(--color-void-text)", fontSize: "0.75rem", fontWeight: 600, letterSpacing: "0.05em", padding: "0.625rem 1.25rem", borderRadius: "0.5rem", textDecoration: "none" }}>
          <span>Open Workspace</span><span>→</span>
        </Link>
      </div>
    </header>
  );
}

function HeroSection() {
  return (
    <section style={{ position: "relative", minHeight: "100dvh", display: "flex", flexDirection: "column", justifyContent: "center", paddingTop: "8rem", paddingBottom: "5rem", overflow: "hidden" }}>
      <div className="hero-mesh" />
      <div className="hero-grid-pattern" />
      <div style={{ position: "absolute", inset: 0, opacity: 0.5, pointerEvents: "none" }}><CitationWebGL /></div>
      <div className="ratio-container" style={{ position: "relative", zIndex: 10 }}>
        <div className="hero-content">
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }} className="hero-eyebrow">
            <span className="label-tag">RATIO DECIDENDI · LEGAL RAG ENGINE</span>
          </motion.div>
          <motion.h1 initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9, delay: 0.1 }} className="hero-title">Ratio</motion.h1>
          <motion.p initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9, delay: 0.2 }} className="hero-subtitle">
            Indian legal AI built not to search text strings, but to extract{" "}
            <span style={{ color: "var(--color-ivory)", fontStyle: "normal", fontWeight: 500 }}>judicial authority &amp; ratio decidendi</span>{" "}
            across ten thousand court judgments.
          </motion.p>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9, delay: 0.3 }} className="hero-ctas">
            <Link href="/research" style={{ display: "flex", alignItems: "center", gap: "0.75rem", background: "var(--color-ivory)", color: "var(--color-void-text)", fontWeight: 600, fontSize: "0.875rem", padding: "1rem 2rem", borderRadius: "0.75rem", textDecoration: "none", boxShadow: "0 8px 30px rgba(255,255,255,0.1)" }}>
              <span>Launch Research Workspace</span><span>→</span>
            </Link>
            <a href="#architecture" style={{ display: "flex", alignItems: "center", gap: "0.5rem", border: "1px solid var(--color-graphite-border)", background: "var(--color-graphite-deep)", color: "var(--color-parchment)", fontSize: "0.875rem", padding: "1rem 1.75rem", borderRadius: "0.75rem", textDecoration: "none" }}>
              <span>Explore Pipeline Architecture</span><span>↓</span>
            </a>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 1, delay: 0.45 }} className="hero-metrics glass-panel" style={{ borderRadius: "1rem" }}>
            <div className="hero-metrics-inner" style={{ padding: "1.75rem 2rem" }}>
              {STATS.map((stat, idx) => (
                <div key={idx} className="hero-metric-item">
                  <span className="hero-metric-value">{stat.value}</span>
                  <span className="hero-metric-label">{stat.label}</span>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}


function RatioDecidendiSection() {
  const POINTS = [
    {
      label: "NOT the outcome",
      body: "Whether the appeal was allowed or dismissed is the dispositio. It is not the ratio. Two cases can have identical ratios and opposite outcomes.",
      color: "var(--color-rust)",
    },
    {
      label: "NOT the judge's commentary",
      body: "Obiter dicta are observations made in passing. Persuasive in future courts, but not binding. Retrieving them as holdings is a structural error.",
      color: "var(--color-gold-dim)",
    },
    {
      label: "The binding legal principle",
      body: "The ratio is the rule the court articulates as the basis of its decision. Under Article 141, every Supreme Court ratio binds all courts in India. This is what Ratio retrieves.",
      color: "var(--color-emerald-bright)",
    },
  ];

  return (
    <section id="ratio-decidendi" className="ratio-section" style={{ borderTop: "1px solid var(--color-graphite-border)" }}>
      <div className="ratio-container">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "5rem", alignItems: "start" }}>
          <FadeIn>
            <span className="ratio-section-label" style={{ color: "var(--color-gold)" }}>WHAT IS RATIO DECIDENDI?</span>
            <h2 style={{ fontFamily: "var(--font-serif)", fontWeight: 300, fontSize: "clamp(1.75rem, 3vw, 2.75rem)", lineHeight: 1.15, color: "var(--color-ivory)", margin: "1.25rem 0 1.5rem" }}>
              The reason<br />for the decision.
            </h2>
            <p style={{ color: "var(--color-fog)", fontSize: "1rem", lineHeight: 1.8, fontWeight: 300, marginBottom: "1.5rem" }}>
              Latin for the legal principle a court articulates when deciding a case. Not the outcome. Not the commentary. The rule that future courts are bound to follow.
            </p>
            <p style={{ color: "var(--color-fog)", fontSize: "1rem", lineHeight: 1.8, fontWeight: 300 }}>
              Under Article 141 of the Indian Constitution, any ratio declared by the Supreme Court is law across the entire country. This product is named after the thing it actually retrieves.
            </p>
          </FadeIn>
          <FadeIn delay={0.2}>
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              {POINTS.map((p, i) => (
                <div key={i} style={{ padding: "1.5rem", borderRadius: "0.875rem", border: "1px solid var(--color-graphite-border)", background: "var(--color-graphite-deep)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.625rem", marginBottom: "0.75rem" }}>
                    <span style={{ width: "0.5rem", height: "0.5rem", borderRadius: "9999px", background: p.color, flexShrink: 0 }} />
                    <span style={{ fontSize: "0.6875rem", fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase" as const, color: p.color, fontFamily: "var(--font-mono)" }}>{p.label}</span>
                  </div>
                  <p style={{ fontSize: "0.9rem", color: "var(--color-fog)", lineHeight: 1.7, margin: 0 }}>{p.body}</p>
                </div>
              ))}
            </div>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}

function EditorialSection() {
  return (
    <section className="ratio-section">
      <div className="ratio-container" style={{ maxWidth: "52rem", margin: "0 auto", textAlign: "center" }}>
        <FadeIn>
          <span className="ratio-section-label" style={{ color: "var(--color-gold)" }}>THE GROUNDING PROBLEM</span>
          <h2 style={{ fontFamily: "var(--font-serif)", fontWeight: 300, fontSize: "clamp(1.75rem, 4vw, 3rem)", lineHeight: 1.2, color: "var(--color-ivory)", marginBottom: "2rem" }}>
            Generic RAG treats legal judgments as generic documents.{" "}
            <em style={{ color: "var(--color-parchment)" }}>Legal text is not generic prose.</em>
          </h2>
        </FadeIn>
        <FadeIn delay={0.15}>
          <p style={{ fontSize: "1.125rem", color: "var(--color-fog)", lineHeight: 1.8, fontWeight: 300, marginBottom: "2rem" }}>
            A single Supreme Court judgment records counsel submissions, lower court findings, distinguished precedents, and the court&apos;s own binding holding - often in the exact same paragraph.
          </p>
        </FadeIn>
        <FadeIn delay={0.25}>
          <div style={{ padding: "1.5rem", borderRadius: "0.75rem", border: "1px solid var(--color-graphite-border)", background: "var(--color-graphite-deep)", textAlign: "left", maxWidth: "36rem", margin: "0 auto" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.75rem" }}>
              <span style={{ width: "0.625rem", height: "0.625rem", borderRadius: "9999px", background: "var(--color-gold)", flexShrink: 0 }} />
              <span className="label-tag" style={{ color: "var(--color-ivory)" }}>RATIO PRINCIPLE</span>
            </div>
            <p style={{ fontSize: "0.875rem", color: "var(--color-parchment)", lineHeight: 1.7 }}>
              Disambiguating counsel submission from binding judicial ratio is not an aesthetic preference - it is a structural legal requirement. Ratio\'s stance detection is built around this distinction.
            </p>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}

function ArchitectureSection() {
  return (
    <section id="architecture" className="ratio-section">
      <div className="ratio-container">
        <FadeIn className="ratio-section-header" style={{ maxWidth: "36rem" }}>
          <span className="ratio-section-label" style={{ color: "var(--color-gold)" }}>SYSTEM PIPELINE</span>
          <h2 className="ratio-section-title">Five layers of legal precision</h2>
          <p className="ratio-section-body">
            An interactive map of Ratio&apos;s retrieval pipeline. Use the arrows, the dots, or drag to move through it, click any stage to read the engineering rationale.
          </p>
        </FadeIn>
      </div>

      {/* Full-bleed interactive diagram */}
      <div
        style={{
          width: "100%",
          height: "760px",
          position: "relative",
          background: "linear-gradient(180deg, var(--color-void) 0%, var(--color-graphite-deep) 40%, var(--color-graphite-deep) 60%, var(--color-void) 100%)",
          borderTop: "1px solid var(--color-graphite-border)",
          borderBottom: "1px solid var(--color-graphite-border)",
          marginTop: "2rem",
          overflow: "hidden",
        }}
      >
        {/* Grid background */}
        <div style={{ position: "absolute", inset: 0, backgroundImage: "linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px)", backgroundSize: "48px 48px", maskImage: "radial-gradient(ellipse 80% 80% at 50% 50%, black 0%, transparent 100%)", pointerEvents: "none" }} />
        <RagArchitectureDiagram />
      </div>

      {/* Full diagram link */}
      <div className="ratio-container" style={{ marginTop: "1.5rem", display: "flex", justifyContent: "center" }}>
        <a
          href="/architecture"
          style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem", fontSize: "0.75rem", fontWeight: 600, letterSpacing: "0.08em", color: "var(--color-fog)", textDecoration: "none", border: "1px solid var(--color-graphite-border)", padding: "0.625rem 1.25rem", borderRadius: "0.5rem", background: "var(--color-graphite-deep)", transition: "all 0.2s ease" }}
          onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--color-graphite-light)"; e.currentTarget.style.color = "var(--color-ivory)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--color-graphite-border)"; e.currentTarget.style.color = "var(--color-fog)"; }}
        >
          <span>Open Full Pipeline Explorer</span>
          <span>↗</span>
        </a>
      </div>
    </section>
  );
}

function InnovationsSection() {
  return (
    <section id="innovations" className="ratio-section">
      <div className="ratio-container">
        <FadeIn className="ratio-section-header" style={{ maxWidth: "36rem" }}>
          <span className="ratio-section-label" style={{ color: "var(--color-emerald-bright)" }}>CORE INNOVATIONS</span>
          <h2 className="ratio-section-title">Built for judicial ground truth</h2>
          <p className="ratio-section-body">Structural guarantees where the schema can enforce them, honest measured numbers where it can't - stance labelling reduces hallucinated holdings, it isn't a hard guarantee the way the citation schema is.</p>
        </FadeIn>
        <div className="innovations-grid">
          {INNOVATIONS.map((inn, idx) => (
            <FadeIn key={inn.tag} delay={idx * 0.1}>
              <div className="spotlight-card innovation-card">
                <div className="innovation-card-body">
                  <span className="innovation-card-tag">{inn.tag}</span>
                  <h3 className="innovation-card-title">{inn.title}</h3>
                  <p className="innovation-card-text">{inn.body}</p>
                </div>
                <div className="innovation-card-metric">
                  <span className="innovation-card-metric-value">{inn.metric}</span>
                  <span className="innovation-card-metric-label">{inn.metricLabel}</span>
                </div>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  );
}

function CorpusSection() {
  return (
    <section id="corpus" className="ratio-section">
      <div className="ratio-container">
        <div className="corpus-grid">
          <FadeIn>
            <span className="ratio-section-label" style={{ color: "var(--color-sapphire-bright)" }}>THE INDEXED CORPUS</span>
            <h2 style={{ fontFamily: "var(--font-serif)", fontWeight: 300, fontSize: "clamp(2rem, 4vw, 3rem)", lineHeight: 1.15, color: "var(--color-ivory)", marginBottom: "1.5rem" }}>A decade of Indian jurisprudence, fully indexed.</h2>
            <p style={{ color: "var(--color-parchment)", fontSize: "1rem", lineHeight: 1.75, marginBottom: "1.5rem" }}>
              Built over the Hugging Face{" "}
              <a href="https://huggingface.co/datasets/opennyaiorg/InJudgements_dataset" target="_blank" rel="noopener noreferrer" style={{ color: "var(--color-gold)", textDecoration: "underline", textUnderlineOffset: "4px" }}>OpenNyAI InJudgements dataset</a>
              . Supreme Court of India, High Courts, and Tribunals - pre-processed into a 580,939-chunk hybrid search index.
            </p>
            <p style={{ color: "var(--color-fog)", fontSize: "0.875rem", lineHeight: 1.75 }}>Every chunk carries parent metadata, tribunal tier weight, and full case citations to prevent context dilution during generation.</p>
          </FadeIn>
          <FadeIn delay={0.2}>
            <div className="glass-panel" style={{ padding: "2rem", borderRadius: "1rem" }}>
              <h3 className="label-tag" style={{ color: "var(--color-ash)", marginBottom: "1.5rem", fontSize: "10px", display: "block" }}>JUDICIAL HIERARCHY RE-RANKING WEIGHTS</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                {COURT_ROWS.map((row) => (
                  <div key={row.label} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "1rem", borderRadius: "0.75rem", background: "var(--color-graphite-deep)", border: "1px solid var(--color-graphite-border)", gap: "1rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                      <span style={{ width: "0.75rem", height: "0.75rem", borderRadius: "9999px", background: row.dot, flexShrink: 0 }} />
                      <div>
                        <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--color-ivory)", display: "block" }}>{row.label}</span>
                        <span style={{ fontSize: "0.75rem", color: "var(--color-fog)" }}>{row.sub}</span>
                      </div>
                    </div>
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", fontWeight: 700, color: row.color, padding: "0.25rem 0.625rem", borderRadius: "0.25rem", background: row.bg, whiteSpace: "nowrap", flexShrink: 0 }}>{row.weight}</span>
                  </div>
                ))}
              </div>
            </div>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}

function BottomCTA() {
  return (
    <section className="cta-section">
      <div className="hero-mesh" />
      <div className="ratio-container" style={{ position: "relative", zIndex: 10 }}>
        <FadeIn>
          <div className="cta-inner">
            <span className="cta-label">READY FOR RESEARCH</span>
            <h2 className="cta-title">Legal reasoning, finally legible.</h2>
            <p className="cta-body">Ask complex legal questions. Receive verified holdings, verbatim passage citations, and precedent authority scores.</p>
            <Link href="/research" style={{ display: "inline-flex", alignItems: "center", gap: "0.75rem", background: "var(--color-ivory)", color: "var(--color-void-text)", fontWeight: 600, fontSize: "0.875rem", padding: "1rem 2.25rem", borderRadius: "0.75rem", textDecoration: "none", boxShadow: "0 12px 40px rgba(255,255,255,0.15)" }}>
              <span>Enter the Workspace</span><span>→</span>
            </Link>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer style={{ paddingTop: "2rem", paddingBottom: "2rem", borderTop: "1px solid var(--color-graphite-border)", background: "var(--color-void)" }}>
      <div className="ratio-container" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "1rem" }}>
        <RatioWordmark size="sm" />
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "1.5rem" }}>
          {["BM25 × BGE-SMALL", "K2-HORIZON-375B-A23B", "OPENNYAI DATASET"].map((tag) => (
            <span key={tag} className="label-tag" style={{ color: "var(--color-ash)", fontSize: "9px" }}>{tag}</span>
          ))}
        </div>
      </div>
    </footer>
  );
}

export default function LandingPage() {
  return (
    <div style={{ background: "var(--color-void)", minHeight: "100vh", color: "var(--color-ivory)" }}>
      <HeaderNav />
      <HeroSection />
      <RatioDecidendiSection />
      <EditorialSection />
      <ArchitectureSection />
      <InnovationsSection />
      <CorpusSection />
      <BottomCTA />
      <Footer />
    </div>
  );
}
