"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface DiagNode {
  id: string;
  label: string;
  sub: string;
  accent: string;
  glow: string;
  col: number;
  rowOffset: number; // 0 = center, -1 = top, 1 = bottom
  detail: { title: string; desc: string; tech: string; why: string };
}

interface Edge {
  from: string;
  to: string;
  label?: string;
  dashed?: boolean;
  feedback?: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Data
// ─────────────────────────────────────────────────────────────────────────────

const NODES: DiagNode[] = [
  {
    id: "query", label: "User Query", sub: "Entry point", col: 0, rowOffset: 0,
    accent: "#F4F2EB", glow: "rgba(244,242,235,0.08)",
    detail: { title: "Incoming Legal Query", tech: "Input Layer", desc: "A barrister may type 'What is the test for vicarious liability?' or paste 'AIR 1974 SC 224'. The system must handle both without conflation.", why: "Legal queries span free-form doctrine and exact statutory references. Treating them identically destroys retrieval quality." },
  },
  {
    id: "router", label: "Query Router", sub: "01 · Deterministic", col: 1, rowOffset: 0,
    accent: "#E5C158", glow: "rgba(229,193,88,0.14)",
    detail: { title: "Deterministic Query Router", tech: "Regex + Rule Engine", desc: "Inspects every query before retrieval fires. Regex patterns detect statutory citations (AIR, SCC) and dispatch them to exact-match BM25. Everything else routes to vector search.", why: "Dense embeddings collapse 'AIR' into 'air', destroying citation precision. Measured, not assumed: 0.925 recall and 1.000 MRR on exact-citation queries, versus 0.574 for BM25 alone - real and large, not a claim of zero loss." },
  },
  {
    id: "bm25", label: "BM25 Search", sub: "Case-sensitive tokeniser", col: 2, rowOffset: -1,
    accent: "#5882C7", glow: "rgba(88,130,199,0.14)",
    detail: { title: "Case-Sensitive BM25 Engine", tech: "BM25 + Custom Tokeniser", desc: "A modified BM25 index that preserves original capitalisation across 414,122 posting lists. Queries for 'CrPC' do not collapse into unrelated lowercase tokens.", why: "Standard NLP tokenisers lowercase everything. 'AIR' (All India Reporter) becomes 'air'. A custom tokeniser prevents this and preserves statutory citation fidelity." },
  },
  {
    id: "dense", label: "Dense Retrieval", sub: "BAAI/bge-small-en-v1.5", col: 2, rowOffset: 1,
    accent: "#8B61C7", glow: "rgba(139,97,199,0.14)",
    detail: { title: "Bi-Encoder Dense Retrieval", tech: "BAAI/bge-small-en-v1.5 + FAISS", desc: "A fine-tuned bi-encoder encodes queries and all chunks into a shared embedding space. Approximate nearest-neighbour search retrieves semantically similar passages even when no statutory keyword matches.", why: "Doctrinal questions like 'what is the doctrine of indoor management' have no exact statutory keyword. Only semantic search reaches relevant passages." },
  },
  {
    id: "rrf", label: "RRF Merge", sub: "02 · Reciprocal Rank Fusion", col: 3, rowOffset: 0,
    accent: "#5882C7", glow: "rgba(88,130,199,0.12)",
    detail: { title: "Reciprocal Rank Fusion", tech: "RRF with k=60", desc: "Candidate lists from BM25 and dense search are merged using RRF: each document gets score 1/(k + rank) from each list, then scores are summed.", why: "Measured, not assumed: on exact-citation queries, RRF fusion actually scores below BM25 alone (0.565 vs 0.574 recall in the 313-label benchmark), because averaging two systems that are both wrong about identifiers is still wrong. That negative result is why identifier queries bypass RRF entirely and go to the exact-match router below - fusion earns its place on conceptual queries, not by assumption on every query." },
  },
  {
    id: "authority", label: "Authority Weighting", sub: "03 · Court Hierarchy", col: 4, rowOffset: 0,
    accent: "#E5C158", glow: "rgba(229,193,88,0.12)",
    detail: { title: "Judicial Authority Weighting", tech: "Metadata-based re-ranker", desc: "Every passage carries a court-tier metadata field. A multiplier applies at re-ranking: Supreme Court (1.0x), High Courts (0.75x), Tribunals (0.5x).", why: "Semantic relevance without legal authority is useless. A passage from a reversed single-judge order may be textually identical to a binding SC holding but is not law." },
  },
  {
    id: "stance", label: "Stance Detection", sub: "04 · Holding vs Submission", col: 5, rowOffset: 0,
    accent: "#429B6C", glow: "rgba(66,155,108,0.12)",
    detail: { title: "Stance Detection Filter", tech: "Rule-based tagger", desc: "Every passage is tagged at index time: HOLDING (ratio decidendi), SUBMISSION (argument), DICTA, or PROCEDURE. Only HOLDING passages enter the synthesis prompt.", why: "Judgments record what counsel submitted AND what the court held in the same paragraph. Without stance tagging, the LLM hallucinates advocate submissions as settled law." },
  },
  {
    id: "crag", label: "CRAG Check", sub: "05 · Confidence Gate", col: 6, rowOffset: 0,
    accent: "#C75858", glow: "rgba(199,88,88,0.12)",
    detail: { title: "Corrective RAG Confidence Gate", tech: "CRAG + Query Decomposition", desc: "Retrieved context is scored on coverage. If no clear holding is found, a self-reflection loop activates: the query is decomposed into sub-questions, each retrieved independently.", why: "Complex legal matters span multiple sections. Single-pass retrieval misses sub-holdings. CRAG treats low confidence as a signal to decompose rather than hallucinate." },
  },
  {
    id: "context", label: "Context Assembly", sub: "Verbatim passage packing", col: 7, rowOffset: 0,
    accent: "#429B6C", glow: "rgba(66,155,108,0.10)",
    detail: { title: "Context Assembly", tech: "Prompt engineering + JSON schema", desc: "Top-k passages are assembled into a structured prompt. Each includes: verbatim text, source case name, court tier, year, passage ID, and stance label.", why: "Unstructured context leads to citation hallucination. A strict passage-ID schema forces the model to ground every claim in a retrievable source." },
  },
  {
    id: "llm", label: "LLM Generation", sub: "K2-Horizon-375B-A23B", col: 8, rowOffset: 0,
    accent: "#8B61C7", glow: "rgba(139,97,199,0.14)",
    detail: { title: "LLM Synthesis Layer", tech: "K2-Horizon-375B-A23B (MoE)", desc: "Generates a structured legal answer: holding summary, applicable legal test, and enumerated citations referencing passage IDs.", why: "A sparse mixture-of-experts model: 375B total parameters, roughly 23B active per token. Named for what it costs to run, not inflated as a dense 375B claim - the active-parameter count is the honest number for latency and compute." },
  },
  {
    id: "answer", label: "Cited Answer", sub: "Verified + attributable", col: 9, rowOffset: 0,
    accent: "#56C48B", glow: "rgba(86,196,139,0.12)",
    detail: { title: "Verified Legal Answer", tech: "Citation-grounded output", desc: "The final response carries: plain-language holding summary, ratio decidendi of each cited case, and verbatim passage text for every referenced claim.", why: "Legal professionals cannot act on unattributed AI output. Every claim must be verifiable against the original judgment text." },
  },
];

const EDGES: Edge[] = [
  { from: "query", to: "router" },
  { from: "router", to: "bm25", label: "Citation path" },
  { from: "router", to: "dense", label: "Semantic path" },
  { from: "bm25", to: "rrf" },
  { from: "dense", to: "rrf" },
  { from: "rrf", to: "authority" },
  { from: "authority", to: "stance" },
  { from: "stance", to: "crag" },
  { from: "crag", to: "context" },
  { from: "crag", to: "router", label: "Decompose", dashed: true, feedback: true },
  { from: "context", to: "llm" },
  { from: "llm", to: "answer" },
];

// ─────────────────────────────────────────────────────────────────────────────
// Layout Math
// ─────────────────────────────────────────────────────────────────────────────

const COL_W = 260;
const NODE_W = 210;
const NODE_H = 76;
const PAD_X = 48;
const CANVAS_W = NODES.length * COL_W + PAD_X * 2;
const CANVAS_H = 320;
const MID_Y = CANVAS_H / 2;

function getNodePos(node: DiagNode) {
  return {
    x: PAD_X + node.col * COL_W,
    y: MID_Y - (NODE_H / 2) + (node.rowOffset * 84),
  };
}

const nodeMap = Object.fromEntries(NODES.map((n) => [n.id, n]));
const nodeOrder = NODES.map((n) => n.id);

function buildPath(from: DiagNode, to: DiagNode): string {
  const fPos = getNodePos(from);
  const tPos = getNodePos(to);
  const fx = fPos.x + NODE_W;
  const fy = fPos.y + NODE_H / 2;
  const tx = tPos.x;
  const ty = tPos.y + NODE_H / 2;

  // Feedback loop (CRAG to Router)
  if (from.id === "crag" && to.id === "router") {
    const bottomY = CANVAS_H - 16;
    return `M ${fx - NODE_W/2} ${fPos.y + NODE_H} C ${fx - NODE_W/2} ${bottomY}, ${tx + NODE_W/2} ${bottomY}, ${tx + NODE_W/2} ${tPos.y + NODE_H}`;
  }

  // Straight line if same row
  if (from.rowOffset === to.rowOffset && from.rowOffset === 0) {
    return `M ${fx} ${fy} L ${tx} ${ty}`;
  }

  // S-curve for splitting/merging
  const midX = (fx + tx) / 2;
  return `M ${fx} ${fy} C ${midX} ${fy}, ${midX} ${ty}, ${tx} ${ty}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

export function RagArchitectureDiagram() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const infoPanelRef = useRef<HTMLDivElement>(null);
  const [activeId, setActiveId] = useState<string>("router");
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(true);

  // Drag-to-pan state. Refs, not state, because these update on every
  // mousemove and a state update per pixel would re-render the whole
  // diagram every frame of a drag.
  const dragState = useRef<{ dragging: boolean; startX: number; startScroll: number; moved: boolean }>({
    dragging: false, startX: 0, startScroll: 0, moved: false,
  });
  const [isDragging, setIsDragging] = useState(false);

  const updateEdgeFades = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setCanScrollLeft(el.scrollLeft > 4);
    setCanScrollRight(el.scrollLeft < el.scrollWidth - el.clientWidth - 4);
  }, []);

  useEffect(() => {
    updateEdgeFades();
    const el = scrollRef.current;
    if (!el) return;
    el.addEventListener("scroll", updateEdgeFades, { passive: true });
    window.addEventListener("resize", updateEdgeFades);
    return () => {
      el.removeEventListener("scroll", updateEdgeFades);
      window.removeEventListener("resize", updateEdgeFades);
    };
  }, [updateEdgeFades]);

  // Centres the given node in the visible scroll area and selects it.
  // Used by the arrow buttons, the progress dots and keyboard navigation,
  // so every way of moving through the pipeline lands in the same place.
  const goToId = useCallback((id: string) => {
    setActiveId(id);
    const el = scrollRef.current;
    const node = nodeMap[id];
    if (!el || !node) return;
    const pos = getNodePos(node);
    const target = pos.x + NODE_W / 2 - el.clientWidth / 2;
    el.scrollTo({ left: Math.max(0, target), behavior: "smooth" });
    // Scroll the info panel into view so the user sees the explanation
    setTimeout(() => {
      infoPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 80);
  }, []);

  const step = useCallback((dir: 1 | -1) => {
    const idx = nodeOrder.indexOf(activeId);
    const next = nodeOrder[Math.min(Math.max(idx + dir, 0), nodeOrder.length - 1)];
    goToId(next);
  }, [activeId, goToId]);

  const activeIndex = nodeOrder.indexOf(activeId);

  const activeNode = NODES.find((n) => n.id === activeId)!;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", width: "100%", background: "var(--color-void)" }}>

      {/* ─── TOP: PIPELINE STRIP ─── */}
      <div style={{ position: "relative", flex: 1, minHeight: 0 }}>
        <div
          ref={scrollRef}
          tabIndex={0}
          role="region"
          aria-label="Retrieval pipeline diagram, use the arrow keys or the buttons to move between stages"
          onKeyDown={(e) => {
            if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
            if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
          }}
          onMouseDown={(e) => {
            dragState.current = { dragging: true, startX: e.clientX, startScroll: scrollRef.current!.scrollLeft, moved: false };
            setIsDragging(true);
          }}
          onMouseMove={(e) => {
            if (!dragState.current.dragging || !scrollRef.current) return;
            const dx = e.clientX - dragState.current.startX;
            if (Math.abs(dx) > 3) dragState.current.moved = true;
            scrollRef.current.scrollLeft = dragState.current.startScroll - dx;
          }}
          onMouseUp={() => { dragState.current.dragging = false; setIsDragging(false); }}
          onMouseLeave={() => { dragState.current.dragging = false; setIsDragging(false); }}
          style={{
            height: "100%",
            width: "100%",
            overflowX: "auto",
            overflowY: "hidden",
            position: "relative",
            cursor: isDragging ? "grabbing" : "grab",
            borderBottom: "1px solid var(--color-graphite-border)",
            background: "linear-gradient(to bottom, var(--color-void) 0%, var(--color-graphite-deep) 100%)",
            outline: "none",
            userSelect: isDragging ? "none" : undefined,
          }}
          // No wheel-hijacking: a normal vertical scroll gesture over this
          // element used to be converted into horizontal pipeline scroll,
          // which meant scrolling the page with the cursor anywhere over
          // the diagram silently stopped working. Horizontal navigation is
          // now the buttons, the dots, drag, and native trackpad
          // two-finger horizontal swipe (which overflow-x:auto already
          // supports with no JS), none of which touch vertical scroll.
        >
          <div style={{ width: CANVAS_W, height: CANVAS_H, position: "relative" }}>
            {/* SVG Edges */}
            <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
              <defs>
                <marker id="arr-def" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="var(--color-graphite-border)" /></marker>
                <marker id="arr-hot" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#E5C158" /></marker>
                <marker id="arr-fb" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#C75858" /></marker>
              </defs>

              {EDGES.map((edge, i) => {
                const fn = nodeMap[edge.from];
                const tn = nodeMap[edge.to];
                const d = buildPath(fn, tn);
                const isActive = activeId === edge.from || activeId === edge.to || hoveredId === edge.from || hoveredId === edge.to;
                const isFb = edge.feedback;
                const color = isFb ? "#C75858" : isActive ? "#E5C158" : "var(--color-graphite-border)";
                const mark = isFb ? "arr-fb" : isActive ? "arr-hot" : "arr-def";

                return (
                  <g key={i}>
                    <path d={d} fill="none" stroke={color} strokeWidth={isActive ? 2 : 1} strokeDasharray={edge.dashed ? "4 4" : undefined} markerEnd={`url(#${mark})`} style={{ transition: "stroke 0.3s ease" }} />
                    {isActive && !isFb && (
                      <circle r={3} fill="#E5C158">
                        <animateMotion dur="2s" repeatCount="indefinite" path={d} />
                      </circle>
                    )}
                    {isActive && isFb && (
                      <circle r={3} fill="#C75858">
                        <animateMotion dur="2.5s" repeatCount="indefinite" path={d} />
                      </circle>
                    )}
                    {edge.label && (() => {
                      const fx = getNodePos(fn).x + NODE_W;
                      const tx = getNodePos(tn).x;
                      const mx = (fx + tx) / 2;
                      const my = (getNodePos(fn).y + getNodePos(tn).y) / 2 + (isFb ? 100 : NODE_H / 2 - 12);
                      return (
                        <text x={mx} y={my} textAnchor="middle" fill={color} fontSize="9.5" fontWeight={isActive ? 600 : 400} fontFamily="var(--font-mono)" letterSpacing="0.08em">{edge.label}</text>
                      );
                    })()}
                  </g>
                );
              })}
            </svg>

            {/* HTML Nodes */}
            {NODES.map((node) => {
              const pos = getNodePos(node);
              const isSel = activeId === node.id;
              const isHov = hoveredId === node.id;
              const active = isSel || isHov;

              return (
                <motion.div
                  key={node.id}
                  onClick={() => {
                    // A drag that ends over a node shouldn't also select it -
                    // dragState.moved distinguishes "clicked" from "dragged
                    // and happened to release here".
                    if (dragState.current.moved) return;
                    goToId(node.id);
                  }}
                  onMouseEnter={() => setHoveredId(node.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  animate={{
                    scale: active ? 1.06 : 1,
                    borderColor: isSel ? node.accent : isHov ? `${node.accent}60` : "var(--color-graphite-border)",
                    boxShadow: isSel
                      ? `0 0 0 1px ${node.accent}40, 0 16px 40px -10px ${node.glow}, inset 0 1px 0 rgba(255,255,255,0.08)`
                      : `0 8px 24px -6px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03)`
                  }}
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  style={{
                    position: "absolute",
                    left: pos.x,
                    top: pos.y,
                    width: NODE_W,
                    height: NODE_H,
                    borderRadius: "0.875rem",
                    borderWidth: "1px",
                    borderStyle: "solid",
                    background: "var(--color-void)",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    padding: "0 1.125rem",
                    gap: "0.75rem",
                    overflow: "hidden",
                    zIndex: active ? 10 : 1,
                  }}
                >
                  {/* Background glow */}
                  <div style={{ position: "absolute", inset: 0, background: active ? node.glow : "transparent", transition: "background 0.3s ease" }} />

                  {/* Accent bar */}
                  <div style={{ width: "3px", height: "36px", borderRadius: "2px", background: node.accent, opacity: active ? 1 : 0.35, transition: "opacity 0.3s ease", zIndex: 2, flexShrink: 0 }} />

                  <div style={{ zIndex: 2, minWidth: 0 }}>
                    <div style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--color-ivory)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{node.label}</div>
                    <div style={{ fontSize: "0.5625rem", color: active ? node.accent : "var(--color-ash)", letterSpacing: "0.09em", textTransform: "uppercase", fontFamily: "var(--font-mono)", marginTop: "0.2rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", transition: "color 0.3s ease" }}>{node.sub}</div>
                  </div>

                  {/* Animated selection ring component */}
                  {isSel && (
                    <motion.div layoutId="selection-ring" style={{ position: "absolute", inset: 0, border: `1.5px solid ${node.accent}`, borderRadius: "inherit", zIndex: 3 }} transition={{ type: "spring", stiffness: 300, damping: 30 }} />
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* Edge fades: a visual "there's more this way" cue, shown only
            on the side that actually has more to scroll to. */}
        <div style={{ position: "absolute", top: 0, bottom: "1px", left: 0, width: "64px", background: "linear-gradient(to right, var(--color-void), transparent)", pointerEvents: "none", opacity: canScrollLeft ? 1 : 0, transition: "opacity 0.25s ease" }} />
        <div style={{ position: "absolute", top: 0, bottom: "1px", right: 0, width: "64px", background: "linear-gradient(to left, var(--color-void), transparent)", pointerEvents: "none", opacity: canScrollRight ? 1 : 0, transition: "opacity 0.25s ease" }} />

        {/* Prev / next: an explicit, discoverable way to move through the
            pipeline, since relying on a visitor to guess "scroll sideways"
            was the whole problem with the previous version. */}
        <button
          aria-label="Previous stage"
          onClick={() => step(-1)}
          disabled={activeIndex === 0}
          style={{
            position: "absolute", left: "14px", top: "50%", transform: "translateY(-50%)",
            width: "36px", height: "36px", borderRadius: "50%", zIndex: 15,
            display: "flex", alignItems: "center", justifyContent: "center",
            background: "rgba(10,10,14,0.85)", border: "1px solid var(--color-graphite-border)",
            color: activeIndex === 0 ? "var(--color-graphite-border)" : "var(--color-ivory)",
            cursor: activeIndex === 0 ? "default" : "pointer",
            opacity: activeIndex === 0 ? 0.4 : 1,
            transition: "opacity 0.2s ease, border-color 0.2s ease",
            backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
          }}
        >
          <ChevronLeft size={18} strokeWidth={1.75} />
        </button>
        <button
          aria-label="Next stage"
          onClick={() => step(1)}
          disabled={activeIndex === nodeOrder.length - 1}
          style={{
            position: "absolute", right: "14px", top: "50%", transform: "translateY(-50%)",
            width: "36px", height: "36px", borderRadius: "50%", zIndex: 15,
            display: "flex", alignItems: "center", justifyContent: "center",
            background: "rgba(10,10,14,0.85)", border: "1px solid var(--color-graphite-border)",
            color: activeIndex === nodeOrder.length - 1 ? "var(--color-graphite-border)" : "var(--color-ivory)",
            cursor: activeIndex === nodeOrder.length - 1 ? "default" : "pointer",
            opacity: activeIndex === nodeOrder.length - 1 ? 0.4 : 1,
            transition: "opacity 0.2s ease, border-color 0.2s ease",
            backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
          }}
        >
          <ChevronRight size={18} strokeWidth={1.75} />
        </button>
      </div>

      {/* Progress dots: which of the N stages is active, and a direct way
          to jump to any of them, the same pattern a carousel or an
          onboarding flow already teaches people to expect. */}
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "0.5rem", padding: "0.875rem 0", background: "var(--color-void)", borderBottom: "1px solid var(--color-graphite-border)" }}>
        {NODES.map((node) => {
          const isSel = node.id === activeId;
          return (
            <button
              key={node.id}
              aria-label={`Go to ${node.label}`}
              onClick={() => goToId(node.id)}
              style={{
                width: isSel ? "20px" : "6px",
                height: "6px",
                borderRadius: "3px",
                background: isSel ? node.accent : "var(--color-graphite-border)",
                border: "none",
                cursor: "pointer",
                padding: 0,
                transition: "width 0.25s ease, background 0.25s ease",
              }}
            />
          );
        })}
      </div>

      {/* ─── BOTTOM: FIXED INFO PANEL ─── */}
      <div ref={infoPanelRef} className="architecture-diagram-info" style={{ height: "260px", background: "var(--color-void)", position: "relative", zIndex: 20 }}>
        <AnimatePresence mode="wait">
          <motion.div
            key={activeId}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="architecture-info-panel" style={{ width: "100%", maxWidth: "1200px", margin: "0 auto", padding: "2rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "3rem", height: "100%" }}
          >
            {/* Left Col: Title & Desc */}
            <div style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
                <span style={{ fontSize: "0.625rem", fontWeight: 700, letterSpacing: "0.15em", textTransform: "uppercase", color: activeNode.accent, fontFamily: "var(--font-mono)", background: activeNode.glow, border: `1px solid ${activeNode.accent}40`, padding: "0.25rem 0.625rem", borderRadius: "0.25rem" }}>
                  {activeNode.sub}
                </span>
                <span style={{ fontSize: "0.625rem", fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--color-ash)", fontFamily: "var(--font-mono)" }}>
                  {activeNode.detail.tech}
                </span>
                <span style={{ marginLeft: "auto", fontSize: "0.625rem", color: "var(--color-ash)", fontFamily: "var(--font-mono)" }}>
                  {activeIndex + 1} / {nodeOrder.length}
                </span>
              </div>
              <h3 style={{ fontSize: "1.5rem", fontWeight: 300, fontFamily: "var(--font-serif)", color: "var(--color-ivory)", marginBottom: "1rem" }}>
                {activeNode.detail.title}
              </h3>
              <p style={{ fontSize: "0.9375rem", color: "var(--color-fog)", lineHeight: 1.7 }}>
                {activeNode.detail.desc}
              </p>
            </div>

            {/* Right Col: Why it matters */}
            <div style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
              <div style={{ padding: "1.5rem", borderRadius: "0.75rem", background: activeNode.glow, border: `1px solid ${activeNode.accent}30`, height: "100%", display: "flex", flexDirection: "column", justifyContent: "center" }}>
                <span style={{ fontSize: "0.625rem", fontWeight: 700, letterSpacing: "0.15em", textTransform: "uppercase", color: activeNode.accent, display: "block", marginBottom: "0.75rem", fontFamily: "var(--font-mono)" }}>
                  Architectural Rationale
                </span>
                <p style={{ fontSize: "0.9375rem", color: "var(--color-ivory)", lineHeight: 1.7, fontWeight: 300 }}>
                  {activeNode.detail.why}
                </p>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
