"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import { AnimatePresence, motion, useMotionValue, useSpring } from "framer-motion";
import { SearchIcon, Landmark, Scale, ScrollText, Shield } from "lucide-react";

import { EvidencePanel } from "@/components/EvidencePanel";
import { AnswerCanvas } from "@/components/AnswerCanvas";
import { RatioWordmark } from "@/components/RatioWordmark";
import { AgentTerminal, AgentThought } from "@/components/AgentTerminal";

interface SourceMeta {
  index: number;
  court: string;
  title: string;
  url?: string;
  cited_by?: number;
  stance?: string;
  cited?: boolean;
}

type AppState = "idle" | "searching" | "complete";


function InfoTooltip({ title, body }: { title: string; body: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ position: "relative", display: "inline-flex" }}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-label={`Learn about ${title}`}
        style={{
          width: "18px", height: "18px", borderRadius: "50%",
          border: "1px solid var(--color-graphite-mid)",
          background: "transparent", color: "var(--color-ash)",
          fontSize: "10px", fontWeight: 600, cursor: "pointer",
          display: "flex", alignItems: "center", justifyContent: "center",
          lineHeight: 1, flexShrink: 0,
          transition: "border-color 0.15s ease, color 0.15s ease",
        }}
        onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--color-gold-dim)"; e.currentTarget.style.color = "var(--color-gold)"; }}
        onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--color-graphite-mid)"; e.currentTarget.style.color = "var(--color-ash)"; }}
      >
        ?
      </button>
      {open && (
        <div
          style={{
            position: "absolute", bottom: "calc(100% + 8px)", left: "50%",
            transform: "translateX(-50%)", zIndex: 100,
            width: "260px", padding: "1rem",
            background: "var(--color-obsidian)", border: "1px solid var(--color-graphite-border)",
            borderRadius: "0.75rem",
            boxShadow: "0 16px 48px -8px rgba(0,0,0,0.8)",
          }}
        >
          <p style={{ fontSize: "0.6875rem", fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--color-gold)", fontFamily: "var(--font-mono)", marginBottom: "0.5rem" }}>
            {title}
          </p>
          <p style={{ fontSize: "0.8125rem", color: "var(--color-fog)", lineHeight: 1.65, margin: 0 }}>{body}</p>
          <button onClick={() => setOpen(false)} style={{ position: "absolute", top: "8px", right: "10px", background: "none", border: "none", color: "var(--color-ash)", cursor: "pointer", fontSize: "14px", lineHeight: 1 }}>x</button>
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [appState, setAppState] = useState<AppState>("idle");
  const [statusText, setStatusText] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SourceMeta[]>([]);
  const [thoughts, setThoughts] = useState<AgentThought[]>([]);
  const [refused, setRefused] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Magnetic cursor effect on search wrapper
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    mouseX.set(e.clientX - rect.left - rect.width / 2);
    mouseY.set(e.clientY - rect.top - rect.height / 2);
  };

  const handleSubmit = useCallback(
    async (e?: React.FormEvent) => {
      e?.preventDefault();
      const q = query.trim();
      if (!q) return;

      setSubmittedQuery(q);
      setAppState("searching");
      setAnswer("");
      setSources([]);
      setThoughts([]);
      setRefused(false);
      setStatusText("Engaging Autonomous Agent...");

      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      try {
        const res = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}/agent_stream?query=${encodeURIComponent(q)}&k=6`,
          { signal: abortControllerRef.current.signal }
        );

        if (!res.body) throw new Error("No stream body");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const delimiterRegex = /\r?\n\r?\n/;
          let match;
          while ((match = delimiterRegex.exec(buffer)) !== null) {
            const chunk = buffer.slice(0, match.index);
            buffer = buffer.slice(match.index + match[0].length);

            let currentEvent = "";
            for (const line of chunk.split("\n")) {
              if (line.startsWith("event: ")) {
                currentEvent = line.slice(7).trim();
              } else if (line.startsWith("data: ")) {
                const payload = line.slice(6).trim();
                if (!payload) continue;
                try {
                  const data = JSON.parse(payload);
                  if (currentEvent === "status") {
                    setStatusText(data.message);
                    setThoughts((prev) => [...prev, { id: crypto.randomUUID(), type: "status", content: data.message }]);
                  } else if (currentEvent === "scratchpad") {
                    setThoughts((prev) => [...prev, { id: crypto.randomUUID(), type: "scratchpad", content: data.content }]);
                  } else if (currentEvent === "search") {
                    setThoughts((prev) => [...prev, { id: crypto.randomUUID(), type: "search", content: data.query }]);
                  } else if (currentEvent === "delta") {
                    setAnswer((prev) => prev + data.token);
                  } else if (currentEvent === "sources") {
                    setSources(data.sources);
                    setRefused(data.refused);
                    setAppState("complete");
                    setStatusText("");
                  }
                } catch (_) {}
              }
            }
          }
        }
      } catch (err: any) {
        if (err.name === 'AbortError') {
          console.log('Fetch aborted intentionally');
          return;
        }
        console.error(err);
        setStatusText("Connection error — ensure the Ratio engine is running.");
        setAppState("idle");
      }
    },
    [query]
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    }
  };

  const isActive = appState !== "idle";

  return (
    <div className="relative flex h-screen w-screen overflow-hidden bg-[var(--color-void)]">
      {/* Ambient background glow */}
      <div className="ambient-glow" />

      {/* ── IDLE STATE: Centered landing ── */}
      <AnimatePresence>
        {!isActive && (
          <motion.div
            key="landing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, scale: 0.97, transition: { duration: 0.3 } }}
            className="absolute inset-0 flex flex-col items-center justify-center z-10 px-6"
          >
            {/* Ambient subtle data-grid or radial background for AI active state */}
            <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", width: "1200px", height: "1200px", background: "radial-gradient(circle, rgba(229,193,88,0.035) 0%, transparent 50%)", zIndex: -1, pointerEvents: "none" }} />
            
            {/* Wordmark */}
            <motion.div
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: "spring", stiffness: 400, damping: 30 }}
              className="mb-4"
              style={{ filter: "drop-shadow(0 4px 24px rgba(229,193,88,0.15))" }}
            >
              <RatioWordmark size="xl" />
            </motion.div>

            {/* Tagline */}
            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: "spring", stiffness: 400, damping: 30, delay: 0.1 }}
              style={{ fontSize: "0.75rem", letterSpacing: "0.25em", textTransform: "uppercase", marginBottom: "4rem", fontWeight: 300, background: "linear-gradient(90deg, var(--color-fog) 0%, var(--color-ivory) 50%, var(--color-fog) 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}
            >
              Indian Jurisprudence · AI Reasoning Engine
            </motion.p>

            {/* Omni-Search Bar */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: "spring", stiffness: 400, damping: 30, delay: 0.18 }}
              style={{ width: "100%", maxWidth: "44rem", position: "relative" }}
            >
              <form onSubmit={handleSubmit} style={{ 
                position: "relative",
                display: "flex", alignItems: "flex-start", // Align items to top to allow natural box growth
                width: "100%", 
                minHeight: "6rem", // Increased from 4.5rem to make the box taller by default
                background: "rgba(7, 7, 10, 0.6)",
                backdropFilter: "blur(20px) saturate(1.5)",
                WebkitBackdropFilter: "blur(20px) saturate(1.5)",
                border: "1px solid var(--color-graphite-border)",
                borderRadius: "1.25rem",
                padding: "1rem 1.5rem",
                boxShadow: "0 0 0 1px rgba(229,193,88,0.0), 0 24px 48px -12px rgba(0,0,0,0.8), inset 0 2px 8px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06)",
                transition: "all 0.3s ease"
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = "rgba(229,193,88,0.4)";
                e.currentTarget.style.boxShadow = "0 0 0 1px rgba(229,193,88,0.2), 0 32px 64px -12px rgba(0,0,0,0.9), inset 0 2px 8px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.08)";
                e.currentTarget.style.background = "rgba(10, 10, 14, 0.8)";
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = "var(--color-graphite-border)";
                e.currentTarget.style.boxShadow = "0 0 0 1px rgba(229,193,88,0.0), 0 24px 48px -12px rgba(0,0,0,0.8), inset 0 2px 8px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06)";
                e.currentTarget.style.background = "rgba(7, 7, 10, 0.6)";
              }}
              >
                <SearchIcon style={{ color: "var(--color-gold)", opacity: 0.8, marginRight: "1rem", marginTop: "0.2rem" }} size={22} strokeWidth={1.5} />
                <textarea
                  ref={inputRef}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit();
                    }
                  }}
                  placeholder="Enter a complex legal scenario, or type a precise citation..."
                  style={{
                    flex: 1, 
                    minHeight: "4rem", // Fill more of the box
                    maxHeight: "14rem",
                    background: "transparent", 
                    border: "none", 
                    outline: "none",
                    color: "var(--color-ivory)", 
                    fontSize: "0.9375rem", // Reduced from 1.125rem
                    fontWeight: 300,
                    lineHeight: 1.6, // Add line height for readability
                    fontFamily: "var(--font-sans)",
                    resize: "none",
                    overflowY: "auto",
                    paddingTop: "0.3rem"
                  }}
                  rows={query.split("\n").length > 1 ? Math.min(query.split("\n").length, 6) : 1}
                  className="hide-scrollbar"
                  autoFocus
                />
                <AnimatePresence>
                  {query.length > 0 && (
                    <motion.button
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      type="submit"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      style={{
                        display: "flex", alignItems: "center", justifyContent: "center",
                        width: "2.5rem", height: "2.5rem", borderRadius: "0.75rem",
                        background: "var(--color-gold)", color: "var(--color-void)",
                        fontSize: "1rem", fontWeight: 600, border: "none", cursor: "pointer",
                        boxShadow: "0 4px 12px rgba(229,193,88,0.3)",
                        position: "absolute", bottom: "1rem", right: "1.5rem" // Pin button to bottom right
                      }}
                    >
                      ↵
                    </motion.button>
                  )}
                </AnimatePresence>
                {/* Hidden submit button for programatic trigger */}
                <button id="hidden-submit-btn" type="submit" style={{ display: "none" }} />
              </form>

              {/* A failed submit resets appState to "idle" (see the catch
                  block in handleSubmit), which is the right recovery so the
                  user isn't stuck on a dead loading screen, but statusText
                  used to only ever render inside the active-state UI. The
                  error message was being set and never shown: someone
                  hitting a cold-starting or unreachable backend saw their
                  query just silently do nothing. */}
              <AnimatePresence>
                {appState === "idle" && statusText && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    style={{ marginTop: "1rem", padding: "0.75rem 1.25rem", borderRadius: "0.75rem", background: "rgba(199,88,88,0.08)", border: "1px solid rgba(199,88,88,0.25)", color: "var(--color-parchment)", fontSize: "0.8125rem", textAlign: "center" }}
                  >
                    {statusText}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Premium Suggestion Cards */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ type: "spring", stiffness: 400, damping: 30, delay: 0.35 }}
                style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginTop: "1.5rem" }}
              >
                {[
                  { title: "Writ of Mandamus", sub: "Constitutional requirements", icon: <Landmark size={20} strokeWidth={1.5} className="text-[var(--color-ash)]" /> },
                  { title: "Temporary Injunction", sub: "Order 39 Rule 1 CPC", icon: <Scale size={20} strokeWidth={1.5} className="text-[var(--color-ash)]" /> },
                  { title: "Promissory Estoppel", sub: "Contractual reliance", icon: <ScrollText size={20} strokeWidth={1.5} className="text-[var(--color-ash)]" /> },
                  { title: "Preventive Detention", sub: "Article 22 safeguards", icon: <Shield size={20} strokeWidth={1.5} className="text-[var(--color-ash)]" /> },
                ].map((s) => (
                  <button
                    key={s.title}
                    onClick={() => {
                      const newQ = `${s.title} - ${s.sub}`;
                      setQuery(newQ);
                      // Fire handle submit with the exact string rather than waiting for state
                      const fakeEvent = { preventDefault: () => {} } as React.FormEvent;
                      // We must pass the query explicitly to avoid stale state dependency
                      setQuery((prev) => {
                         // We don't have access to submit inside this block easily without refactoring handleSubmit
                         return newQ;
                      });
                      setTimeout(() => {
                         // Fallback hack since we don't want to rewrite the whole component right now
                         const submitBtn = document.getElementById('hidden-submit-btn');
                         if (submitBtn) submitBtn.click();
                      }, 50);
                    }}
                    style={{
                      display: "flex", alignItems: "center", gap: "1rem", padding: "1rem 1.25rem",
                      background: "rgba(20,20,25,0.4)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
                      border: "1px solid var(--color-graphite-border)", borderRadius: "1rem",
                      cursor: "pointer", transition: "all 0.25s ease", textAlign: "left",
                      boxShadow: "inset 0 1px 0 rgba(255,255,255,0.02)"
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = "rgba(30,30,38,0.6)";
                      e.currentTarget.style.borderColor = "rgba(229,193,88,0.25)";
                      e.currentTarget.style.transform = "translateY(-2px)";
                      e.currentTarget.style.boxShadow = "0 8px 24px -8px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = "rgba(20,20,25,0.4)";
                      e.currentTarget.style.borderColor = "var(--color-graphite-border)";
                      e.currentTarget.style.transform = "translateY(0)";
                      e.currentTarget.style.boxShadow = "inset 0 1px 0 rgba(255,255,255,0.02)";
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", width: "2.5rem", height: "2.5rem", borderRadius: "0.5rem", background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)" }}>
                      {s.icon}
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.2rem" }}>
                      <span style={{ fontSize: "0.875rem", fontWeight: 500, color: "var(--color-ivory)" }}>{s.title}</span>
                      <span style={{ fontSize: "0.6875rem", color: "var(--color-ash)", fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.05em" }}>{s.sub}</span>
                    </div>
                  </button>
                ))}
              </motion.div>

              {/* Instructions / Dataset Info */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                style={{ marginTop: "2.5rem", padding: "1.25rem 1.5rem", background: "rgba(20,20,25,0.4)", border: "1px solid var(--color-graphite-border)", borderRadius: "1rem", textAlign: "left" }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
                  <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: "var(--color-gold)", boxShadow: "0 0 8px var(--color-gold)" }} />
                  <span style={{ fontSize: "0.6875rem", letterSpacing: "0.15em", textTransform: "uppercase", color: "var(--color-ivory)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>Engine Capabilities</span>
                </div>
                <p style={{ fontSize: "0.875rem", color: "var(--color-parchment)", lineHeight: 1.7, marginBottom: "0.75rem" }}>
                  Ratio is strictly grounded in the <strong>OpenNyAI InJudgements</strong> dataset, indexing thousands of Supreme Court and High Court cases. It uses a bespoke <strong>Deterministic Query Router</strong> and <strong>Hybrid BM25 Pipeline</strong>.
                </p>
                <p style={{ fontSize: "0.875rem", color: "var(--color-fog)", lineHeight: 1.7 }}>
                  <strong style={{ color: "var(--color-ivory)" }}>How to test:</strong> You can paste exact legal citations (e.g., <code style={{ color: "var(--color-gold)", background: "rgba(229,193,88,0.1)", padding: "0.1rem 0.3rem", borderRadius: "0.2rem" }}>AIR 1974 SC 224</code>) to see the deterministic router preserve exact capitalization, or ask complex doctrinal questions (e.g., <em style={{ color: "var(--color-ivory)" }}>&quot;Does promissory estoppel apply against the State?&quot;</em>) to trigger the multi-hop dense semantic search.
                </p>
              </motion.div>

              <p style={{ marginTop: "1.25rem", fontSize: "0.75rem", color: "var(--color-ash)", lineHeight: 1.6, textAlign: "center" }}>
                Ratio is a research and portfolio project, not a lawyer. Answers are generated from a fixed corpus and can be wrong, incomplete, or out of date. Nothing here is legal advice; verify anything that matters against the primary source.
              </p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── ACTIVE STATE: Split workspace ── */}
      <AnimatePresence>
        {isActive && (
          <motion.div
            key="workspace"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            className="research-workspace" style={{ display: "flex", width: "100%", height: "100%" }}
          >
            {/* LEFT: Answer Canvas */}
            <div className="research-answer-col" style={{ display: "flex", flexDirection: "column", width: "55%", minWidth: 0, flexShrink: 0, height: "100%", borderRight: "1px solid var(--color-graphite-border)" }}>
              {/* Top bar */}
              <div style={{ display: "flex", alignItems: "center", gap: "1rem", padding: "1.25rem 2rem", borderBottom: "1px solid var(--color-graphite-border)" }}>
                <RatioWordmark size="sm" />
                <div style={{ height: "20px", width: "1px", background: "var(--color-graphite-border)" }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ color: "var(--color-ash)", fontSize: "0.7rem", textTransform: "uppercase", fontFamily: "var(--font-mono)", letterSpacing: "0.15em" }}>
                    Research Session
                  </p>
                </div>
                <button
                  onClick={() => {
                    setAppState("idle");
                    setQuery("");
                    setAnswer("");
                    setSources([]);
                  }}
                  style={{ flexShrink: 0, color: "var(--color-ash)", fontSize: "0.75rem", border: "1px solid var(--color-graphite-border)", padding: "0.375rem 0.75rem", borderRadius: "0.5rem", cursor: "pointer", background: "transparent", transition: "all 0.2s ease" }}
                  onMouseEnter={(e) => e.currentTarget.style.color = "var(--color-fog)"}
                  onMouseLeave={(e) => e.currentTarget.style.color = "var(--color-ash)"}
                >
                  New query
                </button>
              </div>

              {/* Scrollable answer area */}
              <div className="hide-scrollbar" style={{ flex: 1, overflowY: "auto", padding: "1.75rem 2rem" }}>
                <AnimatePresence>
                  {(appState === "searching" || (appState === "complete" && !answer)) && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0, overflow: "hidden" }}
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                    >
                      <AgentTerminal thoughts={thoughts} />
                    </motion.div>
                  )}
                </AnimatePresence>
                <AnswerCanvas
                  answer={answer}
                  query={submittedQuery}
                  sources={sources}
                  status={statusText}
                  refused={refused}
                  isLoading={appState === "searching"}
                />
                {appState === "complete" && answer && !refused && (
                  <p style={{ marginTop: "1.5rem", fontSize: "0.6875rem", color: "var(--color-ash)", lineHeight: 1.6, borderTop: "1px solid var(--color-graphite-border)", paddingTop: "1rem" }}>
                    Not legal advice. Generated from a fixed corpus and can be wrong or incomplete; verify against the cited sources before relying on it.
                  </p>
                )}
              </div>\n              {/* Search bar (persistent, compact) */}
              <div style={{ padding: "1rem 2rem 1.5rem", borderTop: "1px solid var(--color-graphite-border)", background: "var(--color-void)" }}>
                <form onSubmit={handleSubmit} style={{ 
                  display: "flex", alignItems: "center", width: "100%", padding: "0.75rem 1rem", borderRadius: "0.625rem",
                  background: "rgba(7, 7, 10, 0.6)", border: "1px solid var(--color-graphite-border)",
                  boxShadow: "inset 0 1px 0 rgba(255,255,255,0.03)", transition: "border-color 0.2s ease"
                }}
                onFocus={(e) => e.currentTarget.style.borderColor = "rgba(229,193,88,0.4)"}
                onBlur={(e) => e.currentTarget.style.borderColor = "var(--color-graphite-border)"}
                >
                  <SearchIcon style={{ color: "var(--color-ash)", flexShrink: 0, marginRight: "0.75rem" }} size={15} strokeWidth={1.5} />
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Refine or ask a follow-up..."
                    style={{ flex: 1, background: "transparent", border: "none", outline: "none", color: "var(--color-ivory)", fontSize: "0.875rem", fontFamily: "var(--font-sans)" }}
                  />
                </form>
                {/* Facet Filters */}
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.75rem", padding: "0 0.25rem" }}>
                  <button className="facet-chip active">
                    <span style={{ width: "6px", height: "6px", borderRadius: "50%", backgroundColor: "var(--color-gold)" }}></span>
                    All Law
                  </button>
                  <button className="facet-chip">Supreme Court</button>
                  <button className="facet-chip">Patna HC</button>
                  <button className="facet-chip">Statutes</button>
                </div>
              </div>


            </div>

                          {/* RIGHT: Evidence Panel */}
            <motion.div
              initial={{ x: 60, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ type: "spring", stiffness: 400, damping: 30, delay: 0.1 }}
              className="research-evidence-col" style={{ display: "flex", flexDirection: "column", width: "45%", minWidth: 0, flexShrink: 0, height: "100%", background: "var(--color-obsidian)" }}
            >
              <EvidencePanel
                sources={sources}
                isLoading={appState === "searching"}
                sourceCount={sources.length}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
