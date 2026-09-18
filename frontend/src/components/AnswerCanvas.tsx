"use client";

import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, Edit3, Send, CheckCircle2 } from "lucide-react";
import { useState, useEffect } from "react";

interface Props {
  answer: string;
  query: string;
  sources?: any[];
  status: string;
  refused: boolean;
  isLoading: boolean;
}

function CitationBadge({ num, sources }: { num: string, sources?: any[] }) {
  const [isHovered, setIsHovered] = useState(false);
  const sourceIndex = parseInt(num) - 1;
  const source = sources?.[sourceIndex];

  return (
    <span 
      className="relative inline-block"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <span className="citation-badge">
        {num}
      </span>
      <AnimatePresence>
        {isHovered && source && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 5, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 500, damping: 25 }}
            className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 z-50 pointer-events-none"
          >
            <div className="apple-blur rounded-xl p-4 flex flex-col gap-2 border border-white/10 shadow-2xl">
              <div className="text-[0.65rem] uppercase tracking-widest text-[var(--color-gold)] font-bold">
                Source {num} · {source.court || "Unknown Court"}
              </div>
              <p className="text-[0.85rem] text-[var(--color-ivory)] leading-relaxed font-sans line-clamp-3">
                {source.title || "Untitled Document"}
              </p>
              <div className="text-xs text-[var(--color-ash)] font-mono mt-1 pt-2 border-t border-white/5 flex items-center justify-between">
                <span>{source.stance ? `stance: ${source.stance}` : ""}</span>
                {typeof source.cited_by === "number" && <span>cited by {source.cited_by}</span>}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </span>
  );
}

// Render answer text with interactive citation badges
function renderAnswer(text: string, sources?: any[]) {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      return (
        <CitationBadge key={i} num={match[1]} sources={sources} />
      );
    }
    // Process markdown bold **text** inline
    const boldParts = part.split(/(\*\*[^*]+\*\*)/g);
    return (
      <span key={i}>
        {boldParts.map((bp, j) => {
          if (bp.startsWith("**") && bp.endsWith("**")) {
            return (
              <strong key={j} className="font-semibold text-[var(--color-ivory)]">
                {bp.slice(2, -2)}
              </strong>
            );
          }
          return bp;
        })}
      </span>
    );
  });
}

function renderAnswerWithThinking(text: string, sources?: any[]) {
  if (!text.includes("<think>")) return <span className="whitespace-pre-wrap break-words">{renderAnswer(text, sources)}</span>;

  const parts = [];
  let currentText = text;
  
  while (currentText.includes("<think>")) {
    const startIdx = currentText.indexOf("<think>");
    if (startIdx > 0) {
      parts.push({ type: 'text', content: currentText.slice(0, startIdx) });
    }
    
    const endIdx = currentText.indexOf("</think>", startIdx);
    if (endIdx !== -1) {
      parts.push({ type: 'think', content: currentText.slice(startIdx + 7, endIdx) });
      currentText = currentText.slice(endIdx + 8);
    } else {
      parts.push({ type: 'think', content: currentText.slice(startIdx + 7), isStreaming: true });
      currentText = "";
      break;
    }
  }
  
  if (currentText.length > 0) {
    parts.push({ type: 'text', content: currentText });
  }

  return parts.map((part, i) => {
    if (part.type === 'think') {
      return (
        <details key={i} className="group my-5 rounded-xl bg-[#0F0F13] border border-[var(--color-graphite-border)] shadow-inner overflow-hidden">
          <summary className="flex items-center gap-2 p-3 cursor-pointer outline-none hover:bg-white/5 transition-colors list-none [&::-webkit-details-marker]:hidden">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-ash)] opacity-70 group-open:bg-[var(--color-gold)] transition-colors" />
            <span className="text-[0.625rem] uppercase tracking-[0.2em] text-[var(--color-ash)] font-bold opacity-70 group-open:text-[var(--color-gold)] transition-colors select-none">
              Engine Reasoning
            </span>
            {part.isStreaming && (
               <motion.span
                 className="inline-block w-1 h-4 bg-[var(--color-gold)] opacity-50 ml-1"
                 animate={{ opacity: [0.2, 0.7, 0.2] }}
                 transition={{ duration: 1.2, repeat: Infinity }}
               />
            )}
            <div className="ml-auto">
              <svg className="w-3.5 h-3.5 text-[var(--color-ash)] opacity-50 transition-transform group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </summary>
          <div className="p-4 pt-1 border-t border-white/5">
            <p className="text-[0.875rem] text-[var(--color-fog)] leading-relaxed italic opacity-80 whitespace-pre-wrap break-words">
              {part.content.trim()}
            </p>
          </div>
        </details>
      );
    } else {
      return <span key={i} className="whitespace-pre-wrap break-words">{renderAnswer(part.content, sources)}</span>;
    }
  });
}

// Processing animation — shows while retrieving
function ReasoningIndicator({ status }: { status: string }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-col gap-6 py-4"
    >
      {/* Animated line segments — visualising retrieval */}
      <div className="flex flex-col gap-2.5">
        {[1, 0.7, 0.85, 0.5].map((w, i) => (
          <motion.div
            key={i}
            className="h-[3px] rounded-full bg-[var(--color-graphite-mid)]"
            style={{ width: `${w * 100}%` }}
            animate={{ opacity: [0.3, 0.7, 0.3] }}
            transition={{ duration: 1.8, repeat: Infinity, delay: i * 0.18, ease: "easeInOut" }}
          />
        ))}
      </div>

      {/* Status text */}
      {status && (
        <motion.p
          key={status}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-xs text-[var(--color-ash)] font-mono tracking-wider uppercase"
        >
          {status}
        </motion.p>
      )}

      {/* Skeleton answer lines */}
      <div className="flex flex-col gap-3 mt-2">
        {[0.95, 0.88, 0.72, 0.9, 0.65].map((w, i) => (
          <motion.div
            key={i}
            className="h-3.5 rounded-md bg-[var(--color-graphite-mid)]"
            style={{ width: `${w * 100}%` }}
            animate={{ opacity: [0.2, 0.4, 0.2] }}
            transition={{ duration: 2.2, repeat: Infinity, delay: i * 0.12 }}
          />
        ))}
      </div>
    </motion.div>
  );
}

export function AnswerCanvas({ answer, query, sources, status, refused, isLoading }: Props) {
  const [isEditing, setIsEditing] = useState(false);
  const [editedAnswer, setEditedAnswer] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);

  // Verification console log when answer completes
  useEffect(() => {
    if (!isLoading && answer && !refused) {
      console.log("=========================================");
      console.log("RAG Verification Report");
      console.log("=========================================");
      console.log("Query:", query);
      console.log("Length (chars):", answer.length);
      console.log("Contains Citations:", /\[\d+\]/.test(answer) ? "Yes ✅" : "No ❌");
      console.log("Sources Used:", sources?.length ?? 0);
      console.log("World Class Quality Check:");
      if (answer.length > 500 && /\[\d+\]/.test(answer)) {
        console.log("  Status: EXCELLENT 🌟 (Detailed and grounded)");
      } else if (answer.length > 100) {
        console.log("  Status: ADEQUATE 👍 (Could be more detailed)");
      } else {
        console.log("  Status: POOR ⚠️ (Too brief)");
      }
      console.log("=========================================");
    }
  }, [isLoading, answer, refused, query, sources]);

  useEffect(() => {
    if (answer) {
      setEditedAnswer(answer);
      setSubmitSuccess(false);
      setIsEditing(false);
    }
  }, [answer]);

  const handleSubmitCorrection = async () => {
    if (!editedAnswer.trim() || editedAnswer === answer) {
      setIsEditing(false);
      return;
    }
    
    setIsSubmitting(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/submit_correction", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          original_answer: answer,
          corrected_answer: editedAnswer,
        }),
      });
      if (response.ok) {
        setSubmitSuccess(true);
        setTimeout(() => setSubmitSuccess(false), 3000);
      }
    } catch (e) {
      console.error("Failed to submit correction", e);
    } finally {
      setIsSubmitting(false);
      setIsEditing(false);
    }
  };

  return (
    <div className="relative">
      <AnimatePresence mode="wait">
        {isLoading && !answer && (
          <ReasoningIndicator key="loading" status={status} />
        )}
      </AnimatePresence>

      {/* Refused state */}
      {refused && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-start gap-4 p-5 rounded-xl border border-amber-900/30 bg-amber-950/10 mb-6"
        >
          <AlertTriangle className="text-amber-600 shrink-0 mt-0.5" size={16} strokeWidth={1.5} />
          <div>
            <p className="text-amber-500/90 text-sm font-medium mb-1">Insufficient source material</p>
            <p className="text-amber-700/70 text-xs leading-relaxed">
              The retrieved passages do not contain enough grounded information to form a reliable answer. Try rephrasing or narrowing your query.
            </p>
          </div>
        </motion.div>
      )}

      {/* Error / Generation Failed state */}
      {!isLoading && !answer && !refused && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-start gap-4 p-5 rounded-xl border border-red-900/30 bg-red-950/10 mb-6"
        >
          <AlertTriangle className="text-red-600 shrink-0 mt-0.5" size={16} strokeWidth={1.5} />
          <div>
            <p className="text-red-500/90 text-sm font-medium mb-1">Generation Failed</p>
            <p className="text-red-700/70 text-xs leading-relaxed">
              {status.includes("Error") ? status : "The agent stopped unexpectedly without returning a final answer. Please try again or simplify your query."}
            </p>
          </div>
        </motion.div>
      )}

      {/* User Query Bubble */}
      {query && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex justify-end mb-8"
        >
          <div className="bg-[var(--color-graphite-deep)] border border-[var(--color-graphite-border)] text-[var(--color-ivory)] px-5 py-3.5 rounded-2xl max-w-[85%] text-[0.95rem] leading-relaxed shadow-sm">
            {query}
          </div>
        </motion.div>
      )}

      {/* Assistant Answer Bubble */}
      {answer && !isEditing && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ type: "spring", stiffness: 400, damping: 30 }}
          className="flex gap-4"
        >
          <div className="w-8 h-8 rounded-full bg-[rgba(229,193,88,0.15)] border border-[var(--color-gold)]/30 flex items-center justify-center shrink-0 mt-1 shadow-[0_0_12px_rgba(229,193,88,0.1)]">
            <span className="text-[var(--color-gold)] font-bold font-serif text-sm">R</span>
          </div>
          <div className="answer-prose text-[var(--color-parchment)] text-[0.975rem] leading-[1.9] flex-1">
            <div>{renderAnswerWithThinking(answer, sources)}</div>

          {/* Cursor blink while still loading */}
          {isLoading && (
            <motion.span
              className="inline-block w-[2px] h-4 bg-[var(--color-gold)] ml-0.5 align-middle"
              animate={{ opacity: [1, 0] }}
              transition={{ duration: 0.7, repeat: Infinity }}
            />
          )}

          {/* HITL Edit Button */}
          {!isLoading && !refused && (
            <div className="mt-8 pt-4 border-t border-[var(--color-graphite-border)] flex items-center justify-between">
              <span className="text-xs text-[var(--color-ash)] opacity-70">See a hallucination or poor citation?</span>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setIsEditing(true)}
                className="flex items-center gap-2 text-[var(--color-ivory)] bg-[var(--color-obsidian)] border border-[var(--color-graphite-border)] px-4 py-2 rounded-md text-sm font-medium hover:bg-white/5 transition-colors"
              >
                <Edit3 size={14} /> Correct Answer
              </motion.button>
            </div>
          )}
          
          {submitSuccess && (
             <motion.div 
               initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} 
               className="mt-4 flex items-center gap-2 text-green-400 text-sm bg-green-950/30 p-3 rounded-lg border border-green-900/50"
             >
               <CheckCircle2 size={16} /> Saved to DPO Training Dataset
             </motion.div>
          )}
          </div>
        </motion.div>
      )}

      {/* Editing State */}
      {isEditing && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-[var(--color-ivory)] font-medium text-sm">Human-in-the-Loop DPO Feedback</h3>
            <button onClick={() => setIsEditing(false)} className="text-[var(--color-ash)] hover:text-white text-sm">Cancel</button>
          </div>
          <textarea
            value={editedAnswer}
            onChange={(e) => setEditedAnswer(e.target.value)}
            className="w-full bg-[var(--color-obsidian)] border border-[var(--color-gold)]/40 rounded-xl p-4 text-[var(--color-parchment)] text-[0.975rem] leading-[1.9] min-h-[400px] outline-none focus:border-[var(--color-gold)] transition-colors resize-y"
          />
          <div className="flex justify-end">
            <button
              onClick={handleSubmitCorrection}
              disabled={isSubmitting}
              className="flex items-center gap-2 bg-[var(--color-gold)] text-black px-6 py-2.5 rounded-lg font-bold text-sm hover:opacity-90 transition-opacity disabled:opacity-50"
            >
              {isSubmitting ? "Saving..." : <><Send size={14} /> Submit to Training Data</>}
            </button>
          </div>
        </motion.div>
      )}


    </div>
  );
}
