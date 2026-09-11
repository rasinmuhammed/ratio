"use client";

import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle } from "lucide-react";

interface Props {
  answer: string;
  status: string;
  refused: boolean;
  isLoading: boolean;
}

// Render answer text with interactive citation badges
function renderAnswer(text: string) {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      return (
        <span key={i} className="citation-badge">
          {match[1]}
        </span>
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

function renderAnswerWithThinking(text: string) {
  if (!text.includes("<think>")) return <span className="whitespace-pre-wrap break-words">{renderAnswer(text)}</span>;

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
      return <span key={i} className="whitespace-pre-wrap break-words">{renderAnswer(part.content)}</span>;
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

export function AnswerCanvas({ answer, status, refused, isLoading }: Props) {
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

      {/* Streaming Answer */}
      {answer && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="answer-prose text-[var(--color-parchment)] text-[0.975rem] leading-[1.9]"
        >
          <div>{renderAnswerWithThinking(answer)}</div>

          {/* Cursor blink while still loading */}
          {isLoading && (
            <motion.span
              className="inline-block w-[2px] h-4 bg-[var(--color-gold)] ml-0.5 align-middle"
              animate={{ opacity: [1, 0] }}
              transition={{ duration: 0.7, repeat: Infinity }}
            />
          )}
        </motion.div>
      )}
    </div>
  );
}
