#!/bin/bash
# Automatically resume summary generation if it hits API rate limits and crashes.

while true; do
  LLM_PROVIDER=ifm PYTHONPATH=src .venv/bin/python scripts/generate_summaries.py --limit 10588
  EXIT_CODE=$?
  
  if [ $EXIT_CODE -eq 0 ]; then
    echo "Finished generating all summaries successfully!"
    break
  fi
  
  echo "Script crashed with exit code $EXIT_CODE (likely API rate limit)."
  echo "Restarting in 60 seconds to resume progress..."
  sleep 60
done
