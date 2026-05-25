Auto-continue: intent and design constraints
=============================================

Problem
-------
The local model (Qwen via vLLM on RunPod) sometimes produces incomplete responses:
  - Tool calls only, no text
  - Empty responses (reasoning only, no content)
  - Truncated mid-sentence

In these cases the session goes idle but the task isn't done. The user would need
to manually type "continue" — which breaks flow.

Goal
----
Automatically detect incomplete responses and retry, so the model can finish its
work without user intervention.

Constraints
-----------
1. Must NOT auto-continue when the response is genuinely complete (has real text,
   doesn't end with a question, last tool wasn't a completion tool like "question").
2. Must break cycles: if the model keeps producing empty responses to auto-continue,
   stop after a few retries.
3. Must NOT interfere with user input: if the user sends a message while the watcher
   is about to auto-continue, skip it.
4. Must NOT hang: any network call, SSE read, or HTTP request must have a timeout.
5. Must be observable: log every decision (continue / skip / cycle-break) with reason.

Current implementation (minimal)
--------------------------------
The watcher is a passive observer. It connects to opencode's /global/event SSE
endpoint and logs session.status transitions (busy/idle/retry). That's it.

When re-adding auto-continue, the decision logic should:
  - Read the last assistant message from the opencode API
  - Check for incomplete response patterns (empty, tool-only, near-empty tokens)
  - Send a continue message to the session
  - Track retries per session to break cycles

What NOT to do
--------------
- Do NOT abort the session before sending continue — this was suspected of leaving
  opencode's Effect runtime in a dirty state (stale memoMap entries, leaked fibers,
  held semaphore locks) that caused the next tool call to hang forever.
- Do NOT add hang detection that kills busy sessions — a busy session is generating
  tokens or running tools; killing it mid-stream causes the same state corruption.
- Do NOT add loop detection, leak repair, or thinking tag repair — these are proxy
  concerns, not watcher concerns.
- Do NOT add reasoning dumps, stream traces, or policy injection — these are
  debugging features, not core functionality.

Re-adding auto-continue
-----------------------
1. Start with the current passive watcher (just logging)
2. Add should_continue() logic (read messages, check patterns)
3. Add send_continue() (just send_message, NO abort first)
4. Test for a day, verify no hangs
5. If stable, add cycle-breaking (retry counter, max retries)
6. If stable, add user-input protection (skip if session goes busy again)
