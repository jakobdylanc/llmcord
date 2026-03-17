---
description: Debugging and issue resolution with non-invasive instrumentation and cleanup
applyTo: "**/*"
---

Systematically debug issues, isolate root causes, and implement surgical fixes.

**Input**: A bug description, stack trace, or failing test. If the objective is vague, you MUST use the **AskUserQuestion tool** to clarify expected vs. actual behavior.

## Tool Preferences

**Use:**
- `grep_search` - Find error messages or relevant code patterns
- `get_errors` - Check for compile/lint errors
- `read_file` - Examine source files
- `run_in_terminal` - Run tests or the application
- `mcp_pylance_mcp_s_pylanceFileSyntaxErrors` - Validate Python syntax

**Steps**

1. **Analyze and Hypothesize**
   - Use `read_file` and `grep_search` to build a mental model of the relevant code.
   - Formulate a hypothesis for the root cause.
   - **Do not** modify production logic yet.
   - State: "Hypothesis: <description of the suspected bug>"

2. **Instrument for Evidence**
   - Propose specific print or logging statements to validate the hypothesis.
   - Use language-appropriate calls (e.g., `print()`, `console.log`).
   - **Requirement**: Every debug line MUST be tagged with `// DEBUG: TEMP`.
   - Apply changes and use `run_in_terminal` to execute the reproduction steps.

3. **Verify the Failure**
   - Parse the output from the terminal.
   - Use `get_errors` or `mcp_pylance_mcp_s_pylanceFileSyntaxErrors` to ensure instrumentation didn't break anything.
   - If hypothesis is refuted, revert changes and return to Step 1.
   - If confirmed, proceed to planning the fix.

4. **Implement Surgical Fix**
   - Design the most minimal change required to resolve the issue.
   - **Check Call Sites**: Use `grep_search` to ensure the fix does not break existing public APIs.
   - If a breaking change is necessary, you MUST warn: "⚠️ BREAKING CHANGE REQUIRED".

5. **Validation and Cleanup**
   - After the fix, run `run_in_terminal` to verify the solution.
   - Use `get_errors` to confirm no new lint/compile issues were introduced.
   - **Automatic Action**: Remove all lines containing the `// DEBUG: TEMP` tag.
   - Verify that the codebase returns to a production-ready state.

6. **Final Report**
   - Summarize the Root Cause and the Fix.
   - Confirm all temporary debug artifacts are purged.

**Output During Implementation**

```
## 🔍 Debugging: <issue-name>

**Hypothesis:** <brief theory>
**Instrumentation:** Using `grep_search` to find relevant patterns and adding temporary logs to `[file.ext]`.
[...applying logs with // DEBUG: TEMP...]
✓ Instrumentation applied. Please run the code.
```

**Output On Completion**

```
## ✅ Issue Resolved

**Root Cause:** <description>
**Fix:** <minimal change description>
**Cleanup:** Purged all // DEBUG: TEMP lines. ✓

The bug is fixed and the codebase is clean.
```

**Guardrails**
- **Evidence First**: Never change logic before seeing proof via logs or tests.
- **Tool-Driven**: Always use `get_errors` or `mcp_pylance_mcp_s_pylanceFileSyntaxErrors` after modifying files.
- **Tagging**: Always use `// DEBUG: TEMP` for temporary code to ensure 100% removal.
- **Scope**: Keep changes minimal and focused. Avoid unrelated refactoring.
- **API Integrity**: Prioritize non-breaking changes to maintain system stability.
- **Verification**: Use `run_in_terminal` to confirm the fix before final cleanup.