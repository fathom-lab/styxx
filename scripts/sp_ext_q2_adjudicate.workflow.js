export const meta = {
  name: 'sp-ext-q2-adjudicate',
  description: 'Adjudicate the frozen 40-candidate Q2 sample; 3 reject-lenses each',
  phases: [{ title: 'Reject' }],
}

const ROOT = 'C:/Users/heyzo/AppData/Local/Temp/spfull'

const VERDICT = {
  type: 'object',
  properties: {
    id: { type: 'integer' },
    reject: { type: 'boolean', description: 'true = NOT a silent-pass case. Default true.' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    which_requirement_fails: { type: 'string' },
    prefix_behaviour: { type: 'string', description: 'One sentence quoting the actual pre-fix line.' },
    postfix_behaviour: { type: 'string' },
    subtype: { type: 'string', description: 'SP-1..SP-8 if accepting, else empty' },
    consumer: { type: 'string', description: 'Who reads the value and could be misled, or "did not verify"' },
    module_path: { type: 'string', description: 'EXACT path from git show --name-only. Never inferred.' },
    defect_line: { type: 'integer', description: 'Line number of the flattering return in <sha>~1 source, 0 if not applicable' },
    rationale: { type: 'string' },
  },
  required: ['id','reject','confidence','which_requirement_fails','prefix_behaviour','postfix_behaviour','subtype','consumer','module_path','defect_line','rationale'],
}

const C = [
  { id: 1, repo: "deepeval", sha: "0bb7105c1814", subj: "feat(optimization): add prompt optimizer and refactor GEPA runner with status callbacks and docs", rem: "- return 0", add: "+ if self.status_callback is not None:" },
  { id: 2, repo: "pandera", sha: "c9e7a6e2f974", subj: "Refactor schema backends (#1144)", rem: "- return True", add: "+ failure_cases = None" },
  { id: 3, repo: "inspect_ai", sha: "acd139cc75aa", subj: "fix(web_browser): warn once when AppKit is missing on the macOS scale-factor probe", rem: "- return 1.0", add: "+def _warn_missing_appkit() -> None:" },
  { id: 4, repo: "whylogs", sha: "fab54d3c6926", subj: "Use time-based scheduler and validate writers (#50)", rem: "- return True", add: "+ def check_writer(self, writer: Writer) -> None:" },
  { id: 5, repo: "inspect_ai", sha: "a9a9ec00169c", subj: "Record dead sandboxes as sandbox_unavailable tool errors", rem: "- if (e.length === 0) return 0;", add: "+ if ((opts == null ? void 0 : opts.onChange) && !(isInitial && opts.skipInitialOnChange))" },
  { id: 6, repo: "inspect_ai", sha: "ed34d0492812", subj: "Round-3 review fixes", rem: "- return True", add: "+ # compare reducers, treating the unrecorded default (None) and an" },
  { id: 7, repo: "great_expectations", sha: "0412d5e87438", subj: "All expectations (except table-level expectations) converted to use new decorator syntax.", rem: "- # return True", add: "+ raise ValueError(\"Unable to use provided format. \" + e.message)" },
  { id: 8, repo: "inspect_ai", sha: "e3f45aa2e1c2", subj: "Convert Bedrock to Converse API (#853)", rem: "- return True", add: "+ tool_config = None" },
  { id: 9, repo: "lm-evaluation-harness", sha: "9e98d1a91a0a", subj: "removed test lines", rem: "- return 0", add: "+# task_name: (group_name, None)," },
  { id: 10, repo: "inspect_ai", sha: "34beafda81d3", subj: "Prevent math scorer from executing model output (#4361)", rem: "- return True", add: "+ expression: Any | None" },
  { id: 11, repo: "great_expectations", sha: "fb1fbb032439", subj: "[MAINTENANCE] Cloud integration tests documentation (#8871)", rem: "- return True", add: "+ pytest.skip(\"no snowflake credentials\")" },
  { id: 12, repo: "garak", sha: "d27e4eefed90", subj: "addressed additional reviewer feedback, cleaned up docs/code, fixed detector edge cases", rem: "- assert result == [0.0], \"Should return 0.0 when no system prompt to match\"", add: "+def _make_attempt(system_prompt=SYSTEM_PROMPT, output_text=None):" },
  { id: 13, repo: "trulens", sha: "1592a6181dee", subj: "work", rem: "- return True", add: "+logger.warning(\"`trulens_eval.tru_db` is deprecated, use `trulens_eval.db` instead.\")" },
  { id: 14, repo: "great_expectations", sha: "e61aa581f5b9", subj: "working on storing unit tests in list of dictionaries, partially finished", rem: "- # return True,exceptions", add: "+# def expect_column_values_to_be_of_type(self, column, type_, target_datasource, mostly=N" },
  { id: 15, repo: "great_expectations", sha: "cf5cc4086db0", subj: "[MAINTENANCE] split data context files (#4879)", rem: "- return True", add: "+ def _get_metric_configuration_tuples(metric_configuration, base_kwargs=None):" },
  { id: 16, repo: "deepeval", sha: "b1dec13531f7", subj: "feat(config,utils)!: introduce Pydantic-based Settings and unified dotenv autoload", rem: "- return True", add: "+def should_skip_on_missing_params() -> bool:" },
  { id: 17, repo: "great_expectations", sha: "5d39175438ea", subj: "[BUGFIX] Check for datetime-parseable strings in validate_metric_value_between_configuration (#2419)", rem: "- return True", add: "+ min_val is not None or max_val is not None" },
  { id: 18, repo: "inspect_ai", sha: "d980d3f4b391", subj: "fix counts on reused logs; test rework", rem: "- return True", add: "+ assert evals is not None, \"both tasks never reached 'in flight' together\"" },
  { id: 19, repo: "trulens", sha: "a8d473e40450", subj: "instrumentation notebook updates and fixes (#953)", rem: "- return True", add: "+ Will raise an error if accessed in some dynamic way. Accesses that are" },
  { id: 20, repo: "lm-evaluation-harness", sha: "55ea2888491f", subj: "initialize_tasks returns list of tasks and groups", rem: "- return 0", add: "+def initialize_tasks(verbosity=\"INFO\", include_path=None):" },
  { id: 21, repo: "great_expectations", sha: "ec9ca47f058e", subj: "[MAINTENANCE] Add `AddedDiagnostics` helper class to `is_added` flows (#10249)", rem: "- return True, []", add: "+ diagnostics.raise_for_error()" },
  { id: 22, repo: "inspect_ai", sha: "a0bb23201fda", subj: "improved metrics value_to_float string conversion (#196)", rem: "- and return 0.", add: "+ give a warning and return 0." },
  { id: 23, repo: "ragas", sha: "5df03eeebef8", subj: "Fix/tool call accuracy (#2300)", rem: "- return 0.0", add: "+ warnings.warn(\"No tool calls found in the user input\")" },
  { id: 24, repo: "deepeval", sha: "f2538f046cab", subj: "Rewrote MIPROv2", rem: "- return True", add: "+ ) -> None:" },
  { id: 25, repo: "great_expectations", sha: "5bdef2f03e7a", subj: "Convert a bunch more expectations Add expect_column_value_lengths_to_equal Add parameters to validate", rem: "- return True", add: "+ # #!!! This depends on the definition of null. Should we include np.nan in the definiti" },
  { id: 26, repo: "inspect_ai", sha: "f4531b5935c3", subj: "discover compose.yaml for filesystem tasks w/ chdir=True (#294)", rem: "- return True", add: "+def find_compose_file(parent: str = \"\") -> str | None:" },
  { id: 27, repo: "inspect_ai", sha: "d56ece571255", subj: "compaction: improve error messages/logging for improved visibility when native compaction fails", rem: "- return True", add: "+ return _compaction_from_message(message) is not None" },
  { id: 28, repo: "lm-evaluation-harness", sha: "0f9c16247082", subj: "Refactor and implement SAT evaluation", rem: "- return True", add: "+ raise NotImplementedError('SAT Analogies dataset is not provided. Follow instructions o" },
  { id: 29, repo: "lm-evaluation-harness", sha: "e3fee7ea811f", subj: "nit", rem: "- return True", add: "+ return self.config.training_split is not None" },
  { id: 30, repo: "ragas", sha: "4cb829df9855", subj: "chore: remove deprecated functions (#2412)", rem: "- \" return 0\\n\",", add: "+ \" def init(self, run_config=None):\\n\"," },
  { id: 31, repo: "great_expectations", sha: "ff0f0726a447", subj: "[MAINTENANCE] Add datasources to ConfigurationBundle (#6092)", rem: "- return True", add: "+ anonymous_usage_stats_is_none: Set usage stats to None, overrides anonymous_usage_stats" },
  { id: 32, repo: "deepeval", sha: "9d4ab7a5d2d8", subj: ".", rem: "- return \"ok\"", add: "+ return None" },
  { id: 33, repo: "inspect_ai", sha: "9c413b27da00", subj: "Use FastAPI for viewer", rem: "- return True", add: "+ auth_callback=auth_callback if authorization else None," },
  { id: 34, repo: "inspect_ai", sha: "efd04a8dade7", subj: "run_multiple refactor", rem: "- return True", add: "+ async def next() -> list[TaskRunOptions] | None:" },
  { id: 35, repo: "deepeval", sha: "9b14e3278726", subj: "copro updated", rem: "- return True", add: "+ random_state: Optional[Union[int, random.Random]] = None," },
  { id: 36, repo: "inspect_ai", sha: "630081cb32b8", subj: "compaction: improve error messages/logging for improved visibility when native compaction fails (#3480)", rem: "- return True", add: "+ return _compaction_from_message(message) is not None" },
  { id: 37, repo: "deepeval", sha: "85c90b195f45", subj: "Update gateway provider models to use standardized base class", rem: "- return True", add: "+ if temperature is not None:" },
  { id: 38, repo: "giskard", sha: "dd75e974ee76", subj: "fix(checks)!: return None from SuiteResult.pass_rate when nothing evaluated (#2753)", rem: "- return 1.0", add: "+ def pass_rate(self) -> float | None:" },
  { id: 39, repo: "lm-evaluation-harness", sha: "fea3d34fb849", subj: "fix metrics", rem: "- return True", add: "+ is_greedy: list[bool] | None = None," },
  { id: 40, repo: "deepchecks", sha: "2672ac2b68e5", subj: "aggregation methods for object detection (#1870)", rem: "- return True", add: "+ if np.isnan(class_score) or class_id not in data_classes:" }
]

const LENSES = [
  { key: 'R2', text: 'Argue REJECT on requirement 2: the pre-fix value was ALREADY DISTINGUISHABLE. Look for None, NaN, a raise, a separate flag, a distinct sentinel. CRITICAL: a value in the ALARMING direction is also a reject — an unmeasured comparison scored INCORRECT/False/fail is failing CLOSED, and R2 requires indistinguishability from a HEALTHY measurement specifically.' },
  { key: 'R1', text: 'Argue REJECT on requirement 1: no reachable path where the measurement DID NOT HAPPEN. Look for whether the guarded expression is DETERMINATE at setup — a boolean flag with a literal default, a capability constant, a fixed dimension, an enum — rather than absent runtime data.' },
  { key: 'EXCL', text: 'Argue REJECT on the exclusion list: refactor, rename, retyping, formatting, performance, an arithmetic correction leaving measured/unmeasured status unchanged, docs/CI, or a test-only change that is not itself an inert control. READ THE WHOLE DIFF — the matched hunk is often incidental to a large unrelated change.' },
]

phase('Reject')
const judged = await pipeline(
  C,
  (c) => parallel(LENSES.map((lens) => () =>
    agent(
`Adjudicate a candidate for SP-EXT, a benchmark of SILENT-PASS defects mined from
real fix commits in code this project did not write.

CANDIDATE ${c.id}
  repo    ${c.repo}
  commit  ${c.sha}
  subject ${c.subj}
  matched hunk:  REMOVED  ${c.rem}
                 ADDED    ${c.add}
  path    ${ROOT}/${c.repo}

Read the real diff:   git -C "${ROOT}/${c.repo}" show ${c.sha}
File list (use this for module_path, NEVER infer a path):
                      git -C "${ROOT}/${c.repo}" show --name-only --format= ${c.sha}
Pre-fix source:       git -C "${ROOT}/${c.repo}" show ${c.sha}~1:<path>

INCLUSION RULE — all three must hold of the PRE-FIX code:
  R1  a reachable path existed where the measurement DID NOT HAPPEN
  R2  on it, the value was INDISTINGUISHABLE FROM A REAL, HEALTHY measurement
      (not None, not NaN, not a raise, not a distinct state — AND NOT a value in
       the alarming direction: failing closed is not a silent pass)
  R3  the fix made the absence visible (raise, NaN, None, distinct state, flag,
      skip, warning, fail closed)

YOUR LENS — ${lens.key}: ${lens.text}

RULES
- Default reject=true. The burden is on the candidate.
- Uncertainty resolves to REJECT.
- Quote the ACTUAL pre-fix line. The commit subject often does not describe the
  matched hunk at all.
- module_path and defect_line must come from git output you actually ran. An
  invented path in a benchmark other people rely on is not a typo.
- A false "this is a real case" poisons a public corpus. That is much worse than
  an empty one.`,
      { label: `q2:${c.repo}-${c.id}-${lens.key}`, phase: 'Reject', schema: VERDICT })
  )).then(vs => {
    const v = vs.filter(Boolean)
    const rejects = v.filter(x => x.reject).length
    return { c, verdicts: v, n_reject: rejects, n_reviewers: v.length,
             accepted: v.length >= 2 && rejects < Math.ceil(v.length / 2) }
  })
)

const rows = judged.filter(Boolean).map(j => ({
  id: j.c.id, repo: j.c.repo, sha: j.c.sha, subject: j.c.subj,
  accepted: j.accepted, rejects: `${j.n_reject}/${j.n_reviewers}`,
  paths: j.verdicts.map(v => v.module_path), lines: j.verdicts.map(v => v.defect_line),
  subtypes: j.verdicts.map(v => v.subtype).filter(Boolean),
  consumers: j.verdicts.map(v => v.consumer),
  prefix: j.verdicts.map(v => v.prefix_behaviour),
  postfix: j.verdicts.map(v => v.postfix_behaviour),
  fails: j.verdicts.map(v => v.which_requirement_fails).filter(Boolean),
  rationales: j.verdicts.map(v => ({ reject: v.reject, conf: v.confidence, why: v.rationale })),
}))
const accepted = rows.filter(r => r.accepted)
return { rows, n_accepted: accepted.length, n_total: rows.length,
         accepted_ids: accepted.map(r => r.id),
         repos: [...new Set(accepted.map(r => r.repo))] }
