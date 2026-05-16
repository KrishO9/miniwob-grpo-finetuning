# MiniWoB SFT Implementation Plan

Date: 2026-05-16

This document turns the current `research.md` direction into an implementation plan for supervised fine-tuning a browser-control policy on diverse MiniWoB tasks before returning to GRPO.

References downloaded in this repo:

- `docs/references/how_to_train_your_llm_web_agent_2507_04103.pdf`
- `docs/references/toolrl_reward_is_all_tool_learning_needs_2504_13958.pdf`

External sources:

- How to Train Your LLM Web Agent: A Statistical Diagnosis: https://arxiv.org/pdf/2507.04103
- ToolRL: Reward is All Tool Learning Needs: https://arxiv.org/pdf/2504.13958

## 1. Current State

The BrowserGym HF Space and Kaggle training path are working.

What we have proven:

- The remote BrowserGym environment can reset and step.
- The action parser can canonicalize simple outputs like `Action: click('13')`.
- W&B logging works.
- GRPO can receive non-zero reward and update early.

What failed:

- Training on one repeated `click-test` state collapses.
- Gemma started with poor format behavior, then learned a narrow action after prompt/parser fixes.
- Qwen 0.5B produced clean short actions, but collapsed immediately to zero reward variance.

Conclusion:

- The next bottleneck is data and objective design, not infrastructure.
- We should train a student with verified diverse demonstrations first.
- Then GRPO should start from the SFT checkpoint.

## 2. Paper-Derived Principles

### 2.1 From How to Train Your LLM Web Agent

The paper studies web-agent post-training as a compute allocation problem. The relevant lessons for us are:

- Use a two-stage pipeline: SFT from teacher demonstrations, then on-policy RL.
- Pure RL from an unprepared model is weaker and less compute-efficient.
- Pure SFT can plateau; RL after some SFT improves the compute-performance frontier.
- Branching into RL too early or too late can hurt; evaluate checkpoints and branch when the student is format-stable.
- Evaluate both held-out goals and held-out tasks.
- Curriculum and error-log feedback matter.

How this maps to our project:

- SFT should teach syntax, grounding, stopping behavior, and common action patterns.
- GRPO should optimize environment success after those basics are stable.
- We should not continue using one repeated click prompt as a serious RL task.

### 2.2 From ToolRL

ToolRL studies tool-use training with reward-driven learning. The relevant lesson is not "skip SFT" for our setting. The relevant lessons are:

- Dense, step-wise feedback helps tool learning.
- Tool calls should be executable and verified.
- Bad intermediate tool choices should be filtered or penalized.
- Training improves when the model sees structured tool interaction traces, not just final task success.

How this maps to our project:

- Every teacher action must be executable in BrowserGym.
- We should keep state-action transition records, not only final trajectories.
- Multi-step tasks should contribute one SFT example per verified state-action pair.

## 3. Target Pipeline

The pipeline is:

```text
BrowserGym MiniWoB task registry
  -> raw state collection across tasks/seeds
  -> teacher or oracle action proposal
  -> parser and canonicalizer
  -> BrowserGym verification
  -> verified SFT dataset
  -> Qwen 0.5B SFT
  -> offline and env evaluation
  -> GRPO from SFT checkpoint
```

Core rule:

```text
Teacher proposes. BrowserGym verifies. Only verified examples train the student.
```

## 4. Data Sources

### 4.0 Internet research findings

MiniWoB++ is not just one task family. The official MiniWoB++ docs list a wide set of tasks including click tasks, form tasks, text-entry tasks, email tasks, table tasks, flight tasks, arithmetic tasks, dragging tasks, and visual tasks. The same docs also list no-delay variants for tasks where animation or delayed UI state can make collection noisier.

Relevant docs:

- MiniWoB++ task list: https://miniwob.farama.org/environments/list/
- MiniWoB++ action space: https://miniwob.farama.org/content/action_space/
- MiniWoB++ reward docs: https://miniwob.farama.org/content/reward/
- MiniWoB++ observation docs: https://miniwob.farama.org/content/observation_space/
- BrowserGym docs: https://browsergym.readthedocs.io/

The most important implementation implication is:

```text
The best first dataset should be generated from BrowserGym/MiniWoB itself, not scraped from unrelated traces.
```

Reason:

- BrowserGym gives us the exact observation/action/reward interface we will use at training and evaluation time.
- MiniWoB seeds generate many task instances, goals, layouts, and values.
- Environment verification is available immediately.

Existing demonstration datasets are still useful, but only after conversion. The Pix2Act paper reports using MiniWoB++ human demonstrations and converting them into a different action space; it also notes conversion failures and unsupported behaviors. That matches what we saw in the raw HF sample with repeated `keydown`, `mousedown`, `mouseup`, and `scroll` events. Those traces are not ready-to-train SFT targets for our current high-level action language.

Relevant papers:

- From Pixels to UI Actions: https://arxiv.org/abs/2306.00245
- A data-driven approach for learning to control computers: https://arxiv.org/abs/2202.08137

So the dataset source ranking is:

```text
1. BrowserGym-generated states + verified rule/teacher actions
2. Successful BrowserGym trajectories generated by a teacher policy
3. Converted public demonstration traces, only after action collapsing and verification
4. Raw event logs, only as analysis material
```

### 4.1 Primary source: BrowserGym MiniWoB resets

Use the installed BrowserGym MiniWoB environments as the source of truth.

Data comes from:

- task id
- seed
- goal
- accessibility tree text
- optional screenshot
- environment reward/done after action

Why this is the primary source:

- It matches the inference environment.
- It avoids dataset/schema mismatch.
- It lets us verify every action directly.

### 4.2 Teacher source: Qwen2.5-Coder 7B

Use `Qwen/Qwen2.5-Coder-7B-Instruct` as a teacher for tasks where rule oracles are not enough.

Use deterministic generation:

```text
do_sample = false
max_new_tokens = 48
temperature = unset or 0
top_p = unset
```

Teacher outputs are never trusted directly. They must pass:

```text
parse -> canonicalize -> validate bid -> execute in BrowserGym -> keep only if verified
```

### 4.3 Rule-based source

Use rule oracles for easy tasks:

- single click tasks
- simple text-entry tasks
- obvious form fields

Rule data is preferred when available because it is cheap, deterministic, and easy to debug.

### 4.4 Optional source: existing HF trace datasets

Raw trace datasets with events like `keydown`, `mousedown`, `mouseup`, and `scroll` can be useful only after preprocessing.

Use them as:

- raw source for task-state-action mining
- auxiliary data for later phases

Do not train directly on raw event spam. Collapse traces into high-level canonical actions first.

### 4.5 Real dataset construction strategy

The SFT dataset should be built from three real streams.

Stream A: generated verified states

```text
BrowserGym task_id + seed -> reset -> goal/axtree -> oracle or teacher action -> verify -> SFT example
```

This is the main stream.

Stream B: successful teacher trajectories

```text
BrowserGym task_id + seed -> reset -> Qwen-Coder acts for N steps -> keep only successful traces -> emit one SFT example per step
```

This is the main path for multi-step tasks.

Stream C: converted public demonstrations

```text
raw event trace -> collapse low-level events -> map to canonical action -> replay/verify in BrowserGym -> keep only verified transitions
```

This is optional and should come after Stream A/B are working.

The first SFT milestone should use Stream A and Stream B only.

### 4.6 Diversity dimensions we must track

Every collected example should carry enough metadata to measure diversity.

Required diversity fields:

```text
task_id
task_family
seed
goal_template_or_goal_hash
axtree_hash
action_name
referenced_role
step_index
teacher_source
verified_success
```

The data quality report must show:

```text
examples_per_task
unique_goals_per_task
unique_axtrees_per_task
action_distribution_per_task
step_index_distribution
teacher_source_distribution
verification_success_rate_per_task
deduplication_rate
```

Reject or downsample a task batch if:

```text
one action type is above 90 percent
one normalized AX tree is above 30 percent
verification success is below 20 percent for teacher-labeled data
average prompt length exceeds target model context budget
```

## 5. Task Curriculum

Do not train on all MiniWoB tasks uniformly at first.

The official MiniWoB++ task list includes many task styles:

```text
click tasks
text-entry tasks
forms
menus and lists
date pickers
flight booking tasks
email tasks
tables
arithmetic and logic tasks
dragging and drawing tasks
visual spatial tasks
```

For our first text-only AX-tree agent, do not include everything. The first dataset should prioritize tasks where the required information is visible in `goal` and `axtree_txt`.

Initial inclusion criteria:

```text
goal is textual
target element appears in AX tree
task can be solved by click/type/keyboard_press/noop
no precise coordinate drawing required
no drag gesture required
no purely visual geometry required
no realtime animation dependency
```

Initial exclusion criteria:

```text
drag-box / drag-shape / draw-line style tasks
circle-center / find-midpoint / right-angle geometry tasks
visual-addition and other screenshot-heavy tasks
tasks where reward depends on timing or animation unless a nodelay variant exists
```

### Phase A: click grounding

Goal:

```text
Teach click syntax and bid grounding.
```

Candidate task families:

- `click-test`
- `click-button`
- `click-link`
- `click-dialog`
- `click-option`
- `click-checkboxes`
- `click-tab`

Target:

- 1,000 to 3,000 verified examples
- mostly rule-labeled
- held-out seeds

Sampling plan:

```text
5-10 tasks
100-300 seeds per task
at least 5 unique AX trees per task
at least 3 clickable roles if available
```

### Phase B: text entry

Goal:

```text
Teach field selection and typing.
```

Candidate task families:

- `enter-text`
- `focus-text`
- `login-user`
- `use-autocomplete`

Target:

- 1,000 to 3,000 verified examples
- mix of rules and Qwen-Coder teacher

Sampling plan:

```text
5-8 tasks
100-300 seeds per task
balance short and long text values
include fields with different labels and positions
```

Canonical action examples:

```text
click('15')
type('15', 'Miami')
keyboard_press('ENTER')
```

### Phase C: forms and layout variation

Goal:

```text
Teach goal decomposition and field grounding under layout variation.
```

Candidate task families:

- `choose-list`
- `choose-date-easy`
- `choose-date-medium`
- `multi-layouts`
- `multi-orderings`
- flight-booking style MiniWoB variants if available

Target:

- 2,000 to 5,000 verified examples
- mostly teacher-labeled with verification

Sampling plan:

```text
5-10 tasks
100-500 seeds per task
collect all intermediate states in successful traces
prefer nodelay variants where available
```

### Phase D: short multi-step tasks

Goal:

```text
Teach sequential state-action behavior.
```

Candidate task families:

- `email-inbox-noscroll`
- `email-inbox-forward`
- `email-inbox-reply`
- `email-inbox-delete`
- `click-tab-2-easy`
- `click-tab-2-medium`

Target:

- 500 to 2,000 successful traces
- each trace emits multiple SFT examples

Sampling plan:

```text
3-6 tasks initially
50-200 seeds per task
max 8 steps per trace
keep only successful full traces for first SFT version
```

## 5.5 Proposed first real task set

The exact task ids must be verified by `scripts/list_miniwob_tasks.py` against our installed BrowserGym version. Based on the official MiniWoB++ task list, the first candidate set should be:

Phase A candidates:

```text
click-test
click-test-2
click-button
click-link
click-dialog
click-dialog-2
click-option
click-checkboxes
click-tab
click-menu
```

Phase B candidates:

```text
enter-text
enter-text-2
enter-password
enter-date
enter-time
focus-text
focus-text-2
login-user
use-autocomplete-nodelay
```

Phase C candidates:

```text
choose-list
choose-date-easy
choose-date-medium
choose-date-nodelay
book-flight-nodelay
flight.Alaska
flight.Alaska-auto
multi-layouts
multi-orderings
form-sequence
```

Phase D candidates:

```text
email-inbox-noscroll
email-inbox-delete
email-inbox-forward
email-inbox-reply
email-inbox-important
click-tab-2-easy
click-tab-2-medium
```

Do not assume all of these are available in BrowserGym with the exact same names. Treat this as a candidate list to match against registry output.

## 6. Repository Layout

Add this structure:

```text
browser-control/
  configs/
    sft_qwen05b_miniwob.yaml
    sft_qwen15b_miniwob.yaml
    grpo_qwen05b_after_sft.yaml

  data/
    task_registry/
      miniwob_tasks.txt
      task_buckets.yaml

    raw_states/
      phase_a_click.jsonl
      phase_b_text.jsonl
      phase_c_forms.jsonl
      phase_d_multistep.jsonl

    teacher_actions/
      rule_actions.jsonl
      qwen_coder_7b_actions.jsonl
      rejected_teacher_actions.jsonl

    verified/
      verified_transitions.jsonl
      rejected_transitions.jsonl

    sft/
      train.jsonl
      val.jsonl
      test.jsonl

  reports/
    data_quality_report.md
    sft_eval_qwen05b.md
    grpo_after_sft_report.md

  scripts/
    list_miniwob_tasks.py
    collect_miniwob_states.py
    label_with_rule_oracles.py
    label_with_qwen_coder_teacher.py
    verify_teacher_actions.py
    build_sft_dataset.py
    train_sft.py
    eval_sft_policy.py
    run_grpo_after_sft.py

  src/browser_control/
    actions/
      parser.py
      validators.py

    data/
      schemas.py
      task_buckets.py
      dataset_builder.py

    teachers/
      rule_oracles.py
      qwen_coder_teacher.py
      prompts.py

    envs/
      verifier.py

    training/
      sft.py
      grpo_after_sft.py
```

## 7. Data Schemas

### 7.1 Raw state record

One record per observed state before labeling.

```json
{
  "record_type": "raw_state",
  "task_id": "browsergym/miniwob.click-test",
  "task_family": "click",
  "seed": 23,
  "episode_id": "browsergym-miniwob.click-test_seed-23",
  "step_index": 0,
  "goal": "Click the button.",
  "axtree_txt": "RootWebArea 'Click Test Task', focused\n\t[13] button 'Click Me!'",
  "url": null,
  "screenshot_path": null,
  "done": false,
  "metadata": {
    "browsergym_version": null,
    "openenv_version": null,
    "collection_time": "2026-05-16T00:00:00Z"
  }
}
```

### 7.2 Teacher proposal record

```json
{
  "record_type": "teacher_action",
  "episode_id": "browsergym-miniwob.click-test_seed-23",
  "task_id": "browsergym/miniwob.click-test",
  "seed": 23,
  "step_index": 0,
  "teacher_source": "qwen2.5-coder-7b-instruct",
  "raw_teacher_output": "Action: click('13')",
  "parsed_action": "click('13')",
  "valid_action": true,
  "action_name": "click",
  "referenced_bid": "13",
  "referenced_bid_exists": true,
  "teacher_parse_error": null
}
```

### 7.3 Verification record

```json
{
  "record_type": "verification",
  "episode_id": "browsergym-miniwob.click-test_seed-23",
  "task_id": "browsergym/miniwob.click-test",
  "seed": 23,
  "step_index": 0,
  "action": "click('13')",
  "env_reward": 1.0,
  "done": true,
  "env_error": "",
  "last_action_error": false,
  "verified_success": true,
  "next_goal": "Click the button.",
  "next_axtree_txt": ""
}
```

### 7.4 SFT example

Use chat messages because both Qwen and Gemma are instruction/chat models.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a browser-control policy. Output exactly one BrowserGym action on one line. Do not explain."
    },
    {
      "role": "user",
      "content": "Goal: Click the button.\n\nPage structure:\nRootWebArea 'Click Test Task', focused\n\t[13] button 'Click Me!'\n\nWhat action do you take?"
    },
    {
      "role": "assistant",
      "content": "click('13')"
    }
  ],
  "metadata": {
    "task_id": "browsergym/miniwob.click-test",
    "task_family": "click",
    "seed": 23,
    "step_index": 0,
    "teacher_source": "rule_oracle",
    "env_reward": 1.0,
    "verified_success": true,
    "split": "train"
  }
}
```

## 8. Action Space

Start with a small canonical action space.

Supported in SFT phase 1:

```text
click('<bid>')
type('<bid>', '<text>')
keyboard_press('<key>')
noop()
```

Do not train the student on raw browser events:

```text
keydown
keypress
keyup
mousedown
mouseup
scroll
```

Those can exist in raw trace datasets, but they must be collapsed into semantic actions before SFT.

## 9. Parser And Validator

### 9.1 Parser behavior

Recover action intent from common teacher outputs:

```text
Action: click('13')
click(13)
`click('13')`
The correct action is click('13')
```

Canonicalize to:

```text
click('13')
```

### 9.2 Validation checks

For every parsed action:

- action name is supported
- referenced bid exists in the current `axtree_txt`
- text payload is safe to serialize
- action string matches BrowserGym client syntax
- action is executable without environment error

Invalid examples are rejected or sent to review.

## 10. Task Discovery

Create:

```text
scripts/list_miniwob_tasks.py
```

Responsibilities:

- import `browsergym.miniwob`
- list registered MiniWoB env ids from Gymnasium
- write `data/task_registry/miniwob_tasks.txt`

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/list_miniwob_tasks.py
```

Acceptance criteria:

- prints all MiniWoB env ids in the installed BrowserGym version
- no hardcoded task names required for discovery

## 11. Task Bucketing

Create:

```text
data/task_registry/task_buckets.yaml
```

Initial format:

```yaml
phase_a_click:
  description: "Single-click grounding tasks"
  max_steps: 1
  teacher: rule_oracle
  tasks:
    - browsergym/miniwob.click-test

phase_b_text:
  description: "Text entry and field grounding"
  max_steps: 3
  teacher: rule_oracle_or_qwen_coder
  tasks: []

phase_c_forms:
  description: "Simple forms and layout variation"
  max_steps: 5
  teacher: qwen_coder
  tasks: []

phase_d_multistep:
  description: "Short multi-step tasks"
  max_steps: 8
  teacher: qwen_coder
  tasks: []
```

Fill this after task discovery. Do not assume all task ids exist until registry output is checked.

## 12. State Collection

Create:

```text
scripts/collect_miniwob_states.py
```

Inputs:

- task bucket name
- seed range
- max steps per episode
- output path
- BrowserGym URL or local env mode

Behavior:

1. For each task and seed, reset environment.
2. Save raw state record.
3. If collecting multi-step data, ask teacher/oracle for action, step, and save next state.
4. Stop on `done`, env error, or max steps.

Initial command:

```bash
PYTHONPATH=src .venv/bin/python scripts/collect_miniwob_states.py \
  --bucket phase_a_click \
  --seeds 0:500 \
  --out data/raw_states/phase_a_click.jsonl
```

Acceptance criteria:

- at least 500 raw states
- each record has task id, seed, goal, axtree
- duplicate rate is reported
- failed resets are logged separately

### 12.1 How to get multiple examples per task

MiniWoB tasks are generated from task code and random seeds. A single task can produce many different instances:

```text
different goal values
different labels
different element order
different target ids
different distractors
different form values
```

Therefore, use seed ranges as the main diversity lever.

For single-step tasks:

```text
one seed -> one reset state -> one action label -> one SFT example
```

For multi-step tasks:

```text
one seed -> one successful trace -> N state-action examples
```

Example:

```text
email-inbox-forward seed 17
  step 0: inbox state -> click email from Cathryn
  step 1: email open -> click forward
  step 2: forward dialog -> type recipient
  step 3: forward dialog -> click send
```

This produces four SFT examples from one successful trace.

### 12.2 Seed allocation

Initial seed allocation:

```text
phase_a_click:
  train seeds: 0-799
  val seeds: 800-899
  test seeds: 900-999

phase_b_text:
  train seeds: 0-799
  val seeds: 800-899
  test seeds: 900-999

phase_c_forms:
  train seeds: 0-499
  val seeds: 500-599
  test seeds: 600-699

phase_d_multistep:
  train seeds: 0-199
  val seeds: 200-249
  test seeds: 250-299
```

This can be adjusted after measuring generation cost and duplicate rate.

### 12.3 Deduplication

Deduplicate before splitting and again after splitting.

Use two hashes:

```text
state_hash = sha256(task_id + normalized_goal + normalized_axtree)
example_hash = sha256(task_id + normalized_goal + normalized_axtree + canonical_action)
```

Why two hashes:

- `state_hash` measures observation diversity.
- `example_hash` catches exact training duplicates.

If the same `state_hash` appears across train and test, move all copies to one split.

### 12.4 Task balance

Do not let easy click tasks dominate.

Initial maximums:

```text
max 25 percent from any one task
max 40 percent from phase A after phase B exists
max 60 percent from rule-based labels after Qwen-Coder labels exist
```

Initial minimums:

```text
at least 5 tasks in phase A
at least 3 tasks in phase B before first SFT
at least 1 held-out task family for evaluation
```

## 13. Teacher Labeling

### 13.1 Rule oracles

Create:

```text
src/browser_control/teachers/rule_oracles.py
scripts/label_with_rule_oracles.py
```

Initial oracles:

- click first matching button/link
- click unique clickable
- fill text field when goal contains obvious value
- keyboard press for simple submit tasks

Rule oracle output:

```json
{
  "teacher_source": "rule_oracle",
  "raw_teacher_output": "click('13')",
  "parsed_action": "click('13')"
}
```

### 13.2 Qwen-Coder 7B teacher

Create:

```text
src/browser_control/teachers/qwen_coder_teacher.py
scripts/label_with_qwen_coder_teacher.py
```

Teacher prompt:

```text
You are an expert BrowserGym web agent.

Given a task goal and page accessibility tree, output exactly one BrowserGym action.

Allowed actions:
click('<bid>')
type('<bid>', '<text>')
keyboard_press('<key>')
noop()

Rules:
- Output only one action.
- Do not explain.
- Do not use markdown.
- Use only bid values that appear in the page structure.
- Prefer the action that makes progress toward the goal.

Goal:
{goal}

Page structure:
{axtree_txt}

Action:
```

Kaggle-friendly model loading:

```python
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
load_in_4bit = True
device_map = "auto"
max_new_tokens = 48
do_sample = False
```

Acceptance criteria:

- parse rate above 80 percent on phase B/C samples
- bid hallucination rate below 20 percent before verification
- all outputs logged raw and parsed

## 14. Verification

Create:

```text
scripts/verify_teacher_actions.py
src/browser_control/envs/verifier.py
```

Verification modes:

- one-step verification for phase A/B
- trajectory verification for phase C/D

One-step verification:

1. Reset task with same seed.
2. Execute proposed action.
3. Record reward, done, error.
4. Keep example if success criteria pass.

Multi-step verification:

1. Reset task with same seed.
2. For each step, execute proposed action from that state.
3. Keep full trace only if final task succeeds.
4. Emit one SFT example for each state-action pair from successful traces.

Success criteria:

```text
phase_a_click:
  done == true and reward > 0

phase_b_text:
  action executes without error; final trace must solve task when multi-step

phase_c_forms:
  full trace success preferred; partial transitions can be kept only if later confirmed by successful trace

phase_d_multistep:
  keep only successful full traces at first
```

Rejected data is valuable. Save it to:

```text
data/verified/rejected_transitions.jsonl
```

## 14.5 Public demonstration trace conversion

Public MiniWoB demonstration traces can contain low-level events such as:

```text
keydown
keypress
keyup
mousedown
mouseup
scroll
click
```

The sample we inspected looked like this style of data: repeated keyboard and mouse events plus an HTML/DOM snapshot and task fields. This is useful, but not directly usable for our SFT target.

Conversion rule:

```text
raw event trace -> semantic segment -> canonical BrowserGym action -> replay verification
```

Example conversion:

```text
mousedown ref=15 + mouseup ref=15 + click ref=15
  -> click('15')

focus input ref=15 + keypress sequence for "Miami"
  -> type('15', 'Miami')

scroll ref=1 repeated
  -> reject initially unless the task requires scroll and the final trace verifies
```

Do not include converted traces unless:

```text
the canonical action can be replayed in BrowserGym
the resulting trace solves the task
the state snapshot aligns with the action step
```

Why this is later-phase work:

- human or browser event logs often contain duplicates
- renderers can differ between the original trace collection setup and BrowserGym
- some actions like drag/highlight/copy-paste may not map cleanly to our first action space
- verification is mandatory and may reject many examples

Recommended timing:

```text
use public traces only after our generated BrowserGym dataset and SFT evaluator work
```

## 15. Dataset Quality Reports

Create:

```text
scripts/report_dataset_quality.py
reports/data_quality_report.md
```

Report:

- number of records per task
- number of unique goals per task
- number of unique normalized AX trees per task
- duplicate rate
- teacher source mix
- parse success rate
- verification success rate
- action distribution
- average prompt length
- train/val/test split counts

Reject or downsample tasks where:

- state diversity is low
- verification rate is too low
- one action dominates more than 90 percent of samples
- AX trees are too long for the target context window

## 16. Split Strategy

Split by task and seed, not random rows.

Recommended first split:

```text
train seeds: 0-799
val seeds: 800-899
test seeds: 900-999
```

For held-out task evaluation:

```text
train tasks: 80 percent of task families
test tasks: held-out variants
```

Never allow the same `(task_id, seed, step_index)` in multiple splits.

Dedup key:

```text
sha256(task_id + normalized_goal + normalized_axtree + action)
```

## 17. SFT Dataset Builder

Create:

```text
scripts/build_sft_dataset.py
```

Inputs:

- verified transitions
- system prompt template
- split strategy

Outputs:

```text
data/sft/train.jsonl
data/sft/val.jsonl
data/sft/test.jsonl
```

Each record should contain `messages` plus metadata.

Use a stable system prompt:

```text
You are a browser-control policy. Output exactly one BrowserGym action on one line. Do not explain.
```

User prompt:

```text
Goal: {goal}

Page structure:
{axtree_txt}

What action do you take?
```

Assistant target:

```text
{canonical_action}
```

## 18. SFT Training

Initial student:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Reason:

- Qwen already produced short, terminated action outputs in our GRPO test.
- It is small enough for Kaggle.
- It is a better formatter than Gemma 270M in our observations.

Create:

```text
configs/sft_qwen05b_miniwob.yaml
scripts/train_sft.py
```

Initial config:

```yaml
model_name: Qwen/Qwen2.5-0.5B-Instruct
dataset_train: data/sft/train.jsonl
dataset_val: data/sft/val.jsonl
output_dir: /kaggle/working/model_checkpoints/qwen05b-miniwob-sft
max_seq_length: 2048
learning_rate: 2.0e-4
num_train_epochs: 2
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
packing: false
bf16: false
fp16: true
use_peft: true
lora_r: 8
lora_alpha: 16
lora_dropout: 0.0
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
wandb_project: browser-control-sft
```

Training implementation:

- use TRL `SFTTrainer`
- format with tokenizer chat template if `SFTTrainer` does not handle `messages` automatically in installed version
- log validation loss
- save LoRA adapter
- save tokenizer with adapter

## 19. SFT Evaluation

Create:

```text
scripts/eval_sft_policy.py
reports/sft_eval_qwen05b.md
```

Offline metrics:

- valid action rate
- exact canonical match rate
- referenced bid exists rate
- extra text rate
- noop rate
- completion length
- clipped ratio

Environment metrics:

- one-step success rate
- full-trace success rate for multi-step tasks
- average env reward
- env error rate
- per-task success rate

Minimum gate before GRPO:

```text
valid_action_rate >= 95 percent on val
referenced_bid_exists_rate >= 90 percent on val
extra_text_rate <= 10 percent
click task env success >= 85 percent
text-entry task env success >= 60 percent
no severe per-task collapse
```

If the model fails these gates, continue improving SFT data before RL.

## 20. Error Analysis

Every eval run should save:

```text
reports/eval_failures_qwen05b.jsonl
```

Failure schema:

```json
{
  "task_id": "...",
  "seed": 901,
  "goal": "...",
  "axtree_txt": "...",
  "raw_completion": "...",
  "parsed_action": "click('99')",
  "expected_action": "click('13')",
  "env_reward": 0.0,
  "done": false,
  "failure_type": "wrong_bid"
}
```

Failure buckets:

- invalid format
- extra text
- hallucinated bid
- wrong bid
- wrong action type
- premature noop
- task requires multi-step
- AX tree lacks enough information
- teacher label likely wrong

Use these failures to create the next data collection batch.

## 21. GRPO After SFT

Only start GRPO after SFT passes the gates.

Create:

```text
configs/grpo_qwen05b_after_sft.yaml
scripts/run_grpo_after_sft.py
```

Start from:

```text
base model: Qwen/Qwen2.5-0.5B-Instruct
adapter: /kaggle/working/model_checkpoints/qwen05b-miniwob-sft
```

Use lower dense shaping after SFT:

```yaml
valid_action_reward: 0.01
element_id_reward: 0.02
invalid_action_penalty: -0.10
noop_with_clickables_penalty: -0.10
env_error_penalty: -0.20
env_success_reward: 1.0
```

Use diverse task batches:

```text
phase_a_click: 40 percent
phase_b_text: 30 percent
phase_c_forms: 20 percent
phase_d_multistep: 10 percent
```

Watch:

- reward std
- fraction of zero-std groups
- entropy collapse
- per-task success
- action validity

If all samples succeed, increase task diversity.
If all samples fail, reduce task difficulty or add SFT data.

## 22. First Concrete Execution Plan

### Step 1: Discover tasks

```bash
PYTHONPATH=src .venv/bin/python scripts/list_miniwob_tasks.py
```

Output:

```text
data/task_registry/miniwob_tasks.txt
```

### Step 2: Build first task bucket

Manually create:

```text
data/task_registry/task_buckets.yaml
```

Start with 5-10 simple tasks confirmed by registry.

### Step 3: Collect phase A states

```bash
PYTHONPATH=src .venv/bin/python scripts/collect_miniwob_states.py \
  --bucket phase_a_click \
  --seeds 0:500 \
  --out data/raw_states/phase_a_click.jsonl
```

### Step 4: Label with rule oracle

```bash
PYTHONPATH=src .venv/bin/python scripts/label_with_rule_oracles.py \
  --input data/raw_states/phase_a_click.jsonl \
  --out data/teacher_actions/rule_actions.jsonl
```

### Step 5: Verify labels

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_teacher_actions.py \
  --input data/teacher_actions/rule_actions.jsonl \
  --verified-out data/verified/verified_transitions.jsonl \
  --rejected-out data/verified/rejected_transitions.jsonl
```

### Step 6: Build SFT data

```bash
PYTHONPATH=src .venv/bin/python scripts/build_sft_dataset.py \
  --input data/verified/verified_transitions.jsonl \
  --out-dir data/sft
```

### Step 7: Train Qwen 0.5B SFT

```bash
PYTHONPATH=src .venv/bin/python scripts/train_sft.py \
  --config configs/sft_qwen05b_miniwob.yaml
```

### Step 8: Evaluate

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_sft_policy.py \
  --checkpoint /kaggle/working/model_checkpoints/qwen05b-miniwob-sft \
  --test data/sft/test.jsonl \
  --report reports/sft_eval_qwen05b.md
```

## 23. Acceptance Criteria For First SFT Milestone

The first milestone is complete when:

- at least 1,000 verified SFT examples exist
- at least 5 MiniWoB tasks are included
- validation split has held-out seeds
- Qwen 0.5B SFT completes
- valid action rate is at least 95 percent
- click-task environment success is at least 85 percent
- W&B logs train and eval curves
- failure report exists

Do not move to GRPO before this milestone passes.

## 24. Research Risks

### Teacher accuracy risk

Qwen-Coder 7B may output plausible but wrong actions.

Mitigation:

- verify every action
- train only on successful verified examples
- keep rejected examples for prompt improvement

### Dataset leakage risk

MiniWoB can repeat similar states across seeds.

Mitigation:

- split by seed
- hash normalized goal and AX tree
- deduplicate before split

### AX-tree limitation risk

Some tasks require visual or layout information not captured by text AX tree.

Mitigation:

- bucket tasks by AX-tree suitability
- collect screenshots for future VLM experiments
- exclude visual-heavy tasks from first SFT pass

### SFT overfitting risk

The student may memorize task templates.

Mitigation:

- held-out seeds
- held-out task variants
- per-task metrics
- add more tasks before increasing epochs

### RL collapse risk

GRPO may still collapse after SFT.

Mitigation:

- use diverse task batches
- reduce dense format reward
- monitor zero-variance groups
- increase difficulty when all samples succeed

## 25. Immediate Next Work Items

1. Implement shared action parser/canonicalizer module.
2. Implement task registry listing.
3. Implement raw state collection for confirmed MiniWoB task ids.
4. Implement rule oracle for click tasks.
5. Implement verification script.
6. Build first 1,000-example SFT dataset.
7. Train Qwen 0.5B SFT.
8. Evaluate held-out seed success.
9. Only then return to GRPO.

## 26. Decision

The next serious training path is:

```text
Qwen-Coder 7B and rule oracles -> verified MiniWoB SFT data -> Qwen 0.5B SFT -> evaluation -> GRPO from SFT
```

The current one-task GRPO pipeline remains useful as a smoke test, but it should not be used as the main training experiment.
