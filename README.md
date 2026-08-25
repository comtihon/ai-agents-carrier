# AI Agents Carrier

Workflow orchestration backend for AI-assisted software delivery. Workflows are defined in YAML, run as LangGraph graphs, and can involve LLM reasoning, MCP tool calls, human approval gates, cron/webhook triggers, and autonomous code execution via OpenHands.

---

## How to run

```bash
docker-compose up -d          # start MongoDB
pip install -e ".[dev]"       # install deps
cp .env.example .env          # configure (see below)
uvicorn app.main:app --reload
```

API at `http://localhost:8000`. Health check: `GET /health`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | — | `anthropic` or `openai` |
| `LLM_MODEL` | provider default | Model name override |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `GOOGLE_API_KEY` | — | Required when `LLM_PROVIDER=google` |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DATABASE` | `langgraph_backend` | Database name |
| `WORKFLOW_BACKEND` | `localfiles` | `localfiles` or `mongodb` |
| `GRAPH_DEFINITIONS_PATH` | `graphs` | Directory of YAML workflow files (localfiles backend) |
| `BASE_URL` | `http://localhost:8000` | Public URL — used to build approval callback links |
| `WEBHOOK_SECRET` | — | HMAC-SHA256 secret for incoming webhook signatures |
| `PUBSUB_ENABLED` | `false` | Subscribe to the topics of `pubsub` trigger steps |
| `PUBSUB_PROJECT_ID` | — | GCP project short topic / subscription names resolve against |
| `PUBSUB_SUBSCRIPTION_PREFIX` | `aac-` | Prefix for subscriptions the backend creates itself |
| `PUBSUB_ACK_DEADLINE_SECONDS` | `60` | Ack deadline for created subscriptions |
| `PUBSUB_MAX_MESSAGES` | `10` | Messages pulled concurrently per subscription |
| `PUBSUB_DROP_INVALID_MESSAGES` | `true` | Ack (drop) events failing schema validation instead of nacking them |
| `PUBSUB_DELETE_ORPHANED_SUBSCRIPTIONS` | `true` | Delete a backend-created subscription once its last trigger step is gone |
| `OAUTH_ENABLED` | `false` | Enable JWT Bearer auth on all endpoints |
| `OAUTH_JWKS_URL` | — | JWKS endpoint for token validation |
| `OAUTH_ISSUER` | — | Expected token issuer; also where the userinfo endpoint is derived from |
| `OAUTH_AUDIENCE` | — | Expected audience; audience is not verified when unset |
| `AUTH_ENFORCE_PERMISSIONS` | `false` | Enforce the role→permission model. Off = every authenticated caller has full access, and denials are only reported (see [Role-based permissions](#role-based-permissions)) |
| `AUTH_PROJECT_ID` | — | Identity-provider project whose roles claim is read (machine tokens carry `urn:zitadel:iam:org:project:<id>:roles`) |
| `AUTH_ACCESS_ROLES` | `[]` | Roles allowed to reach the API at all. Allow-list: an unlisted role is denied |
| `AUTH_READ_ROLES` | `[]` | Roles granted READ (`GET`/`HEAD`/`OPTIONS`, and the `list_*` / `get_*` tools) |
| `AUTH_WRITE_ROLES` | `[]` | Roles granted WRITE (`POST`/`PUT`/`PATCH`, creates, updates, run control) |
| `AUTH_DELETE_ROLES` | `[]` | Roles granted DELETE (`DELETE`, and the `delete_*` tools) |
| `AUTH_ADMIN_ROLES` | `[]` | Roles granted ADMIN — python steps that run on the backend pod. Keep this narrowest |
| `MANAGEMENT_MCP_API_KEY` | — | Static key for `/mcp/management`. Capped below ADMIN: a shared secret cannot be attributed to a person |
| `OPENHANDS_BASE_URL` | `http://openhands:3000` | OpenHands service URL |
| `OPENHANDS_API_KEY` | — | OpenHands auth token |
| `OPENHANDS_MOCK_MODE` | `true` | Return stub results instead of calling OpenHands |
| `STATE_DIVERGENCE_PROBE` | `false` | Log every key where the runner's hand-merged run state disagrees with the LangGraph checkpoint. Diagnostic only — never writes — and costs one extra checkpoint read per node |
| `DOCKER_REGISTRY_USERNAME` | — | Registry username for pulling private images (DockerRuntime) |
| `DOCKER_REGISTRY_PASSWORD` | — | Registry password / token for pulling private images |
| `LOCAL_AGENT_DIR` | — (`/opt/pi-cloud-agent` in the `-full` image) | Path to a [pi-cloud-agent](https://github.com/comtihon/pi-cloud-agent) checkout. Required for the `local` agent runtime |
| `LOCAL_AGENT_COMMAND` | `node src/server.js` | Command that starts the local agent's HTTP server, run with `LOCAL_AGENT_DIR` as cwd |
| `META_LLM_PROVIDER` | — | LLM provider for post-agent analysis (`anthropic` or `openai`; defaults to `LLM_PROVIDER`) |
| `META_LLM_MODEL` | `claude-haiku-4-5-20251001` | Model for post-agent meta-analysis (haiku recommended for cost/speed) |

### Role-based permissions

Authentication answers *who is calling*; this answers *what they may do*. Five
coarse tiers, and the mapping from identity-provider role names onto them is
entirely configuration, so no deployment's role names live in this repository:

| tier | grants |
|---|---|
| ACCESS | may reach the API at all — the tenancy gate |
| READ | read workflows, runs, definitions, data sources, traces |
| WRITE | create, edit and run workflows; control runs |
| DELETE | delete workflows, runs, agents, data sources |
| ADMIN | python steps that execute on the backend pod (see [`python`](#python--python-script)) |

ADMIN is not "WRITE plus a bit": its blast radius is the backend process and
every credential it holds, not the data.

**How a requirement is derived.** REST routes take theirs from the HTTP method
(`GET`→READ, `POST`/`PUT`/`PATCH`→WRITE, `DELETE`→DELETE, anything unknown→WRITE),
so a newly added route inherits a sane requirement instead of defaulting to
unprotected. The tool surfaces — `/mcp/management` and the chat agent, which both
arrive as one `POST` — cannot use the method, so their shared implementations in
`app/application/management_tools.py` and `app/application/run_control.py` are
each gated individually. The mapping is asserted in
`tests/unit/test_tool_permission_gates.py`, so a new tool fails the suite until
it is classified.

**Principals that are not users.** The `/mcp/management` static API key is a
shared secret with no person behind it: it gets READ/WRITE/DELETE and never
ADMIN. Callers that authenticate by another mechanism and carry no roles at all —
Slack HMAC callbacks, signed webhooks, Pub/Sub triggers, agent-container
callbacks — bypass this model entirely (see `_UNPROTECTED_PREFIXES` in
`app/api/middleware/auth.py`); nothing about enforcement changes them.

**Shadow mode.** With `AUTH_ENFORCE_PERMISSIONS=false` the model is still
evaluated on every authenticated request and every would-be denial is logged
while the request is served, so turning enforcement on is an evidence-based
decision. Read the evidence before flipping it:

```bash
# Every caller that would be locked out, over the last week
gcloud logging read \
  'resource.type="k8s_container"
   AND resource.labels.namespace_name="langgraph"
   AND textPayload:"RBAC shadow"' \
  --project <gcp-project> --freshness=7d --format='value(textPayload)' | sort | uniq -c | sort -rn
```

Each line names the subject id, the permission it would have been denied, the
roles it actually holds, and the method and path. An empty result over a full
business cycle is the go signal; a subject with `roles=[]` is the expected
failure and means its token carries no roles claim — the identity provider needs
to assert roles for that client (and the client to request the roles scope)
*before* enforcement goes on, not after.

Recommended order: deploy with enforcement off → read the query above → fix any
roleless client → enforce in a non-production environment → enforce in
production.

### MCP integrations

MCP servers are declared in configuration, not in code — the same contract
`AGENT_TOOLS` has for bash-level tools. Adding a server is a `values.yaml` entry;
nothing in the backend or the agent image knows any server name.

**Grant chain.** A server travels from the chart to a running agent in four steps:

1. **Declare** it in `mcpIntegrations` (helm) → rendered into `MCP_INTEGRATIONS`
   (a JSON array in the ConfigMap) plus one `secretKeyRef` env var per token.
2. **Offer** it: `GET /api/v1/agents/mcp-integrations` lists every declared
   server with `enabled` and `configured`, which is what the UI's mcp addon
   renders. A declared server with an unresolved token shows as
   `configured: false` rather than disappearing.
3. **Check** it in an agent's `mcp` addon. This is the only thing that grants a
   server: no addon means no servers, and a name the registry does not declare
   grants nothing.
4. **Spawn**: `POST /start` carries `agent_config.mcp_servers` — the checked
   servers only, each with its endpoint or argv and its *resolved* credentials.
   The agent writes them to `mcp.json` and connects; nothing is special-cased on
   its side either.

Bash-level tools follow the same path through `AGENT_TOOLS` and the `tools`
addon, with one extra step: the agent registers a granted tool only if its
`command` is actually on PATH, and the runtime shadows every command it was not
granted with an exit-127 stub.

```yaml
mcpIntegrations:
  - name: github
    transport: streamable_http
    url: https://api.githubcopilot.com/mcp/
    apiKeySecret:                      # → GITHUB_MCP_API_KEY from this Secret
      name: langgraph-backend-secrets
      key: GITHUB_MCP_TOKEN
  - name: jira                         # stdio: sooperset/mcp-atlassian via uvx
    transport: stdio
    command: uvx
    args: ["mcp-atlassian"]
    envFromConfig:                     # copies backend env vars into its env
      JIRA_URL: MCP_JIRA_JIRA_URL
      JIRA_USERNAME: MCP_JIRA_USERNAME
      JIRA_API_TOKEN: MCP_JIRA_API_TOKEN
  - name: semble
    transport: stdio
    command: semble
    prestartHttp: false                # CLI has no --transport/--port flags
    eagerStart: false                  # agent-side binary, not in this container
```

| Field | Meaning |
|---|---|
| `name` | Server name, as shown in the mcp addon picker |
| `enabled` | `false` hides it from agents without deleting the entry (default `true`) |
| `transport` | `streamable_http` \| `sse` \| `stdio` |
| `url` | Endpoint (HTTP transports) |
| `agentUrl` | Endpoint a spawned agent must dial, when `url` is not resolvable from an agent pod |
| `apiKeyEnv` / `apiKeySecret` | Env var carrying the bearer token, and the Secret filling it. Defaults to `<NAME>_MCP_API_KEY` |
| `command` / `args` | Binary and argv (stdio transport) |
| `env` / `envFromConfig` | Literal env vars, or ones copied from a backend env var / `.env` key |
| `prestartHttp` | `false` when the CLI cannot be re-hosted over HTTP by the agent |
| `eagerStart` | `false` when the backend must not dial the server during startup |

Tokens are referenced by env var name, never inlined: the JSON blob lives in a
ConfigMap, so `apiKeyEnv` + `apiKeySecret` keeps the secret in a Secret. For a
local `.env`, `MCP_INTEGRATIONS` accepts an inline `api_key` instead.

**How a server authenticates.** Four mechanisms, and which one applies is a
property of the vendor, not a choice:

| Mechanism | Declare it as | Works from a backend? |
|---|---|---|
| Static bearer token (PAT, app token, copilot token) | HTTP entry, `apiKeyEnv` + `apiKeySecret` | Yes — this is the normal case |
| Static token read by a self-hosted server | stdio entry, `envFromConfig` | Yes |
| No credential (localhost dev servers) | HTTP entry, `auth: none` | Yes |
| User-delegated OAuth 2.1 (auth code + PKCE, often with dynamic client registration) | — | **No** |

The last row is the one that bites. A vendor whose only MCP auth is delegated
OAuth needs a human to complete a browser consent flow, so no deployment-time
configuration reaches it — `mcp.hubspot.com`, `mcp.notion.com`,
`mcp.miro.com` and Figma's hosted endpoint are all in this category today. This
is not something `SERVICE_AUTH_*` can paper over: that facility implements the
OAuth2 **JWT bearer** grant (RFC 7523), a client-to-authorization-server flow,
and these vendors do not offer a client-credentials or JWT-bearer grant for MCP
at all. Two ways around it:

- **Self-host a server that takes a static token.** Notion's official
  `@notionhq/notion-mcp-server` reads an internal-integration token from
  `NOTION_TOKEN`, so it becomes an ordinary stdio entry with `envFromConfig`.
- **Skip MCP and use a data source.** The vendor's REST API almost always
  accepts a static token even when its MCP server does not. This is the right
  answer for HubSpot: its remote MCP server is OAuth-2.1-only, while its REST API
  takes a private-app token — so it goes in as a data source and reaches agents
  through the `datasources` bridge. (`@hubspot/mcp-server` on npm is tooling for
  building HubSpot apps, not a CRM-data server, so it is not a substitute.)

Because a tokenless HTTP entry is far more often an unfinished one than a
genuinely open endpoint, `auth` defaults to `bearer`: an entry with no token
resolving is reported `configured: false` in the picker instead of looking ready
and failing at first use. State `auth: none` to opt out.

**MCP entry or bash tool?** Both registries end at the same agent, so pick by
what the thing actually is:

| The binary… | Goes in | Why |
|---|---|---|
| serves MCP over stdio or HTTP (`semble`, `uvx mcp-atlassian`) | `mcpIntegrations` | The server publishes its own tool schemas and instructions; hand-writing them as `cli_tools` would duplicate what it already declares |
| is only a CLI (`graphify`, `kubectl`, `gcloud`) | `AGENT_TOOLS` | Nothing to introspect, so the invocations are declared as `cli_tools` — and the agent gets a PATH check plus exit-127 stubbing of the binary for agents without the grant |

`semble` is an MCP server (`semble[mcp]`, bare `semble` speaks stdio JSON-RPC) that
happens to live in the agent image rather than behind a URL, which is exactly what
`eagerStart: false` + `prestartHttp: false` express. `graphify` has no MCP mode, so
it is a tool. A binary that is both is declared once, on whichever side you want
the agent to use.

One consequence worth knowing: an ungranted MCP server's launcher is *not*
stubbed on PATH the way an ungranted tool's command is, so an agent can still
invoke that binary from bash. Conversely, a command a granted stdio server needs
is never stubbed on behalf of some other ungranted entry — one entry's denial does
not revoke another's grant.

**Migrating from the per-server env vars.** `MCP_<NAME>_ENABLED` / `_URL` /
`_API_KEY` are no longer read. Port each one to an `mcpIntegrations` entry —
`MCP_FOO_URL` becomes `url`, `MCP_FOO_API_KEY` becomes an `apiKeySecret`, and
Jira's stdio vars become `envFromConfig` as shown above. Stale vars are ignored
rather than rejected, so startup logs every `MCP_<NAME>_ENABLED=true` that has no
matching entry: check the log after upgrading, because an unported server simply
stops being offered.

**Data sources are not MCP entries.** The in-process `datasources` server is
always available and needs no declaration — every data source configured through
`/api/v1/datasources` becomes an MCP tool (`ds_<source>_<operation>`) on it, and
the backend executes those calls itself with the source's stored credentials. So
an API that only offers REST (or one whose MCP server needs an OAuth flow this
registry cannot drive) is added as a data source, and agents reach it by
checking `datasources` in their mcp addon. The backend dials the mount over
loopback while agents are handed `AGENT_CALLBACK_URL` (or `BASE_URL`) +
`/mcp/datasources` — set `MCP_DATASOURCES_API_KEY` so they are let in. Disable
the bridge with `MCP_DATASOURCES_ENABLED=false`, or declare an entry named
`datasources` to override its addressing.

### Docker runtime — private registry auth

When a workflow step uses `runtime: docker`, the backend pulls the agent image via the Docker daemon. For private registries, set credentials via env vars:

| Registry | `DOCKER_REGISTRY_USERNAME` | `DOCKER_REGISTRY_PASSWORD` |
|---|---|---|
| Google Artifact Registry | `oauth2accesstoken` | `$(gcloud auth print-access-token)` |
| AWS ECR | `AWS` | `$(aws ecr get-login-password --region <region>)` |
| Docker Hub / other | your username | password or personal access token |

No credentials set → pull proceeds without auth (public images, or if the Docker daemon already has credentials configured via `docker login`).

**Local `.env`:**
```bash
DOCKER_REGISTRY_USERNAME=oauth2accesstoken
DOCKER_REGISTRY_PASSWORD=ya29.your-token-here
```

**Kubernetes secret:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: registry-credentials
stringData:
  DOCKER_REGISTRY_USERNAME: oauth2accesstoken
  DOCKER_REGISTRY_PASSWORD: <token>
```
Then reference with `envFrom: - secretRef: name: registry-credentials` in the deployment.

### Agent meta-analysis

After each `langgraph-agent` or `claude-agent` step completes, the backend runs a lightweight internal LLM call to decide how to proceed:

- **PROCEED** — agent answered the request; workflow continues normally
- **ASK_CLARIFICATION** — agent was blocked or needs more info; the UI shows a question form before the workflow continues
- **ASK_APPROVAL** — output should be reviewed by a human (falls through to the next `human_approval` step)

Configure with a fast, cheap model to minimise cost:

```env
META_LLM_PROVIDER=anthropic
META_LLM_MODEL=claude-haiku-4-5-20251001
```

When `META_LLM_PROVIDER` is not set, the main `LLM_PROVIDER` value is used.

---

## Workflow definitions

Workflows are YAML files in `graphs/` (or the path set by `GRAPH_DEFINITIONS_PATH`). They are loaded at startup and can also be managed via the REST API (`GET/POST/PUT/DELETE /api/v1/workflows`).  
Workflows can be stored in MongoDB if the backend is configured.

```yaml
id: my-workflow
name: My Workflow
description: "..."
steps:
  - id: step-one
    type: llm
    ...
  - id: step-two
    type: human_approval
    ...
```

Steps run sequentially. Each step can be skipped with `when: <state-key>` — the step is skipped if `state[key]` is falsy.

---

## Step types

> **Removed: `llm_structured`.** It ran a whole tool-calling loop inside one
> graph node, so nothing in it was checkpointed: a restart mid-loop replayed
> every tool call from scratch. Use `langgraph-agent` / `claude-agent` for
> tool-using work — its `output_mapping` replaces the `output` schema — or a
> plain `llm` step when one call is enough. A workflow still carrying an
> `llm_structured` step fails to load with a message naming the step.

### `llm` — single LLM call

One-shot call, no tool loop. Result stored as a string.

```yaml
- id: plan
  type: llm
  system_prompt: "You are a planning assistant."
  user_template: "Context: {context}"
  output_key: plan
```

### `mcp` — call an MCP tool directly

```yaml
- id: fetch_board
  type: mcp
  tool: miro_get_board
  tool_input:
    board_id: "{board_id}"
  output_key: board_data
```

### `human_approval` — pause for human review

Pauses the run (`status: waiting_approval`). Resume via `POST /api/v1/workflows/runs/{id}/approve` or `/reject`. The `approved` and `reject_reason` keys are written to state automatically.

```yaml
- id: approve
  type: human_approval
  interrupt_payload:
    plan: "{plan}"
  notify:                               # optional — send an HTTP notification
    url: "https://hooks.example.com/approval"
    auth:
      type: bearer                      # bearer | basic (optional)
      token: "my-token"
    payload:
      text: "Approval needed: {plan}"
      approve_url: "{approve_url}"      # auto-injected callback URL
      reject_url: "{reject_url}"        # auto-injected callback URL
```

Callback endpoints (no auth required — the UUID is the secret):

```
POST /api/v1/callbacks/{run_id}/approve
POST /api/v1/callbacks/{run_id}/reject   body: {"reason": "..."}
```

### `execute` — run code via OpenHands

```yaml
- id: implement
  type: execute
  when: approved
  repo_template: "{repo}"
  instructions_template: "Implement {ticket_id} per the plan:\n{plan}"
  output_key: implementation
```

### `http_call` — outbound HTTP request

```yaml
- id: create_ticket
  type: http_call
  url: "https://api.example.com/issues"
  method: POST
  headers:
    Authorization: "Bearer {token}"
  body:
    title: "{request}"
  output_key: ticket
```

### `slack` — post to / read from a chat provider

One step type for every chat provider: `provider` names an implementation
registered in `app.infrastructure.messaging` (`slack` today), so a second
provider is a new `MessagingProvider` subclass plus a registry entry rather
than a new step type.

```yaml
- id: read_commands            # action: post | reply | history | thread | dm | delete
  type: slack
  action: history
  channel: "C0BLDDSEB1D"
  oldest: "{window_start_ts}"  # history only — lower time bound
  limit: 200                   # history only
  output_key: slack_messages
  ignore_errors: true          # capture the error under output_key instead
                               #   of failing the run

- id: confirm
  type: slack
  action: reply
  channel: "C0BLDDSEB1D"
  items: overrides.confirmations   # state path to a list of {thread_id, text}
  skip_if_replied: true            # read the thread first; do not post a reply
                                   #   whose text is already there
  output_key: confirmed

- id: warn_owner
  type: slack
  action: dm
  user_id: "{owner_slack_id}"      # the DM channel is opened for you
  text: "{board_warning}"
  output_key: dm
```

There is **no token field**. The provider reads `SLACK_BOT_TOKEN` from the
deployment's settings, so a credential never lives in a workflow definition, a
data source or step config — and provider errors are scrubbed of it before they
can reach run state.

`post`/`reply`/`dm`/`delete` write `{message_id, channel}` (a batch `reply`
writes `{posted, skipped, posted_count, skipped_count}`); `history`/`thread`
write a list of `{id, channel, text, author, thread_id}`, which the Slack
provider augments with its own `ts`/`user`/`thread_ts` names so scripts written
against the Slack Web API keep working.

### `workflow` — spawn a child workflow

```yaml
- id: spawn_child
  type: workflow
  workflow_id: another-workflow
  input_template: "{request}"
  output_key: child_result
```

### `langgraph-agent` / `claude-agent` — autonomous agent step

Spawns a registered agent, sends the task, suspends until the agent calls back with its result. The runtime decides *where* the agent runs — `local` (child process of the backend), `docker` (container) or `k8s` (Helm release) — not *what* it is: all three run [pi-cloud-agent](https://github.com/comtihon/pi-cloud-agent) and speak the same HTTP protocol.

```yaml
- id: researcher
  type: langgraph-agent          # or claude-agent
  agent_id: my-researcher        # must exist in /api/v1/agents
  runtime_override: docker       # local | docker | k8s (defaults to agent's default_runtime)
  image: myregistry/my-agent:1.0 # Docker image override
  output_key: agent_result       # stores the agent's text result
  compression_level: full        # none | lite | full | ultra — caveman-compress the agent's responses
  env_vars:                      # additional env vars forwarded to the container
    - name: GOOGLE_APPLICATION_CREDENTIALS_JSON
      from_config: GOOGLE_APPLICATION_CREDENTIALS_JSON   # from backend config
    - name: MY_VAR
      value: custom-value        # literal value
  output_mapping:                # map individual agent output keys → state keys (optional)
    result: agent_result
```

**After the agent completes**, a meta-LLM call analyzes the output and decides:
- **PROCEED** — result is good, workflow continues
- **ASK_CLARIFICATION** — agent was blocked; UI shows a question form before proceeding
- **ASK_APPROVAL** — output needs human sign-off (falls through to the next `human_approval` step)

Configure the meta-LLM via `META_LLM_PROVIDER` / `META_LLM_MODEL` (default: haiku).

### `parallel` — fan-out to concurrent branches

Starts multiple branches in parallel. Each target step runs concurrently; edges define which steps are in the parallel group.

```yaml
- id: fan_out
  type: parallel
  max_parallel: 3      # max concurrent branches (default: unlimited)
  targets:
    - branch_a
    - branch_b
    - branch_c
```

### `join` — wait for all parallel branches

Waits for all incoming branches to complete before continuing.

```yaml
- id: fan_in
  type: join
  max_timeout: 300     # fail if branches don't finish within N seconds (default: unlimited)
```

### `switch` — conditional routing

Routes to one of several targets based on a condition expression. Conditions are evaluated in order; the first truthy condition wins. `when: null` is an unconditional default.

```yaml
- id: router
  type: switch
  routes:
    - when: "score > 4 and status != 'skip'"   # Python expression; state vars in scope
      next: high_priority
    - when: approved                            # simple bool state key
      next: standard_path
    - when: null                               # default fallback
      next: low_priority
```

**Expression syntax**: any Python expression using state variables. `&&` / `||` / `===` / `!==` are accepted as JS aliases and rewritten to Python equivalents. Available builtins: `len`, `str`, `int`, `float`, `bool`, `abs`, `min`, `max`, `sum`, `round`, `any`, `all`, `sorted`, `isinstance`.

### `python` — Python script

Either inline `code` or a script from the library (`script_id`). When both are
present the library copy wins, so a step always runs what the library holds.

```yaml
- id: transform
  type: python
  code: |
    output = state["items"][0]["value"]
  output_key: result

- id: normalize
  type: python
  script_id: normalize-payload    # from the script library
  sandbox: true                   # default — isolated execution
  sandbox_runtime: local          # local | docker | k8s
  sandbox_image: python:3.12-slim # docker / k8s only
  timeout_seconds: 60
  output_key: normalized
```

**Sandbox.** Enabled by default: the script runs in an isolated interpreter with
no access to the backend's env vars, tools, bash, installed libraries or system
dependencies. `state` arrives as JSON and the value assigned to `output` is
returned the same way, so both must be JSON-serialisable.

| runtime  | isolation |
|----------|-----------|
| `local`  | child `python -I -S` process: empty environment, no site-packages, throw-away cwd, CPU/memory rlimits, wall-clock timeout. Process-level only — no kernel namespaces. |
| `docker` | throw-away container: networking disabled, read-only rootfs, no inherited env, memory limit, all capabilities dropped. |
| `k8s`    | one-shot `Never`-restart pod: service-account token unmounted, read-only rootfs, resource limits, deleted after the run. |

Set `sandbox: false` to keep the legacy behaviour — `exec` inside the backend
process, with full access to its environment and libraries. Use it only for
trusted infrastructure code.

**`local` is not a security boundary.** It clears the environment and filters
imports, which stops an honest script from reaching the backend by accident, but
the child process shares the pod: a script can read the mounted service-account
token with a plain `open()`. Only `docker` and `k8s` put a kernel boundary in the
way. So when `AUTH_ENFORCE_PERMISSIONS` is on, a python step needs ADMIN unless
it names `sandbox_runtime: docker` or `k8s` — `sandbox: false`, `local` and an
unset runtime are all the same privilege, and gating only the first would leave
ADMIN a formality that any WRITE holder walks around by leaving a field unset.

Sandbox defaults come from `SCRIPT_SANDBOX_IMAGE`, `SCRIPT_SANDBOX_TIMEOUT` and
`SCRIPT_SANDBOX_MEMORY_MB`; a step can override the first two.

### Script library

Reusable scripts live in MongoDB (`script_definitions`) and are referenced by
`script_id`. Each has a name, a description and the code. Saving from a Python
node is a save-by-name: an existing name answers `409` so the UI can ask before
overwriting.

**Inline bodies are captured on save.** A `python` step saved with inline `code`
and no `script_id` gets a library entry, and the step is rewritten to point at
it — whether the save came from the UI, the REST API, the chat assistant or the
management MCP server. Code an agent writes is therefore findable and editable
in one place instead of buried in one workflow's definition. The library id is
`<workflow_id>-<step_id>`, so re-saving updates that one entry and two workflows
with a `transform` step never fight over the same document. The inline `code`
stays on the step as the body the node shows when the library is unreachable;
`script_id` is what runs.

### `cron` — scheduled trigger

Entry-point step. Registers a cron job; each firing creates a new run.

```yaml
- id: trigger
  type: cron
  schedule: "0 9 * * 1-5"             # 5-field UTC cron
  request_template: "Daily run on {date}"
```

### `http` — webhook trigger

Entry-point step. Listens at `POST /api/v1/webhooks/{workflow-id}`. The request body is stored under `output_key`.

```yaml
- id: trigger
  type: http
  output_key: webhook_data
```

Incoming requests must include an `X-Webhook-Signature` header (HMAC-SHA256 of the body, keyed with `WEBHOOK_SECRET`).

### `pubsub` — Google Cloud Pub/Sub trigger

Entry-point step. The backend subscribes to the topic on startup (and whenever the workflow is saved); each message that matches `schema` creates a new run, with the decoded message body stored under `output_key` and delivery metadata — topic, subscription, message id, publish time, attributes — under `trigger_info`.

```yaml
- id: trigger
  type: pubsub
  topic: orders                       # short name, or projects/<p>/topics/orders
  output_key: event                   # downstream steps template {event.order_id}
  schema:                             # optional; non-matching events never start a run
    type: object
    required: [order_id]
    properties:
      order_id: { type: string }
```

Instead of repeating topic and schema, a step can point at a pre-configured [event](#events); step fields override whatever the event sets:

```yaml
- id: trigger
  type: pubsub
  event: orders-events                # supplies topic, schema and subscription
  output_key: event
```

`datasource:` is the pre-events spelling of `event:` and still resolves, so workflows written before events existed keep working.

`subscription` may name an existing subscription to pull from. When it is left out, the backend creates one (`{PUBSUB_SUBSCRIPTION_PREFIX}{workflow-id}-{step-id}`) on first use and saves it back into the events — as an update to the event the step named, or as a new `pubsub-<topic>` event — so the next workflow can reuse it instead of creating another.

**Several workflows on one topic.** Every workflow that triggers on a topic gets every event, whichever way it is configured:

- Steps without a `subscription` each get their own, which is Pub/Sub's own fan-out.
- Steps that name the *same* subscription (typically by sharing an event) are served by a single streaming pull with several consumers behind it, and one arriving message starts a run for each of them. Opening one pull per step would make them compete, and Pub/Sub would hand each event to only one workflow. Each consumer's own `schema` still applies: a workflow whose schema does not match the event simply does not run.
- Backend replicas do compete on purpose: several replicas pull the same subscription, so an event starts one run cluster-wide rather than one per pod.

**When a trigger goes away.** Saving a workflow syncs its registrations, so deleting a `pubsub` node (or the whole workflow) stops the pull for it. Once a subscription has no trigger step left anywhere, the backend also deletes it — but only if it created it: a subscription named by a step or event is never removed. Set `PUBSUB_DELETE_ORPHANED_SUBSCRIPTIONS=false` to keep them. Shutdown never deletes anything: events published while the backend is down are waiting on restart.

Requires `PUBSUB_ENABLED=true` and `PUBSUB_PROJECT_ID` (see the configuration table); the service account needs `roles/pubsub.subscriber`, plus `roles/pubsub.editor` if the backend is to create subscriptions itself. Messages that fail schema validation are acknowledged and dropped so they cannot be redelivered forever — set `PUBSUB_DROP_INVALID_MESSAGES=false` to nack them instead and let topic-level retry or dead-lettering handle them.

---

## Agents

Agents are registered persistent definitions that `langgraph-agent` / `claude-agent` steps look up by `agent_id`. Each definition stores the runtime type, Docker image, and the `agent_input` dict (system prompt, model, tools, etc.) forwarded to the agent on every run.

**Runtimes.** All three run the same agent — [pi-cloud-agent](https://github.com/comtihon/pi-cloud-agent), the pi coding agent behind an HTTP server — so a workflow behaves the same on a laptop as in a cluster. Only the packaging differs:

| runtime  | how the agent is started | configured by |
|----------|--------------------------|---------------|
| `local`  | child process of the backend on a free localhost port | `LOCAL_AGENT_DIR` (a pi-cloud-agent checkout), `LOCAL_AGENT_COMMAND` |
| `docker` | container, random host port | `image` on the definition, `DOCKER_REGISTRY_*` for private registries |
| `k8s`    | Helm release in `AGENT_NAMESPACE` | `helm_chart` / `helm_values`, `AGENT_SERVICE_ACCOUNT` |

`local` needs `LOCAL_AGENT_DIR` set — there is no in-process agent to fall back on, and a step whose runtime resolves to `local` without it fails with a configuration error rather than running something different from production. Definitions written before this — `default_runtime: local` was the old inline LangGraph agent — need switching to `k8s`.

### Two images: `slim` and `full`

The backend ships as two images built from the same `Dockerfile`:

| tag | contains | `local` runtime |
|---|---|---|
| `ghcr.io/comtihon/ai-agents-carrier:latest` / `:vX.Y.Z` | the backend alone | unavailable |
| `ghcr.io/comtihon/ai-agents-carrier:latest-full` / `:vX.Y.Z-full` | the backend plus [pi-cloud-agent](https://github.com/comtihon/pi-cloud-agent) and a Node runtime | ready — `LOCAL_AGENT_DIR` is preset to `/opt/pi-cloud-agent` |

The unsuffixed tags stay slim, so an existing deployment keeps resolving to the
image it always did; moving to `full` is a deliberate change of tag.

`full` copies the agent out of the published `ghcr.io/comtihon/pi-cloud-agent`
image rather than rebuilding it, so the two cannot drift: one image remains the
single definition of what the agent is. Pin a different one with
`--build-arg PI_AGENT_IMAGE=ghcr.io/comtihon/pi-cloud-agent:v0.1.5`.

```bash
docker build -t carrier:slim .                  # slim is the default target
docker build -t carrier:full --target full .
```

**What `full` costs you.** The agent stops being isolated. On `docker` and `k8s`
it gets its own container, its own service account and its own filesystem; run
as `local` it is a child process of the backend, sharing the backend's pod,
filesystem, environment and service-account token. Agents run model-authored
code, so that is a real reduction in blast-radius containment — the separate
`langgraph-agent` identity exists precisely to avoid it. Use `full` where that
trade is worth not needing a registry, a chart, or cluster permissions at run
time; prefer `k8s` or `docker` otherwise.

`full` carries Node, the agent and its `git` / `ripgrep` / `jq` helpers. It does
not carry the agent image's heavier tooling (`gcloud`, `kubectl`, `gh`, `uv`,
`semble`, `graphify`), so an agent granted one of those through `AGENT_TOOLS`
needs the `docker` or `k8s` runtime.

```yaml
# Example agent definition (managed via API or copilot_ui)
id: researcher
name: Researcher
default_runtime: docker
image: europe-west4-docker.pkg.dev/myorg/registry/langgraph-agent:0.1.6
agent_input:
  system_prompt: "You are a research agent with access to bash and code-search tools."
  model: claude-opus-4-7
  max_tokens: 8096
health_timeout: 300     # seconds to wait for /health after container starts
```

**Agent HTTP protocol** — the backend calls the agent container:

```
POST /start     {run_id, input, callback_url, agent_config}  → 202 Accepted
GET  /health    → 200 when ready
POST /terminate → graceful shutdown
```

The agent calls back to the backend:

```
POST {callback_url}/api/v1/runs/{run_id}/agent/output    {output: {...}}
POST {callback_url}/api/v1/runs/{run_id}/agent/progress  {message: str}
POST {callback_url}/api/v1/runs/{run_id}/agent/question  {question, options?}
GET  {callback_url}/api/v1/runs/{run_id}/agent/input     (long-poll for answer)
```

Progress messages starting with `__token__:` carry live token counts: `__token__:{"input_tokens":N,"output_tokens":N,"total_tokens":N}` — the backend stores these in `_live_token_usage` and surfaces them in the run response for real-time display.

**Credential forwarding** — any env var in the backend matching a credential suffix (`_API_KEY`, `_TOKEN`, `_JSON`, `_SECRET`, `_CREDENTIALS`) is automatically available to forward to agent containers via the `env_vars` step config. The list is exposed at `GET /api/v1/llm/config/keys` (names only, no values).

---

## Data Sources

A data source is a declarative definition of a remote HTTP or GraphQL API plus the named operations that can be invoked on it. Definitions live in MongoDB (`data_source_definitions`) and are managed through `/api/v1/datasources`, the chat assistant, or copilot_ui.

```yaml
id: github
name: GitHub
kind: http                       # http | graphql
base_url: https://api.github.com
auth:                            # secret values stored in the definition; API responses redact them
  type: bearer                   # bearer | basic | header | none
  token: <token value>           # basic: username/password; header: header_name/value
default_headers:
  Accept: application/vnd.github+json
timeout_seconds: 30
retries:
  attempts: 3
  backoff: 0.5
cache:
  ttl_seconds: 60                # 0 disables caching
operations:
  - name: list_repos
    method: GET
    path: /users/{params.owner}/repos
    params:
      - name: owner
        type: string
        required: true
    mapping: "[].{name: name, full_name: full_name}"   # JMESPath
    paginate:
      type: page                 # cursor | page | offset
      param: page
      max_pages: 5
  - name: repo_languages
    method: GET
    path: /repos/{list_repos.full_name}/languages      # DAG reference → fan-out
```

**DAG references.** Templates resolve `{params.<name>}` from the caller's inputs and `{<operation>.<field.path>}` from another operation of the same source. Field paths are JMESPath. The referenced operations form the dependency closure of the call: they run level by level with `asyncio.gather` and are memoised per request, so a diamond DAG calls each upstream exactly once. Unknown references, self-references and cycles are rejected at save time with HTTP 422.

**Fan-out.** When a referenced upstream result is an array, the dependent operation runs once per element (bounded by an `asyncio.Semaphore`) and returns `[{"<field>": <element value>, "result": <call result>}, ...]`. An operation may bind at most one array upstream; a second one raises at runtime.

**Pagination** runs before dependents see a result: `cursor` follows `cursor_path` (JMESPath) and sends it back as `param`; `page` sends a 1-based page number; `offset` sends the number of items already fetched. Looping stops on an empty page, a missing cursor, or `max_pages`. List pages are concatenated. When an operation has no `mapping`, set `items_path` (JMESPath to the items array in the raw page response) so pagination can detect an empty page and concatenate results — without it, a dict-shaped response never looks "empty" and looping silently runs to `max_pages`.

**Cache** is in-memory and keyed by `(source id, operation, resolved params, resolved upstream/fan-out refs)` — so an operation whose template binds an upstream operation's result gets a distinct cache entry per upstream value — with a monotonic-clock TTL. Expired entries are dropped on access and swept opportunistically once the cache grows past a size threshold. **Auth** env vars are read at execution time and merged over `default_headers`.

**MCP endpoint.** Every `<source> x <operation>` pair is exposed as an MCP tool `ds_<source_id>_<operation>` on a streamable-http server mounted at `/mcp/datasources`. The tool's input schema contains the operation's declared `params` only — upstream dependencies stay invisible. The backend registers this as the `datasources` MCP integration, so agents pick the tools up automatically; the server is connected lazily in the background after startup and re-published on every CRUD change (`MCP_DATASOURCES_ENABLED`, `MCP_DATASOURCES_URL`, `MCP_DATASOURCES_API_KEY`).

**Workflow step.** The same executor is reachable from a workflow:

```yaml
- id: fetch_repos
  type: data_source
  source: github
  operation: list_repos
  params:
    owner: "{repo_owner}"
  output_key: repos
```

Errors are captured as `{"error": "..."}` under `output_key` so the next step can react.

---

## Events

An event is a Google Cloud Pub/Sub topic a workflow can be triggered by. Definitions live in MongoDB (`event_definitions`) and are managed through `/api/v1/events`, the chat assistant, or copilot_ui.

```yaml
id: orders-events
name: Order events
description: Shop orders
topic: orders                    # short name, or projects/<p>/topics/orders
subscription: ""                 # blank: created on first use and saved back here
project_id: ""                   # blank: the backend-wide PUBSUB_PROJECT_ID
event_schema:                    # blank: every message starts a run
  type: object
  required: [order_id]
  properties:
    order_id: { type: string }
```

An event carries no base URL, credentials or operations — it is not called, it is subscribed to. `pubsub` trigger steps point at one with `event: <id>` instead of repeating the topic, schema and subscription in every workflow; see [`pubsub` — Google Cloud Pub/Sub trigger](#pubsub--google-cloud-pubsub-trigger).

**Names are checked.** `POST /api/v1/events` answers `409` when another event already uses the name, so the UI can offer to overwrite that one instead of quietly creating a second event nobody can tell apart. `PUT` is the overwrite.

**Events used to be data sources** with `kind: "pubsub"`. Creating one that way is now a `422`; `scripts/migrations/2026-08-19_move_pubsub_datasources_to_events.py` moves any that already exist (ids are preserved, so `datasource:` references keep resolving). Run it with `--apply` after deploying.

---

## API

```
# Workflows
GET    /api/v1/workflows                      list workflows
POST   /api/v1/workflows                      create workflow
GET    /api/v1/workflows/{id}                 get workflow
PUT    /api/v1/workflows/{id}                 update workflow
DELETE /api/v1/workflows/{id}                 delete workflow

# Runs
POST   /api/v1/workflows/runs                 start a run
GET    /api/v1/workflows/runs/{id}            get run status
GET    /api/v1/workflows/runs/{id}/trace      get LangSmith / token trace
POST   /api/v1/workflows/runs/{id}/approve    approve a paused run
POST   /api/v1/workflows/runs/{id}/reject     reject a paused run

# Agents
GET    /api/v1/agents                         list registered agents
POST   /api/v1/agents                         register an agent
GET    /api/v1/agents/{id}                    get agent definition
PUT    /api/v1/agents/{id}                    update agent definition
DELETE /api/v1/agents/{id}                    delete agent definition

# Scripts (Python script library)
GET    /api/v1/scripts                        list scripts
POST   /api/v1/scripts                        create script (409 on name clash; `overwrite: true` to replace)
GET    /api/v1/scripts/{id}                   get script
PUT    /api/v1/scripts/{id}                   update script
DELETE /api/v1/scripts/{id}                   delete script

# Data sources
GET    /api/v1/datasources                    list data sources
POST   /api/v1/datasources                    create data source
GET    /api/v1/datasources/{id}               get data source definition
PUT    /api/v1/datasources/{id}               update data source definition
DELETE /api/v1/datasources/{id}               delete data source definition
GET    /api/v1/events                         list events
POST   /api/v1/events                         create event (409 on a duplicate name)
GET    /api/v1/events/{id}                    get event definition
PUT    /api/v1/events/{id}                    update event definition
DELETE /api/v1/events/{id}                    delete event definition
ALL    /mcp/datasources                       MCP (streamable-http) tools for all operations

# Agent callbacks (called by running agent containers)
POST   /api/v1/runs/{id}/agent/output         deliver result, resume run
POST   /api/v1/runs/{id}/agent/progress       send progress / token update
POST   /api/v1/runs/{id}/agent/question       ask a clarifying question
GET    /api/v1/runs/{id}/agent/input          long-poll for answer to question
POST   /api/v1/runs/{id}/agent/reply          submit answer (from UI)

# Config
GET    /api/v1/llm/config/keys                list forwardable credential key names
GET    /api/v1/llm/providers                  list configured LLM providers

# Triggers
POST   /api/v1/webhooks/{workflow-id}         HTTP webhook trigger

# Approval callbacks (no auth)
POST   /api/v1/callbacks/{run-id}/approve
POST   /api/v1/callbacks/{run-id}/reject
```

---

## Add-ons

### Slack approvals

Set `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`, and `SLACK_APPROVALS_CHANNEL` to send approval requests to a Slack channel. The `human_approval` step's `notify` block targets Slack via webhook or the bot token.

### LangSmith tracing

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to send all LLM calls to LangSmith. The LangSmith run URL is included in the run trace response (`GET /runs/{id}/trace`).

### OpenHands code execution

Set `OPENHANDS_BASE_URL` and `OPENHANDS_API_KEY` (set `OPENHANDS_MOCK_MODE=false`) to enable the `execute` step type, which delegates coding tasks to an OpenHands instance.

### Custom LLM providers

`LLM_INTEGRATIONS` accepts a JSON array of OpenAI-compatible endpoints:

```bash
LLM_INTEGRATIONS='[{"name":"ollama","base_url":"http://localhost:11434/v1","default_model":"llama3","api_key_env":"OLLAMA_API_KEY"}]'
LLM_PROVIDER=ollama
```

Any entry can be referenced by name in workflow steps via `llm_provider: ollama`.

---

## Deployment (Helm)

The chart in `helm/` renders the deployment, service, ingress, config/secret env wiring and a namespaced RBAC Role that lets the backend manage agent pods.

### ServiceAccount and cloud identity

By default the chart creates a ServiceAccount named after the release and runs the pod as it. That account carries no cloud identity, so anything reaching a cloud API from inside the pod (`gcloud`, Cloud Logging, Pub/Sub) has none either. Two ways to give it one:

```yaml
# Annotate the chart's own account — GKE Workload Identity needs a matching
# workloadIdentityUser binding for serviceAccount:<project>.svc.id.goog[<ns>/<name>]
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account: my-gsa@my-project.iam.gserviceaccount.com
```

```yaml
# Or run under an account managed elsewhere (terraform, for instance). The
# chart then creates no ServiceAccount, and both the deployment and the
# RoleBinding for the agent-manager Role point at this name.
serviceAccount:
  create: false
  name: langgraph-backend
```

Either way the chart's Role and RoleBinding keep their release-derived names, so the agent-manager permissions follow whichever account the pod actually uses.
