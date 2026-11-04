# Description

Data links: ?

## Environment Server sessions

The server accepts typed `ResourcesSeedSessionRequest` bodies from `single_agent_turn`,
alongside the existing flat seed request. A typed seed returns `SandboxAccess` for the
task sandbox and its working directory. Resources owns that sandbox; the agent borrows
it and must close before verification. The Environment Server closes Resources after
verification or an interrupted episode.

Identical seed retries reuse the sandbox. Conflicting seeds and seeds after close are
rejected. Typed verification captures the patch before releasing the task sandbox, so a
failed stop can retry without losing the patch. Identical verification retries reuse the
completed result. Seed, verify and close serialize per session. Close retains a failed
task or verifier teardown for retry; cleanup is shielded from request cancellation and
bounded by `session_cleanup_timeout_seconds` (120 seconds by default).

Session records are process-local and use Gym's session routing. Successful close fences
late seeds for `session_close_retry_window_seconds`. Shutdown attempts remaining cleanup;
configure a provider-enforced `sandbox_config.ttl_s` for abandoned sessions. This does not
change the benchmark prompt, patch extraction rules or scoring protocol.

# Quickstart
## Apply golden patches
### Start resources server
```bash
gym env start \
    --config resources_servers/swebench/configs/swebench.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    +swebench_resources_server.resources_servers.swebench.is_verifying_golden_patch=true
```

### One golden patch smoke test
In a separate terminal:
```bash
# If you haven't already, prepare SWE Bench benchmark data
gym eval prepare --config benchmarks/swebench/verified/opencode.yaml

python resources_servers/swebench/client.py \
    +benchmark_jsonl=benchmarks/swebench/data/swebench_verified_benchmark.jsonl
```

### Full SWE Verified golden patch smoke test
In a separate terminal:
```bash
python resources_servers/swebench/apply_golden_patch.py \
    +benchmark_jsonl=benchmarks/swebench/data/swebench_verified_benchmark.jsonl \
    +limit=...  # No limit for full samples
```


## Expected golden patch resolve rates
SWE Bench Verified
```bash
Resolved: 492 / 500 (98.40%): ... [19:29, 258.00s/it]
```

SWE Bench Multilingual
```bash
Resolved: 300 / 300 (100.00%): [26:31<00:00,  5.30s/it]
```

SWE Bench Pro


# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
