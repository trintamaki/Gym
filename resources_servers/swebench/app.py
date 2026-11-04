# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import logging
import re
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from shlex import quote
from shutil import rmtree
from time import monotonic, time
from traceback import format_exc
from typing import Any, Dict, Optional, Tuple

from anyio import fail_after
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, PositiveFloat, ValidationError
from swebench.harness.run_evaluation import make_test_spec
from swebench.harness.test_spec.test_spec import LATEST, TestSpec

from docker.models.containers import ExecResult
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.core import (
    VERIFIER_ERROR,
    FailureBoundary,
    NgFailureEvent,
    current_verifier_invocation_id,
    observe_failure,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxAccess,
    SandboxResources,
    SandboxSpec,
    cpu_cap_env,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.trajectory import SandboxObservation
from nemo_gym.utils import bounded_text
from resources_servers.swebench.swebench_patches import (
    patch_swebench_multilingual_golden_patch_pass,
    patch_swebench_multilingual_log_parsing,
    patch_swebench_multilingual_sandbox,
    patch_swebench_verified_env_vars,
    run_instance,
)


# The longest failure_reason a failure event accepts.
_MAX_FAILURE_REASON_CHARS = 2000
LOGGER = logging.getLogger(__name__)


class SwebenchResourcesServerConfig(BaseResourcesServerConfig):
    is_verifying_golden_patch: bool = False
    apply_anti_cheating: bool = True
    # Write the Maven Central mirror to ~/.m2/settings.xml in the eval sandbox. Turn off for providers
    # that mount their own read-only settings.xml, e.g. one that already mirrors Maven Central.
    overwrite_maven_settings: bool = True

    evaluation_timeout: Optional[int] = None

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]

    clear_swebench_debug_logs: bool = True
    session_cleanup_timeout_seconds: PositiveFloat = 120

    def model_post_init(self, context: Any, /) -> None:
        if self.is_verifying_golden_patch and self.clear_swebench_debug_logs:
            print("Turning off logs clear since `is_verifying_golden_patch=true`")
            self.clear_swebench_debug_logs = False


class SWEBenchInstanceRequest(BaseModel):
    # See https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
    # See swebench.harness.run_evaluation.TestSpec https://github.com/SWE-bench/SWE-bench/blob/f7bbbb2ccdf479001d6467c9e34af59e44a840f9/swebench/harness/test_spec/test_spec.py#L28
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    # These are JSON strings.
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    environment_setup_commit: str
    difficulty: str
    subset: str
    split: str


class SWEBenchVerifyRequest(SWEBenchInstanceRequest, BaseVerifyRequest):
    pass


class SWEBenchVerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool
    resolved: bool

    # Misc metrics
    eval_sandbox_start_time_taken: float
    patch_verification_time_taken: float

    instance_id: str
    test_output: str
    model_patch: Optional[str]
    # sha256 of the patch that was verified (UTF-8), so a later re-verification can prove it grades the same patch;
    # None when the patch could not be extracted from the agent's sandbox.
    model_patch_sha256: Optional[str] = None

    log_dir: str

    verifier_sandbox_observation: Optional[SandboxObservation] = Field(
        default=None, exclude_if=lambda value: value is None
    )


# @bxyu-nvidia: This is a wrapper that can be passed directly to a very lightly modified version of `run_instance`
# The method is almost identical to the original, just with async awaits rather than sync.
# See resources_servers/swebench/swebench_patches.py
class DockerContainer(BaseModel):
    id: str
    instance_id: str

    _inner_container: AsyncSandbox
    _eval_return_code: Optional[int] = None
    _sandbox_error_type: Optional[str] = None

    async def exec_run(
        self,
        command: str,
        workdir: Optional[str] = None,
        user: Optional[str] = None,
    ) -> ExecResult:
        try:
            res = await self._inner_container.exec(
                command=command,
                cwd=workdir,
                user=user,
            )
        except Exception as exc:
            self._sandbox_error_type = self._sandbox_error_type or type(exc).__name__
            print(f"Failed to exec in SWE Bench for {self.instance_id}", format_exc(), file=sys.stderr)
            raise
        if res.error_type is not None:
            self._sandbox_error_type = self._sandbox_error_type or res.error_type

        return ExecResult(
            exit_code=res.return_code,
            # @bxyu-nvidia: This is not entirely 1:1, but it works for the purposes of this patch.
            # The sandbox API returns None for an empty stream (docker-py returned bytes).
            output=((res.stdout or "") + (res.stderr or "")).encode(),
        )

    async def exec_run_with_timeout(self, command: str, timeout: int) -> Tuple[str, bool, float]:
        # Returns: test_output: str, timed_out: bool, total_runtime: float
        start_time = time()
        try:
            res = await self._inner_container.exec(
                command=command,
                # AsyncSandbox.exec takes timeout_s, not docker-py's timeout.
                timeout_s=timeout,
            )
            self._eval_return_code = res.return_code if res.error_type is None else None
            if res.error_type is not None:
                self._sandbox_error_type = res.error_type
            timed_out = False

            stdout = res.stdout or ""
            stderr = res.stderr or ""

            maybe_test_output = patch_swebench_multilingual_log_parsing(stdout, stderr, self.instance_id)
            test_output = maybe_test_output or (stdout + stderr)
        except TimeoutError:
            # Gym Sandbox API will throw a timeout error on actual timeout.
            timed_out = True
            self._sandbox_error_type = "TimeoutError"
            test_output = ""
        except Exception as exc:
            self._sandbox_error_type = type(exc).__name__
            print(
                f"Failed to exec_run_with_timeout in SWE Bench for {self.instance_id}", format_exc(), file=sys.stderr
            )
            raise

        return (test_output, timed_out, time() - start_time)

    async def copy(self, src: Path, dest: Path) -> None:
        if "eval.sh" in str(src):
            data = src.read_text()
            src.write_text(patch_swebench_multilingual_golden_patch_pass(data, self.instance_id))

        try:
            await self._inner_container.upload(local_path=src, remote_path=str(dest))
        except Exception as exc:
            self._sandbox_error_type = self._sandbox_error_type or type(exc).__name__
            print(
                f"Failed to upload to verification sandbox in SWE Bench for {self.instance_id}",
                format_exc(),
                file=sys.stderr,
            )
            raise

    async def cleanup(self) -> None:
        try:
            await self._inner_container.stop()
        except Exception as exc:
            self._sandbox_error_type = self._sandbox_error_type or type(exc).__name__
            print("Failed to stop verification sandbox", format_exc(), file=sys.stderr)

    def observation(self, *, wall_time_s: float, evaluation_completed: bool) -> SandboxObservation:
        handle = self._inner_container._handle
        normalized_error = self._sandbox_error_type.lower() if isinstance(self._sandbox_error_type, str) else ""
        if "timeout" in normalized_error:
            outcome = "timeout"
        elif self._sandbox_error_type is not None:
            outcome = "sandbox_error"
        elif evaluation_completed:
            outcome = "completed"
        else:
            outcome = "failed"

        return SandboxObservation(
            role="verifier",
            provider=handle.provider_name if handle is not None else None,
            sandbox_id=handle.sandbox_id if handle is not None else None,
            outcome=outcome,
            exit_code=self._eval_return_code,
            wall_time_s=wall_time_s,
            error_type=self._sandbox_error_type,
        )


# TODO @bxyu-nvidia: Eventually once the sandbox server infra is ready, these seed_session types need to upgrade to pass a sandbox spec.
# They can possibly even omitted once this graduates to core infra.
class SWEBenchSeedSessionRequest(SWEBenchInstanceRequest, BaseSeedSessionRequest):
    sandbox_spec: Optional[Dict[str, Any]] = None


class SWEBenchSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str  # @bxyu-nvidia: Just a plain string URI for now for OpenSandbox backend.


# The agent's working tree is graded by the diff it leaves behind. A plain
# `git diff` shows only unstaged changes to tracked files, so a fix the agent
# staged (`git add`) or committed scored 0 with an empty patch: 14% of Claude
# Code rollouts on SWE-bench Verified did one of the two (2026-10-19). Diff the
# index against the commit the session was seeded at instead, after staging
# the tracked files' changes, so staged and committed fixes count. Untracked
# files are left out on purpose: `git add -A` swept a generated Sphinx `_build`
# tree (1.1 MB of jquery) into one patch; a new file the agent committed is
# still included. Binary sections are dropped, since `git apply` cannot take
# them without binary data.
def patch_extraction_command(workdir: str, seed_head: str) -> str:
    return (
        f"cd {quote(workdir)} && git add -u >/dev/null 2>&1; "
        f"cd {quote(workdir)} && git --no-pager diff --cached {quote(seed_head)}"
    )


def strip_binary_diffs(patch: str) -> str:
    """Remove `diff --git` sections for binary files (git prints "Binary files ... differ" without the data)."""
    if not patch:
        return patch
    sections = re.split(r"(?m)^(?=diff --git )", patch)
    return "".join(sec for sec in sections if not re.search(r"(?m)^Binary files .* differ$", sec))


@dataclass
class _TaskSession:
    seed: ResourcesSeedSessionRequest
    task: SWEBenchInstanceRequest
    access: SandboxAccess | None = None
    model_patch: str | None = None
    verify_request: SWEBenchVerifyRequest | None = None
    verification: SWEBenchVerifyResponse | None = None


class SwebenchResourcesServer(SimpleResourcesServer):
    config: SwebenchResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self._session_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._session_id_to_seed_head: Dict[str, str] = dict()
        self._task_sessions: dict[str, _TaskSession] = {}
        self._verifier_sandboxes: dict[str, AsyncSandbox] = {}
        self._fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def _stop_sandbox(self, sandboxes: dict[str, AsyncSandbox], session_id: str) -> None:
        sandbox = sandboxes.get(session_id)
        if sandbox is not None:
            # A disconnect can cancel the request's AnyIO scope. Keep the owner handle until stop succeeds.
            with fail_after(self.config.session_cleanup_timeout_seconds, shield=True):
                await sandbox.stop()
            sandboxes.pop(session_id)

    async def _release_session(self, session_id: str) -> None:
        await self._stop_sandbox(self._session_id_to_sandbox, session_id)
        self._session_id_to_seed_head.pop(session_id, None)
        await self._stop_sandbox(self._verifier_sandboxes, session_id)

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        """Release the caller-assigned session, retaining failed teardown for a close retry."""
        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            closed = self._fence.closed_episode(session_id)
            state = self._task_sessions.get(session_id)
            expected_episode = closed if closed is not None else state.seed.episode_id if state is not None else None
            if expected_episode is not None and expected_episode != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the resources session")
            await self._release_session(session_id)
            self._task_sessions.pop(session_id, None)
            self._fence.mark_closed(session_id, body.episode_id)
            if request.session.get(SESSION_ID_KEY) == session_id:
                request.session.pop(SESSION_ID_KEY, None)
            return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def shutdown(self) -> None:
        """Release task and verifier sandboxes left by interrupted episodes."""
        session_ids = self._task_sessions.keys() | self._session_id_to_sandbox.keys() | self._verifier_sandboxes.keys()
        for session_id in session_ids:
            async with self._fence.locked(session_id):
                try:
                    await self._release_session(session_id)
                    self._task_sessions.pop(session_id, None)
                except Exception:
                    LOGGER.exception("Failed to stop abandoned SWE-bench session %s", session_id)

    async def _create_sandbox(
        self,
        test_spec: TestSpec,
        *,
        task_session_id: str | None = None,
        verifier_session_id: str | None = None,
    ) -> AsyncSandbox:
        # TODO @bxyu-nvidia: Refactor this after Hemil's swap from Python dataclass to Pydantic BaseModel
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
        resources = dict(self.config.sandbox_config.get("resources", {}))

        # Derive from the final resources map (after the multilingual bump);
        # explicit sandbox_config.env keys win over the derived caps.
        sandbox_resources = SandboxResources.from_mapping(resources)
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            env = cpu_cap_env(sandbox_resources.cpu) | env

        patch_swebench_verified_env_vars(test_spec.repo, env)

        eval_sandbox_spec = SandboxSpec(
            image=test_spec.instance_image_key,
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # Default to container's WORKDIR
            env=env,
            files=dict(),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
                "instance_id": test_spec.instance_id[:63],
            },
            resources=sandbox_resources,
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )
        eval_sandbox = AsyncSandbox(resolved_sandbox_provider)
        # Retain ownership even if start_with_setup raises after its cleanup attempt also failed.
        if task_session_id is not None:
            self._session_id_to_sandbox[task_session_id] = eval_sandbox
        if verifier_session_id is not None:
            self._verifier_sandboxes[verifier_session_id] = eval_sandbox

        async def _run_setup(sandbox: AsyncSandbox) -> None:
            await patch_swebench_multilingual_sandbox(
                test_spec.repo,
                test_spec.instance_id,
                sandbox,
                overwrite_maven_settings=self.config.overwrite_maven_settings,
            )

        await eval_sandbox.start_with_setup(eval_sandbox_spec, _run_setup)

        return eval_sandbox

    def _make_test_spec(self, body: SWEBenchInstanceRequest) -> TestSpec:
        return make_test_spec(
            # This accepts a SWEbenchInstance which is identically our body.
            body.model_dump(),
            namespace="swebench",  # Dockerhub namespace
            instance_image_tag=LATEST,
            env_image_tag=LATEST,
        )

    async def seed_session(
        self, request: Request, body: SWEBenchSeedSessionRequest | ResourcesSeedSessionRequest
    ) -> SWEBenchSeedSessionResponse | ResourcesSeedSessionResponse:
        """Prepare a task sandbox for either a flat caller or an Environment Server episode."""
        if not isinstance(body, ResourcesSeedSessionRequest):
            session_id = request.session[SESSION_ID_KEY]
            await self._seed_task_sandbox(session_id, body)
            return SWEBenchSeedSessionResponse(
                sandbox_handle=self._session_id_to_sandbox[session_id]._handle.sandbox_id
            )

        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            self._fence.raise_if_closed(session_id)
            state = self._task_sessions.get(session_id)
            if state is not None:
                if state.seed != body:
                    raise ResourcesSessionConflictError("resources_session_id is already bound to another seed")
                if state.verify_request is not None:
                    raise ResourcesSessionConflictError("The task sandbox has already entered verification")
            else:
                try:
                    task = SWEBenchInstanceRequest.model_validate(body.task_data)
                except ValidationError as error:
                    raise ResourcesSeedRejectedError(f"task_data is not a SWE-bench instance: {error}") from error
                # TaskId belongs to the caller's task envelope, not to a field inside its task_data.
                state = _TaskSession(seed=body.model_copy(deep=True), task=task)
                self._task_sessions[session_id] = state
            if state.access is None:
                try:
                    # Retry cleanup before retrying a seed whose setup or handoff failed.
                    await self._release_session(session_id)
                    workdir = await self._seed_task_sandbox(session_id, state.task, require_workdir=True)
                    state.access = SandboxAccess(
                        connection=DirectSandboxConnection(
                            provider_config_ref=self.config.sandbox_provider,
                            descriptor=await self._session_id_to_sandbox[session_id].serialize(scope="operate"),
                        ),
                        workdir=workdir,
                    )
                except BaseException:
                    try:
                        await self._release_session(session_id)
                        self._task_sessions.pop(session_id, None)
                    except Exception:
                        LOGGER.exception("Failed to clean up SWE-bench seed %s; close can retry", session_id)
                    raise
            # A refused seed must not replace the caller's cookie with another episode's session.
            request.session[SESSION_ID_KEY] = session_id
            return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=state.access)

    async def _seed_task_sandbox(
        self, session_id: str, body: SWEBenchInstanceRequest, *, require_workdir: bool = False
    ) -> str:
        test_spec = self._make_test_spec(body)
        eval_sandbox = await self._create_sandbox(test_spec, task_session_id=session_id if require_workdir else None)
        self._session_id_to_sandbox[session_id] = eval_sandbox

        wd = ""
        if require_workdir or self.config.apply_anti_cheating:
            result = await eval_sandbox.exec("pwd")
            wd = (result.stdout or "").strip()
            if require_workdir and (
                result.error_type is not None or result.return_code != 0 or not wd.startswith("/")
            ):
                raise RuntimeError("SWE-bench task sandbox did not report an absolute working directory")

        if self.config.apply_anti_cheating:
            # Remove the current Git repo's future history beyond the current commit to prevent the model from cheating.
            anti_cheat_setup_fpath = Path(__file__).parent / "anti_cheat_setup.sh"
            await eval_sandbox.upload(anti_cheat_setup_fpath, f"{wd}/anti_cheat_setup.sh")
            result = await eval_sandbox.exec(
                f"""git reset --hard && WORKING_DIRECTORY={wd} bash anti_cheat_setup.sh && rm anti_cheat_setup.sh"""
            )
            if result.return_code != 0:
                print(f"""Failed to setup anti-cheating for {test_spec.instance_id}. Return code: {result.return_code}
Stdout:
{result.stdout}
Stderr:
{result.stderr}""")

        # The commit to diff against at verify time: HEAD once the sandbox is
        # ready (anti-cheating rewrites refs but not the tree), falling back to
        # the task's base commit if git is unavailable.
        head_result = await eval_sandbox.exec("git rev-parse HEAD")
        seed_head = (head_result.stdout or "").strip() if head_result.return_code == 0 else ""
        self._session_id_to_seed_head[session_id] = seed_head or test_spec.base_commit
        return wd

    async def verify(self, request: Request, body: SWEBenchVerifyRequest) -> SWEBenchVerifyResponse:
        """Grade the task sandbox after agent close; typed retries reuse the captured patch or verdict."""
        session_id = request.session[SESSION_ID_KEY]
        async with self._fence.locked(session_id):
            if self._fence.closed_episode(session_id) is not None:
                return _lost_session_response(body, "The SWE-bench resources session is closed")
            state = self._task_sessions.get(session_id)
            if state is None:
                if not self.config.is_verifying_golden_patch and session_id not in self._session_id_to_sandbox:
                    return _lost_session_response(body, "The SWE-bench task sandbox is missing or was never seeded")
                return await self._verify(request, body)
            if state.access is None:
                return _lost_session_response(body, "The SWE-bench task session did not finish seeding")
            if SWEBenchInstanceRequest.model_validate(body.model_dump()) != state.task:
                raise ResourcesSessionConflictError("Verification task does not match the seeded task")
            if state.verify_request is not None and body != state.verify_request:
                raise ResourcesSessionConflictError("Verification input does not match the first request")
            state.verify_request = body.model_copy(deep=True)
            if state.verification is not None:
                return state.verification
            if state.model_patch is None:
                if self.config.is_verifying_golden_patch:
                    state.model_patch = body.patch
                else:
                    sandbox = self._session_id_to_sandbox.get(session_id)
                    if sandbox is None:
                        return _lost_session_response(body, "The SWE-bench task sandbox is missing")
                    seed_head = self._session_id_to_seed_head[session_id]
                    result = await sandbox.exec(patch_extraction_command(state.access.workdir, seed_head))
                    if result.error_type is not None or result.return_code != 0:
                        raise RuntimeError("Failed to extract the SWE-bench model patch from the task sandbox")
                    state.model_patch = strip_binary_diffs(result.stdout or "")
            # Keep the patch and task handle if stop fails; a retry must never capture an empty replacement patch.
            await self._release_session(session_id)
            try:
                state.verification = await self._verify(request, body, captured_patch=state.model_patch)
            finally:
                try:
                    await self._stop_sandbox(self._verifier_sandboxes, session_id)
                except Exception:
                    LOGGER.exception("Failed to stop SWE-bench verifier %s; close can retry", session_id)
            return state.verification

    async def _verify(
        self, request: Request, body: SWEBenchVerifyRequest, *, captured_patch: str | None = None
    ) -> SWEBenchVerifyResponse:
        """
        Key requirements:
        1. Extract the model_patch from the input container
            Proposal
                1. Spinup a fresh container (need this anyways for running eval)
                2. pwd in the fresh container (defaults to WORKDIR)
                3. cd into WORKDIR in the input container
                4. extract the patch via git
            Notes
                1. DeepSWE expects the model to commit. That will go in the DeepSWE resources server and not this one.
        2. Docker Client - Make a mock client class here that wraps our sandbox client.
        3. Harnesses like OpenCode open a new terminal rather than reusing the existing one. Grab the environment variables and workdir from the outer terminal first, and then export/cd as appropriate in the new terminal
            Notes
                1. This is a harness-specific thing that the harness will handle across benchmarks.
        4. Interleaved thinking - Verify how is the harness behaving i.e. it has interleaved thinking or not and to force interleaved thinking unconditionally.
            Proposal
                1. For seeing what the harness is doing, use model call capture
                2. For forcing, we can add it in the Responses API model proxy i.e. save all the past requests/responses and populate as necessary.
        5. Restrict number of turns - same as interleaved thinking, we could add in Responses API model proxy
        """

        test_spec = self._make_test_spec(body)

        verifier_sandbox_lifecycle_started_at = monotonic()
        eval_sandbox = await self._create_sandbox(
            test_spec, verifier_session_id=request.session[SESSION_ID_KEY] if captured_patch is not None else None
        )
        if captured_patch is not None:
            self._verifier_sandboxes[request.session[SESSION_ID_KEY]] = eval_sandbox
        eval_sandbox_start_time_taken = monotonic() - verifier_sandbox_lifecycle_started_at

        model_patch = ""
        patch_extraction_failed = False
        if captured_patch is not None:
            model_patch = captured_patch
        elif self.config.is_verifying_golden_patch:
            model_patch = body.patch
        else:
            original_sandbox = self._session_id_to_sandbox.pop(request.session[SESSION_ID_KEY])
            try:
                original_workdir = (await eval_sandbox.exec("pwd")).stdout.strip()
                seed_head = (
                    self._session_id_to_seed_head.pop(request.session[SESSION_ID_KEY], None) or test_spec.base_commit
                )
                model_patch_result = await original_sandbox.exec(patch_extraction_command(original_workdir, seed_head))
                model_patch = strip_binary_diffs(model_patch_result.stdout or "")
            except:
                patch_extraction_failed = True
                print("Failed to extract patch from container", format_exc(), file=sys.stderr)
            try:
                await original_sandbox.stop()
            except:
                print("Failed to stop original sandbox", format_exc(), file=sys.stderr)

        run_id = request.session[SESSION_ID_KEY]
        mock_container = DockerContainer(id=run_id, instance_id=test_spec.instance_id)
        mock_container._inner_container = eval_sandbox

        # Res has 2 keys: completed (whether evaluation completed or not), resolved (whether the issue is resolved)
        start_time = time()
        res = await run_instance(
            test_spec=test_spec,
            pred={
                "instance_id": test_spec.instance_id,
                "model_patch": model_patch,
            },
            rm_image=False,
            force_rebuild=False,
            client=mock_container,
            run_id=run_id,
            timeout=self.config.evaluation_timeout,
            rewrite_reports=False,
        )
        patch_verification_time_taken = time() - start_time
        verifier_sandbox_wall_time_s = monotonic() - verifier_sandbox_lifecycle_started_at

        try:
            verifier_sandbox_observation = mock_container.observation(
                wall_time_s=verifier_sandbox_wall_time_s,
                evaluation_completed=res["completed"],
            )
        except Exception:
            verifier_sandbox_observation = None
            print("Failed to build verification sandbox observation", format_exc(), file=sys.stderr)

        evaluation_completed = res["completed"] and mock_container._sandbox_error_type is None
        # A patch the harness could not apply is the model's own graded result when the patch was collected intact
        # and no sandbox operation failed. Any other evaluation that did not complete has no verdict: its zero is not
        # a grade, so the sample is masked and the failure is reported as a verifier error.
        model_patch_unapplied = (
            bool(res.get("patch_apply_failed"))
            and mock_container._sandbox_error_type is None
            and not patch_extraction_failed
        )
        failure_reason = None
        if not evaluation_completed and not model_patch_unapplied:
            if patch_extraction_failed:
                failure_reason = "Failed to extract the model patch from the agent sandbox"
            elif mock_container._sandbox_error_type is not None:
                failure_reason = f"Verification sandbox error: {mock_container._sandbox_error_type}"
            else:
                failure_reason = res.get("error") or "SWE-bench evaluation did not complete"
            failure_reason = bounded_text(failure_reason, _MAX_FAILURE_REASON_CHARS)
            observe_failure(
                NgFailureEvent(
                    boundary=FailureBoundary.RESOURCES_SERVER,
                    failure_kind=VERIFIER_ERROR,
                    failure_reason=failure_reason,
                    operation="verify",
                    operation_id=current_verifier_invocation_id() or run_id,
                    stage="verification",
                )
            )

        log_dir = Path(__file__).parent / "logs/run_evaluation" / run_id

        test_output_fpaths = glob(str(log_dir / "**" / "test_output.txt"), recursive=True)
        test_output = ""
        if test_output_fpaths:
            test_output_fpath = Path(test_output_fpaths[0])
            test_output = test_output_fpath.read_text()

        if self.config.clear_swebench_debug_logs:
            rmtree(str(log_dir), ignore_errors=True)
            log_dir = ""

        return SWEBenchVerifyResponse(
            **body.model_dump(),
            # run_instance returns "completed"; the response field is "evaluation_completed".
            evaluation_completed=evaluation_completed,
            resolved=res["resolved"],
            reward=int(res["resolved"]),
            eval_sandbox_start_time_taken=eval_sandbox_start_time_taken,
            patch_verification_time_taken=patch_verification_time_taken,
            model_patch=model_patch or None,
            model_patch_sha256=None
            if patch_extraction_failed
            else hashlib.sha256(model_patch.encode("utf-8")).hexdigest(),
            mask_sample=failure_reason is not None,
            failure_kind=VERIFIER_ERROR if failure_reason is not None else None,
            failure_reason=failure_reason,
            test_output=test_output,
            log_dir=str(log_dir),
            verifier_sandbox_observation=verifier_sandbox_observation,
        )


def _lost_session_response(body: SWEBenchVerifyRequest, reason: str) -> SWEBenchVerifyResponse:
    """Return an unscored, masked result when the task state is unavailable."""
    return SWEBenchVerifyResponse.model_validate(
        body.model_dump()
        | lost_session_verification(reason)
        | {
            "evaluation_completed": False,
            "resolved": False,
            "eval_sandbox_start_time_taken": 0.0,
            "patch_verification_time_taken": 0.0,
            "instance_id": body.instance_id,
            "model_patch": None,
            "model_patch_sha256": None,
            "test_output": "",
            "log_dir": "",
        }
    )


if __name__ == "__main__":
    SwebenchResourcesServer.run_webserver()
