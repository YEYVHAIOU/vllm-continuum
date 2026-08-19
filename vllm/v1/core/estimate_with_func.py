from vllm.v1.request import Request
from vllm.v1.core.dynamic_ttl_estimator import (
    DynamicTTLEstimator,
    PiecewiseLinearPrefillReloadProfile,
    TTLEstimatorConfig,
)
from typing import Optional
import time
import math
import os
import re
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer

logger = init_logger(__name__)

FIXED_THRESHOLD_CONTINUUM = 2.0  # seconds

class Continuum_Recorder:
    def __init__(self):
        self.job_id_to_history = {}
        # Track scheduling operation timing
        self.scheduling_times = []  # List of {start_time, end_time, duration}

    def print_history(self):
        import os
        import json

        # Per-run output directory (set by launcher); fallback to default
        output_dir = os.environ.get("RUN_OUTPUT_DIR", "./continuum_exp")
        os.makedirs(output_dir, exist_ok=True)

        # Atomic write to avoid partial reads by other processes
        final_path = os.path.join(output_dir, "scheduler_timestamps")
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.job_id_to_history, f, indent=2)
        os.replace(tmp_path, final_path)

    def request_arrives(self, request: Request):
        if request.job_id not in self.job_id_to_history:
            self.job_id_to_history[request.job_id] = []
        self.job_id_to_history[request.job_id].append({"Request_arrival_time": time.time()})
    
    def request_finished(self, request: Request):
        self.job_id_to_history[request.job_id].append({"Request_departure_time": time.time()})

    def request_evicted_from_running_queue(self, request: Request):
        self.job_id_to_history[request.job_id].append({"Request_evicted_from_running_queue_time": time.time()})

    def request_pinned(self, request: Request):
        self.job_id_to_history[request.job_id].append({"pinned_time": time.time()})

    def request_unpinned(self, request: Request):
        self.job_id_to_history[request.job_id].append({"unpinned_time": time.time()})

    def request_waiting_to_running(self, request: Request, prompt_length: int, hit_length: int = 0):
        self.job_id_to_history[request.job_id].append({
            "waiting_to_running": time.time(),
            "prompt_length": prompt_length,
            "hit_length": hit_length
        })
    
    def request_evicted_to_running(self, request: Request, prompt_length: int, hit_length: int):
        self.job_id_to_history[request.job_id].append({
            "evicted_to_running": time.time(),
            "prompt_length": prompt_length,
            "hit_length": hit_length
        })

class ToolCallParser:
    """Parser for extracting function calls from LLM output.

    Uses the same parsing logic as mini-swe-agent to extract bash commands
    from markdown code blocks and identify the function call.

    This can be extended for other datasets with different parsing logic.
    """

    def parse(self, text: str) -> Optional[str]:
        """Parse LLM output and extract the function call name.

        Args:
            text: Output text from the LLM

        Returns:
            The function call name (e.g., "ls", "cd", "git"), or None if not found
        """
        # Same regex pattern as mini-swe-agent: r"```bash\s*\n(.*?)\n```"
        actions = re.findall(r"```bash\s*\n(.*?)\n```", text, re.DOTALL)

        if len(actions) == 1:
            bash_action = actions[0].strip()
            # Extract the first word (command) from the action
            words = bash_action.split()
            if words:
                return words[0]

        return None

class ToolCallEstimator:
    def __init__(
        self,
        tokenizer: Optional[AnyTokenizer] = None,
        model_name: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tokenizer_revision: Optional[str] = None,
        parser: Optional[ToolCallParser] = None,
    ):
        self.func_call_to_exec_time: dict[str, float] = {}
        self.record_func_call_to_exec_time: dict[str, list[float]] = {}

        self.job_to_history: dict[str, list[dict[str, float]]] = {}
        self.job_request_counts: dict[str, int] = {}
        # CONTINUUM_PREFILL_PROFILE_SCALE_V1
        base_prefill_profile_points = (
            (0, 0.000000000),
            (512, 0.016206744),
            (1024, 0.038612129),
            (2048, 0.095269512),
            (3072, 0.164592375),
            (4000, 0.234015222),
        )
        raw_profile_scale = os.environ.get(
            "CONTINUUM_PREFILL_PROFILE_SCALE", "1.0"
        )
        try:
            profile_scale = float(raw_profile_scale)
        except ValueError as exc:
            raise ValueError(
                "CONTINUUM_PREFILL_PROFILE_SCALE must be a positive number, "
                f"got {raw_profile_scale!r}"
            ) from exc
        if not math.isfinite(profile_scale) or profile_scale <= 0.0:
            raise ValueError(
                "CONTINUUM_PREFILL_PROFILE_SCALE must be finite and > 0, "
                f"got {profile_scale!r}"
            )
        # CONTINUUM_HISTORY_THRESHOLD_V1

        raw_history_threshold = os.environ.get(

            "CONTINUUM_HISTORY_THRESHOLD", "100"

        )

        try:

            history_threshold = int(raw_history_threshold)

        except ValueError as exc:

            raise ValueError(

                "CONTINUUM_HISTORY_THRESHOLD must be an integer >= 1, "

                f"got {raw_history_threshold!r}"

            ) from exc

        if history_threshold < 1:

            raise ValueError(

                "CONTINUUM_HISTORY_THRESHOLD must be >= 1, "

                f"got {history_threshold!r}"

            )

        logger.info(

            "Continuum history threshold=%d",

            history_threshold,

        )



        # CONTINUUM_DEFAULT_TTL_ENV_V1

        raw_default_ttl = os.environ.get(

            "CONTINUUM_DEFAULT_TTL_SECONDS", "2.0"

        )

        try:

            default_ttl_seconds = float(raw_default_ttl)

        except ValueError as exc:

            raise ValueError(

                "CONTINUUM_DEFAULT_TTL_SECONDS must be a nonnegative number, "

                f"got {raw_default_ttl!r}"

            ) from exc

        if (

            not math.isfinite(default_ttl_seconds)

            or default_ttl_seconds < 0.0

        ):

            raise ValueError(

                "CONTINUUM_DEFAULT_TTL_SECONDS must be finite and >= 0, "

                f"got {default_ttl_seconds!r}"

            )

        logger.info(

            "Continuum default TTL=%.6f",

            default_ttl_seconds,

        )



        scaled_prefill_profile_points = tuple(


            (tokens, seconds * profile_scale)
            for tokens, seconds in base_prefill_profile_points
        )
        # CONTINUUM_TTL_COUNTERFACTUAL_DIAGNOSTIC_V1
        self.prefill_profile_scale = profile_scale
        self.base_prefill_reload_profile = PiecewiseLinearPrefillReloadProfile(
            points=base_prefill_profile_points
        )
        raw_diagnostic_scales = os.environ.get(
            "CONTINUUM_TTL_DIAGNOSTIC_SCALES", ""
        ).strip()
        diagnostic_scales: list[float] = []
        if raw_diagnostic_scales:
            for raw_scale in raw_diagnostic_scales.split(","):
                raw_scale = raw_scale.strip()
                if not raw_scale:
                    continue
                try:
                    diagnostic_scale = float(raw_scale)
                except ValueError as exc:
                    raise ValueError(
                        "CONTINUUM_TTL_DIAGNOSTIC_SCALES must be a "
                        "comma-separated list of positive numbers, "
                        f"got {raw_diagnostic_scales!r}"
                    ) from exc
                if (
                    not math.isfinite(diagnostic_scale)
                    or diagnostic_scale <= 0.0
                ):
                    raise ValueError(
                        "CONTINUUM_TTL_DIAGNOSTIC_SCALES entries must "
                        f"be finite and > 0, got {diagnostic_scale!r}"
                    )
                diagnostic_scales.append(diagnostic_scale)
        self.ttl_diagnostic_scales = tuple(
            sorted(set(diagnostic_scales))
        )
        logger.info(
            "Continuum prefill profile scale=%.3f, points=%s",
            profile_scale,
            scaled_prefill_profile_points,
        )
        self.dynamic_ttl_estimator = DynamicTTLEstimator(

            config=TTLEstimatorConfig(

                history_threshold=history_threshold,


                default_ttl_seconds=default_ttl_seconds,
            ),

            prefill_reload_profile=PiecewiseLinearPrefillReloadProfile(
                points=scaled_prefill_profile_points
            )
        )
        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name is not None:
            try:
                self.tokenizer = get_tokenizer(
                    tokenizer_name=model_name,
                    tokenizer_mode=tokenizer_mode,
                    trust_remote_code=trust_remote_code,
                    revision=tokenizer_revision,
                )
                logger.info(f"Initialized tokenizer for model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize tokenizer for {model_name}: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None

        # Initialize parser (can be customized for different datasets)
        self.parser = parser if parser is not None else ToolCallParser()

    def get_func_call_exec_time(self, func: str) -> Optional[float]:
        if func not in self.func_call_to_exec_time:
            return None
        return self.func_call_to_exec_time[func]
    
    #TODO Hanchen This is currently just an average 
    def update_func_call_exec_time(self, job_id: str) -> None:
        # Called when the next request of the same job arrives.
        last_departure_time = self.job_to_history[job_id][-1][
            "departure_time"
        ]
        func = self.job_to_history[job_id][-1]["func_call"]
        if func is None:
            return

        exec_time = max(0.0, time.time() - last_departure_time)

        if func not in self.record_func_call_to_exec_time:
            self.record_func_call_to_exec_time[func] = [exec_time]
        else:
            self.record_func_call_to_exec_time[func].append(exec_time)

        records = self.record_func_call_to_exec_time[func]
        self.func_call_to_exec_time[func] = sum(records) / len(records)
        self.dynamic_ttl_estimator.record_tool_duration(func, exec_time)

    #Functions below will be called by outside functions
    def get_or_estimate_ttl(self, request: Request):
        """Return one stable TTL prediction for this request."""
        if request.this_func_call is None:
            return None

        cache = getattr(self, "_ttl_result_cache", None)
        if cache is None:
            cache = {}
            self._ttl_result_cache = cache

        result = cache.get(request.request_id)
        if result is None:
            result = self.dynamic_ttl_estimator.estimate_ttl(
                request.this_func_call,
                context_tokens=request.num_prompt_tokens,
            )
            cache[request.request_id] = result
            logger.info(
                "Continuum TTL precomputed request=%s job=%s tool=%s "
                "ttl=%.6f source=%s probability=%.6f score=%.6f "
                "prefill_reload=%.6f",
                request.request_id,
                request.job_id,
                request.this_func_call,
                result.ttl_seconds,
                result.history_source,
                result.finish_probability,
                result.expected_score,
                result.prefill_reload_cost,
            )
        return result

    def clear_cached_ttl_result(self, request_id: str) -> None:
        cache = getattr(self, "_ttl_result_cache", None)
        if cache is not None:
            cache.pop(request_id, None)

    def set_up_pin(self, request: Request) -> float:
        if request.this_func_call is None:
            return 0.0

        result = self.get_or_estimate_ttl(request)
        if result is None:
            return 0.0
        logger.info(
            "Continuum dynamic TTL request=%s job=%s tool=%s "
            "ttl=%.6f source=%s score=%.6f probability=%.6f "
            "queue_delay=%.6f memoryfulness=%.6f "
            "prefill_reload=%.6f selected_history=%d "
            "global_history=%d tool_history=%d candidates=%d",
            request.request_id,
            request.job_id,
            request.this_func_call,
            result.ttl_seconds,
            result.history_source,
            result.expected_score,
            result.finish_probability,
            result.average_queue_delay,
            result.memoryfulness,
            result.prefill_reload_cost,
            result.selected_history_size,
            result.global_history_size,
            result.tool_history_size,
            result.candidate_count,
        )
        if self.ttl_diagnostic_scales:
            base_prefill_reload = (
                self.base_prefill_reload_profile.estimate_seconds(
                    request.num_prompt_tokens
                )
            )
            for diagnostic_scale in self.ttl_diagnostic_scales:
                diagnostic_result = (
                    self.dynamic_ttl_estimator.estimate_ttl(
                        request.this_func_call,
                        context_tokens=request.num_prompt_tokens,
                        queue_delay_seconds=result.average_queue_delay,
                        memoryfulness=result.memoryfulness,
                        prefill_reload_cost_seconds=(
                            base_prefill_reload * diagnostic_scale
                        ),
                    )
                )
                logger.info(
                    "Continuum TTL counterfactual request=%s job=%s "
                    "tool=%s actual_scale=%.6f diagnostic_scale=%.6f "
                    "actual_ttl=%.6f cf_ttl=%.6f source=%s "
                    "score=%.6f probability=%.6f queue_delay=%.6f "
                    "memoryfulness=%.6f prefill_reload=%.6f "
                    "selected_history=%d global_history=%d "
                    "tool_history=%d candidates=%d",
                    request.request_id,
                    request.job_id,
                    request.this_func_call,
                    self.prefill_profile_scale,
                    diagnostic_scale,
                    result.ttl_seconds,
                    diagnostic_result.ttl_seconds,
                    diagnostic_result.history_source,
                    diagnostic_result.expected_score,
                    diagnostic_result.finish_probability,
                    diagnostic_result.average_queue_delay,
                    diagnostic_result.memoryfulness,
                    diagnostic_result.prefill_reload_cost,
                    diagnostic_result.selected_history_size,
                    diagnostic_result.global_history_size,
                    diagnostic_result.tool_history_size,
                    diagnostic_result.candidate_count,
                )

        ttl_seconds = result.ttl_seconds
        self.clear_cached_ttl_result(request.request_id)
        return ttl_seconds
    def record_queue_transition(
        self,
        request: Request,
        *,
        prompt_length: int,
        hit_length: int,
        resumed_from_preemption: bool = False,
    ) -> None:
        if resumed_from_preemption:
            return
        if request.last_func_call is None:
            return
        if hit_length >= prompt_length:
            return

        queue_delay = max(0.0, time.time() - request.arrival_time)
        self.dynamic_ttl_estimator.record_queue_delay(queue_delay)

    def request_arrives(self, request: Request) -> None:
        logger.info(f"Request job id arriving: {request.job_id}, time is {time.time()}")
        self.job_request_counts[request.job_id] = (
            self.job_request_counts.get(request.job_id, 0) + 1
        )
        # this is called when a job arrives in scheduler.py, if job is new, create an entry,
        if request.job_id not in self.job_to_history:
            self.job_to_history[request.job_id] = []
            if request.last_func_call is not None:
                logger.warning(
                    "Continuum recovered unseen follow-up job=%s "
                    "last_func_call=%s; prior tool duration is unavailable "
                    "and will be skipped",
                    request.job_id,
                    request.last_func_call,
                )
            self.job_to_history[request.job_id].append(
                {"arrival_time": request.arrival_time}
            )
            return
        request.last_func_call = self.job_to_history[request.job_id][-1]["func_call"]
        logger.info(f"Request job id: {request.job_id}, last func call: {request.last_func_call}")

        self.update_func_call_exec_time(request.job_id)

        self.job_to_history[request.job_id].append({"arrival_time": request.arrival_time})
        return
    
    def request_finished(self, request: Request) -> None:
        logger.info(f"Request job id finishing: {request.job_id}, time is {time.time()}")

        # Prefer API metadata; retain generated-output parsing as fallback.
        this_func_call = request.this_func_call
        if (
            this_func_call is None
            and self.tokenizer is not None
            and len(request.output_token_ids) > 0
        ):
            try:
                # Detokenize the output tokens
                output_text = self.tokenizer.decode(
                    request.output_token_ids,
                    skip_special_tokens=True
                )

                # Parse function call using the parser
                this_func_call = self.parser.parse(output_text)

                if this_func_call:
                    logger.info(f"Extracted func_call: {this_func_call} from output")
                else:
                    logger.debug(f"No function call found in output: {output_text[:200]}")
            except Exception as e:
                logger.warning(f"Error detokenizing/parsing output for request {request.request_id}: {e}")

        request.this_func_call = this_func_call
        self.job_to_history[request.job_id].append({
            "departure_time": time.time(),
            "func_call": request.this_func_call
        })
        if bool(request.is_last_step):
            self.dynamic_ttl_estimator.record_completed_program(
                self.job_request_counts.get(request.job_id, 1)
            )
            self.clear_cached_ttl_result(request.request_id)
        return

