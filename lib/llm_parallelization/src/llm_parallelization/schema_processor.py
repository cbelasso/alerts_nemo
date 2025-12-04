"""
Standalone FlexibleSchemaProcessor with Enhanced Mistral Support

This module provides a fully decoupled, standalone processor for schema-based
LLM inference with special support for Mistral models and other models that
require custom tokenizer configurations.
"""

from collections import defaultdict
from datetime import datetime
import json
import multiprocessing
from multiprocessing import Event, Queue, get_context
import os
from queue import Empty
import time
from typing import Dict, Generator, List, Optional, Tuple, Type, Union
import uuid

from pydantic import BaseModel
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

multiprocessing.set_start_method("spawn", force=True)


class FlexibleGuidedModel:
    """
    Enhanced Model class with runtime schema switching for guided decoding.

    Supports special model configurations including Mistral models with custom
    tokenizer modes and loading formats.
    """

    def __init__(
        self,
        default_guided_config: Optional[Dict] = None,
        **llm_kwargs,
    ):
        self.default_guided_config = default_guided_config or {}
        self.llm_kwargs = llm_kwargs
        self.model = self.load_llm(**self.llm_kwargs)

        # Store default sampling params for non-guided generation
        self.base_sampling_params = self.get_sampling_params()

    def load_llm(self, **llm_kwargs) -> LLM:
        """Load the LLM model with support for special configurations."""
        init_kwargs = {
            "model": llm_kwargs["model_path"],
            "trust_remote_code": True,
            "enforce_eager": True,
            "gpu_memory_utilization": llm_kwargs.get("gpu_memory_utilization", 0.9),
            "max_model_len": llm_kwargs.get("max_model_len", None),
            "dtype": llm_kwargs.get("dtype", "auto"),
        }

        # Add optional parameters if present
        if "tokenizer" in llm_kwargs:
            init_kwargs["tokenizer"] = llm_kwargs["tokenizer"]
        if "tokenizer_mode" in llm_kwargs:
            init_kwargs["tokenizer_mode"] = llm_kwargs["tokenizer_mode"]
        if "config_format" in llm_kwargs:
            init_kwargs["config_format"] = llm_kwargs["config_format"]
        if "load_format" in llm_kwargs:
            init_kwargs["load_format"] = llm_kwargs["load_format"]
        if "tensor_parallel_size" in llm_kwargs:
            init_kwargs["tensor_parallel_size"] = llm_kwargs["tensor_parallel_size"]

        return LLM(**init_kwargs)

    def get_sampling_params(self):
        """Get default sampling parameters."""
        return SamplingParams(
            temperature=0.95,
            top_k=50,
            top_p=0.95,
            max_tokens=4098,
            frequency_penalty=2,
            repetition_penalty=1.1,
        )

    def create_guided_sampling_params(
        self, json_schema: Optional[Dict] = None, guided_config: Optional[Dict] = None
    ) -> SamplingParams:
        """
        Create sampling parameters with optional JSON schema-based guided decoding.
        """
        config = {**self.default_guided_config, **(guided_config or {})}

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=config.get("temperature", 0.1),
            top_k=config.get("top_k", 50),
            top_p=config.get("top_p", 0.95),
            max_tokens=config.get("max_tokens", 1000),
            frequency_penalty=config.get("frequency_penalty", 0.0),
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        # Add guided decoding if schema is provided
        if json_schema:
            guided_decoding_params = GuidedDecodingParams(json=json_schema)
            sampling_params.guided_decoding = guided_decoding_params

        return sampling_params

    def generate_with_json_schema(
        self,
        prompts: Union[str, List[str]],
        json_schema: Optional[Dict] = None,
        guided_config: Optional[Dict] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generate outputs with optional JSON schema constraints."""
        prompt_list = [prompts] if isinstance(prompts, str) else prompts

        # Create sampling params for this specific generation
        sampling_params = self.create_guided_sampling_params(json_schema, guided_config)

        return self.model.generate(
            prompts=prompt_list,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
        )

    def generate(
        self, prompts: Union[str, List[str]], use_tqdm: bool = True
    ) -> List[RequestOutput]:
        """Generate without schema constraints (backward compatibility)."""
        prompt_list = [prompts] if isinstance(prompts, str) else prompts
        return self.model.generate(
            prompts=prompt_list,
            sampling_params=self.base_sampling_params,
            use_tqdm=use_tqdm,
        )


class FlexibleSchemaProcessor:
    """
    Standalone processor with runtime schema switching for structured output generation.

    A fully independent processor that can work with different schemas for each request batch,
    making it highly reusable across different tasks and output formats. Supports special
    model configurations like Mistral models that require custom tokenizer modes.

    Args:
        gpu_list: List of GPU indices to use for distributed inference
        llm: Model identifier string
        multiplicity: Number of model instances per GPU
        use_tqdm: Whether to display progress bars
        gpu_memory_utilization: GPU memory utilization fraction
        max_model_len: Maximum sequence length
        default_guided_config: Default guided decoding configuration
        tokenizer: Optional separate tokenizer path (if different from model)
        tokenizer_mode: Tokenizer mode for vLLM (e.g., "auto", "mistral")
        config_format: Config format for special models (e.g., "mistral")
        load_format: Load format for special models (e.g., "mistral")
        tensor_parallel_size: Number of GPUs for tensor parallelism
        skip_tokenizer_init: If True, skip external tokenizer loading (useful for Mistral)
        **extra_llm_args: Additional LLM initialization arguments

    Example:
        >>> # Standard usage
        >>> processor = FlexibleSchemaProcessor(gpu_list=[0, 1])
        >>>
        >>> # Mistral model usage
        >>> processor = FlexibleSchemaProcessor(
        ...     gpu_list=[0, 1],
        ...     llm="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        ...     tokenizer_mode="mistral",
        ...     config_format="mistral",
        ...     load_format="mistral",
        ...     skip_tokenizer_init=True,
        ...     tensor_parallel_size=2
        ... )
    """

    def __init__(
        self,
        gpu_list: list[int],
        llm: str = "meta-llama/Llama-3.2-3B-Instruct",
        multiplicity: int = 1,
        use_tqdm: bool = False,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 2048,
        default_guided_config: Optional[Dict] = None,
        tokenizer: str | None = None,
        tokenizer_mode: str = "auto",
        config_format: str | None = None,
        load_format: str | None = None,
        tensor_parallel_size: int = 1,
        skip_tokenizer_init: bool = False,
        **extra_llm_args,
    ):
        self.gpu_list = gpu_list
        self.llm = llm
        self.multiplicity = multiplicity
        self.use_tqdm = use_tqdm
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.default_guided_config = default_guided_config or {}
        self.skip_tokenizer_init = skip_tokenizer_init

        # Try to load tokenizer if not skipped
        self.tokenizer = None
        if not skip_tokenizer_init:
            try:
                tokenizer_path = tokenizer or self.llm
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                print(f"✓ External tokenizer loaded: {tokenizer_path}")
            except Exception as e:
                print(f"⚠ Could not load external tokenizer: {e}")
                print("  Will rely on vLLM's internal tokenizer for chat formatting")

        self.task_queue: Queue = Queue()
        self.response_queue: Queue = Queue()
        self.load_signal_queue: Queue = Queue()
        self.processes = []
        self.stop_event: Event = Event()
        self.responses = []
        self.num_gpus = len(gpu_list)

        # Assemble LLM initialization arguments
        self.llm_kwargs = {
            "model_path": self.llm,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tokenizer_mode": tokenizer_mode,
            "tensor_parallel_size": tensor_parallel_size,
            **extra_llm_args,
        }

        # Add optional parameters if specified
        if tokenizer is not None:
            self.llm_kwargs["tokenizer"] = tokenizer
        if config_format is not None:
            self.llm_kwargs["config_format"] = config_format
        if load_format is not None:
            self.llm_kwargs["load_format"] = load_format

        self.prepare_processes()

        print("🔄 FlexibleSchemaProcessor initialized - ready for runtime schema switching")

    def format_prompt(self, prompt: str) -> str:
        """
        Format prompt using tokenizer chat template.

        If external tokenizer is not available, returns the prompt as-is.
        For models like Mistral, you should pre-format prompts before passing them in.
        """
        if self.tokenizer is None:
            # No external tokenizer - return prompt as-is
            # User should pre-format or rely on vLLM's internal formatting
            return prompt

        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as e:
            print(f"⚠ Chat template formatting failed: {e}")
            print("  Returning prompt as-is")
            return prompt

    def create_batches(self, prompts: list, batch_size: int) -> list:
        """Create batches from prompt list."""
        return [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

    @staticmethod
    def worker(
        llm_kwargs,
        gpu_num,
        task_queue,
        response_queue,
        stop_event,
        load_signal_queue,
        multiplicity_index,
        parser,
        default_guided_config=None,
    ):
        """
        Worker process with flexible schema support.

        Creates a FlexibleGuidedModel that can handle schema switching at runtime.
        Each task in the queue can specify its own JSON schema and guided config.
        """
        try:
            llm_kwargs["gpu_num"] = gpu_num
            model = FlexibleGuidedModel(
                default_guided_config=default_guided_config, **llm_kwargs
            )
            load_signal_queue.put((gpu_num, multiplicity_index))
        except Exception as e:
            load_signal_queue.put(
                f"Error loading model on GPU {gpu_num} (multiplicity {multiplicity_index}): {str(e)}"
            )
            return

        while not stop_event.is_set():
            request_id = None
            corr_id = None
            request = None

            try:
                task_data = task_queue.get(timeout=1)

                # Handle termination signal
                if len(task_data) == 3:
                    request_id, request, corr_id = task_data
                    if request_id == -1:
                        break
                    # Legacy format without schema
                    response = model.generate(prompts=request, use_tqdm=False)
                else:
                    # New format with JSON schema and config
                    request_id, request, corr_id, json_schema, guided_config = task_data

                    response = model.generate_with_json_schema(
                        prompts=request,
                        json_schema=json_schema,
                        guided_config=guided_config,
                        use_tqdm=False,
                    )

                response_queue.put((request_id, response, corr_id, request))
            except Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                if request_id is not None and corr_id is not None and request is not None:
                    response_queue.put((request_id, None, corr_id, request))

    def prepare_processes(self) -> None:
        """Initialize worker processes with flexible schema support."""
        ctx = get_context("spawn")

        tensor_parallel_size = self.llm_kwargs.get("tensor_parallel_size", 1)

        if tensor_parallel_size > 1:
            # Split gpu_list into chunks for tensor parallelism
            # e.g., [1,2,3,4] with tp_size=2 → [[1,2], [3,4]]
            gpu_chunks = [
                self.gpu_list[i : i + tensor_parallel_size]
                for i in range(0, len(self.gpu_list), tensor_parallel_size)
            ]

            print(
                f"Tensor parallelism: Creating {len(gpu_chunks)} model(s) with {tensor_parallel_size} GPUs each"
            )

            for i in range(self.multiplicity):
                for chunk_idx, gpu_chunk in enumerate(gpu_chunks):
                    # Set ONLY this chunk visible
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_chunk))

                    process = ctx.Process(
                        target=FlexibleSchemaProcessor.worker,
                        args=(
                            self.llm_kwargs,
                            gpu_chunk[0],  # Use first GPU as identifier
                            self.task_queue,
                            self.response_queue,
                            self.stop_event,
                            self.load_signal_queue,
                            i,
                            None,
                            self.default_guided_config,
                        ),
                    )
                    process.start()
                    self.processes.append(process)

                self.wait_for_models_to_load(expected_count=len(gpu_chunks))
        else:
            # Original behavior: one model per GPU
            for i in range(self.multiplicity):
                gpu_memory_utilization = min(self.gpu_memory_utilization + i * 0.2, 1.0)
                print(
                    f"Starting multiplicity round {i + 1} with GPU memory utilization: {gpu_memory_utilization}"
                )

                for gpu_num in self.gpu_list:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
                    process = ctx.Process(
                        target=FlexibleSchemaProcessor.worker,  # target= is required!
                        args=(  # args= is required!
                            self.llm_kwargs,
                            gpu_num,
                            self.task_queue,
                            self.response_queue,
                            self.stop_event,
                            self.load_signal_queue,
                            i,
                            None,  # parser
                            self.default_guided_config,
                        ),
                    )
                    process.start()
                    self.processes.append(process)

                self.wait_for_models_to_load(expected_count=self.num_gpus)

    def wait_for_models_to_load(self, expected_count, timeout=None):
        """Wait for a specific number of models to be loaded."""
        start_time = time.time()
        loaded_models = set()
        errors = []

        while len(loaded_models) < expected_count:
            try:
                result = self.load_signal_queue.get(timeout=1)
                if isinstance(result, tuple):
                    loaded_models.add(result)
                    print(f"Model loaded on GPU {result[0]} (multiplicity {result[1]})")
                else:
                    # This is an error message
                    errors.append(result)
                    print(result)
            except Empty:
                pass

            if timeout is not None and time.time() - start_time > timeout:
                print("Timeout waiting for models to load")
                return False

            if len(errors) + len(loaded_models) == expected_count:
                break

        if errors:
            print("Some models failed to load")
            return False

        print(f"All {expected_count} models in this round loaded successfully")
        return True

    def process_with_schema(
        self,
        prompts: Union[str, List[str]],
        schema: Optional[Type[BaseModel]] = None,
        batch_size: int = 25,
        formatted: bool = False,
        guided_config: Optional[Dict] = None,
        on_batch_end=None,
        timeout=10,
    ) -> List[RequestOutput]:
        """
        Process requests with a specific schema for guided decoding.

        Args:
            prompts: Input prompts to process
            schema: Optional Pydantic schema for structured output
            batch_size: Number of prompts per batch
            formatted: Whether prompts are already formatted (recommended for Mistral)
            guided_config: Optional guided decoding configuration
            on_batch_end: Optional callback function for batch completion
            timeout: Request timeout in seconds

        Returns:
            List[RequestOutput]: Generated responses
        """
        prompt_list = [prompts] if isinstance(prompts, str) else prompts

        if formatted:
            formatted_prompt_list = prompt_list
        else:
            formatted_prompt_list = [
                self.format_prompt(prompt=prompt) for prompt in prompt_list
            ]

        batch_prompts = {
            request_id: batch
            for request_id, batch in enumerate(
                self.create_batches(prompts=formatted_prompt_list, batch_size=batch_size)
            )
        }
        book_keeping_indexes = {
            request_id: batch
            for request_id, batch in enumerate(
                self.create_batches(
                    prompts=list(range(0, len(prompt_list))), batch_size=batch_size
                )
            )
        }

        total_requests = len(batch_prompts)
        response_counter = 0
        current_corr_id = uuid.uuid4()

        # Convert schema to JSON schema if provided
        json_schema = None
        if schema:
            json_schema = schema.model_json_schema()

        # Submit tasks with JSON schema information
        for request_id, prompts_ in batch_prompts.items():
            self.task_queue.put(
                (request_id, prompts_, current_corr_id, json_schema, guided_config)
            )

        processed_responses = {}

        with tqdm(
            total=total_requests,
            colour="#B48EAD",
            leave=False,
            desc=f"Process requests with schema {schema.__name__ if schema else 'None'} {current_corr_id}",
        ) as pbar:
            while response_counter < total_requests and not self.stop_event.is_set():
                try:
                    request_id, response, corr_id, prompts_ = self.response_queue.get(timeout=1)

                    if response is None:
                        print(f"Failed on request_id {request_id}")
                        self.task_queue.put(
                            (request_id, prompts_, current_corr_id, json_schema, guided_config)
                        )
                        continue

                    if current_corr_id != corr_id:
                        raise RuntimeError(
                            f"Current correlation id {current_corr_id} does not match result queue correlation id {corr_id}"
                        )

                    response_counter += 1

                    if on_batch_end:
                        on_batch_end(
                            batch_prompts[request_id],
                            book_keeping_indexes[request_id],
                            response,
                        )

                    processed_responses[request_id] = response
                    pbar.update(1)

                except Empty:
                    continue
                except Exception as e:
                    print(f"Processing error: {e}")

        self.responses = [
            processed_responses[request_id] for request_id in sorted(processed_responses.keys())
        ]

        return self.responses

    def parse_results_with_schema(
        self,
        schema: Type[BaseModel],
        responses: Optional[List[RequestOutput]] = None,
        validate: bool = True,
    ) -> List[Union[BaseModel, Dict, str, None]]:
        """
        Parse results with a specific schema.

        Args:
            schema: Pydantic schema to use for parsing
            responses: Optional specific responses to parse (defaults to last processed)
            validate: Whether to validate against the schema

        Returns:
            List of parsed results
        """
        responses_to_parse = responses or self.responses
        parsed_results = []

        for response in tqdm(
            responses_to_parse, desc=f"Parsing with {schema.__name__ if schema else 'None'}"
        ):
            all_texts = self.extract_all_batch_outputs(response)

            for text_output in all_texts:
                try:
                    # Clean up text
                    text_output = text_output.strip()
                    if text_output.startswith("```json"):
                        text_output = (
                            text_output.replace("```json", "").replace("```", "").strip()
                        )

                    # Parse JSON
                    json_data = json.loads(text_output)

                    if validate:
                        validated_obj = schema(**json_data)
                        parsed_results.append(validated_obj)
                    else:
                        parsed_results.append(json_data)

                except Exception as e:
                    print(f"Failed to parse output: {text_output[:100]}...")
                    print(f"Error: {e}")
                    parsed_results.append(None)

        return parsed_results

    def extract_all_batch_outputs(self, response):
        """Extract all text outputs from a single or batched inference response."""
        all_texts = []

        if isinstance(response, list):
            for resp in response:
                if hasattr(resp, "outputs"):
                    for output in resp.outputs:
                        all_texts.append(output.text)
        else:
            if hasattr(response, "outputs"):
                for output in response.outputs:
                    all_texts.append(output.text)

        return all_texts

    def terminate(self):
        """Terminate all worker processes and clean up resources."""
        if hasattr(self, "stop_event") and self.stop_event is not None:
            self.stop_event.set()
            for _ in range(len(self.processes)):
                self.task_queue.put((-1, "TERMINATE", None))
            for process in self.processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
            self.processes.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear references to avoid potential circular references
        self.stop_event = None
        self.task_queue = None
        self.response_queue = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    def __del__(self):
        try:
            self.terminate()
        except:
            # If an exception occurs during cleanup in __del__,
            # it's generally better to ignore it than to crash the program
            pass
