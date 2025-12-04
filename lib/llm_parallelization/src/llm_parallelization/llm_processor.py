"""
Refactored LLMProcessor - A unified, self-contained processor for schema-based LLM inference

This module provides a single, comprehensive processor class that handles all aspects of
distributed LLM inference with optional schema-based guided decoding. All functionality
is consolidated into one class for simplicity and ease of extension.
"""

from collections import defaultdict
from datetime import datetime
import json
import multiprocessing
from multiprocessing import Event, Queue, get_context
import os
from queue import Empty
import time
from typing import Dict, List, Optional, Type, Union
import uuid

from pydantic import BaseModel
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

multiprocessing.set_start_method("spawn", force=True)


class LLMProcessor:
    """
    Unified processor for distributed LLM inference with optional schema-based guided decoding.

    A self-contained class that handles all aspects of multi-GPU inference including:
    - Multi-process distribution across GPUs
    - Runtime schema switching for structured outputs
    - Support for special model configurations (Mistral, etc.)
    - Tensor parallelism for large models
    - Batch processing with progress tracking

    All functionality is consolidated into a single class for simplicity and maintainability.

    Args:
        gpu_list: List of GPU indices to use for distributed inference
        llm: Model identifier string (e.g., "meta-llama/Llama-3.2-3B-Instruct")
        multiplicity: Number of model instances per GPU for increased throughput
        use_tqdm: Whether to display progress bars during processing
        gpu_memory_utilization: GPU memory utilization fraction (0.0-1.0)
        max_model_len: Maximum sequence length for the model
        default_guided_config: Default configuration for guided decoding (temperature, etc.)
        tokenizer: Optional separate tokenizer path (if different from model)
        tokenizer_mode: Tokenizer mode for vLLM (e.g., "auto", "mistral")
        config_format: Config format for special models (e.g., "mistral")
        load_format: Load format for special models (e.g., "mistral")
        tensor_parallel_size: Number of GPUs for tensor parallelism
        skip_tokenizer_init: If True, skip external tokenizer loading (useful for Mistral)
        **extra_llm_args: Additional LLM initialization arguments

    Example:
        >>> # Standard usage
        >>> processor = LLMProcessor(gpu_list=[0, 1])
        >>> results = processor.process_with_schema(prompts, MySchema)
        >>> parsed = processor.parse_results_with_schema(MySchema)
        >>>
        >>> # Mistral model usage
        >>> processor = LLMProcessor(
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

        # Try to load external tokenizer if not skipped
        self.tokenizer = None
        if not skip_tokenizer_init:
            try:
                tokenizer_path = tokenizer or self.llm
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                print(f"✓ External tokenizer loaded: {tokenizer_path}")
            except Exception as e:
                print(f"⚠ Could not load external tokenizer: {e}")
                print("  Will rely on vLLM's internal tokenizer for chat formatting")

        # Initialize multiprocessing components
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

        # Start worker processes
        self.prepare_processes()

        print("✅ LLMProcessor initialized - ready for inference")

    # ============================================================================
    # Model Loading and Sampling Configuration (formerly in FlexibleGuidedModel)
    # ============================================================================

    @staticmethod
    def load_llm(**llm_kwargs) -> LLM:
        """
        Load the LLM model with support for special configurations.

        This method handles loading vLLM models with various configurations
        including special tokenizer modes and formats for models like Mistral.

        All parameters in llm_kwargs (except 'model_path' and 'gpu_num') are
        passed through to vLLM's LLM constructor, allowing full customization.
        """
        # Base configuration with defaults
        init_kwargs = {
            "model": llm_kwargs["model_path"],
            "trust_remote_code": True,
            "enforce_eager": llm_kwargs.get("enforce_eager", True),  # Allow override
            "gpu_memory_utilization": llm_kwargs.get("gpu_memory_utilization", 0.9),
            "max_model_len": llm_kwargs.get("max_model_len", None),
            "dtype": llm_kwargs.get("dtype", "auto"),
        }

        # List of parameters we've already handled or should skip
        handled_params = {
            "model_path",  # Mapped to "model"
            "gpu_num",  # Internal tracking, not for vLLM
            "enforce_eager",  # Already handled above
            "gpu_memory_utilization",  # Already handled
            "max_model_len",  # Already handled
            "dtype",  # Already handled
        }

        # Pass through all other parameters to vLLM
        # This allows users to specify ANY vLLM parameter
        for key, value in llm_kwargs.items():
            if key not in handled_params and value is not None:
                init_kwargs[key] = value

        return LLM(**init_kwargs)

    @staticmethod
    def get_default_sampling_params() -> SamplingParams:
        """Get default sampling parameters for non-guided generation."""
        return SamplingParams(
            temperature=0.95,
            top_k=50,
            top_p=0.95,
            max_tokens=4098,
            frequency_penalty=2,
            repetition_penalty=1.1,
        )

    @staticmethod
    def create_guided_sampling_params(
        json_schema: Optional[Dict] = None,
        guided_config: Optional[Dict] = None,
        default_config: Optional[Dict] = None,
    ) -> SamplingParams:
        """
        Create sampling parameters with optional JSON schema-based guided decoding.

        Args:
            json_schema: Optional JSON schema dictionary for output structure
            guided_config: Optional configuration overrides for this generation
            default_config: Default configuration to merge with

        Returns:
            SamplingParams: Configured sampling parameters with or without guided decoding
        """
        default_config = default_config or {}
        config = {**default_config, **(guided_config or {})}

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

    # ============================================================================
    # Worker Process Logic
    # ============================================================================

    @staticmethod
    def worker(
        llm_kwargs: Dict,
        gpu_num: int,
        task_queue: Queue,
        response_queue: Queue,
        stop_event: Event,
        load_signal_queue: Queue,
        multiplicity_index: int,
        default_guided_config: Optional[Dict] = None,
    ):
        """
        Worker process with flexible schema support.

        This worker loads a vLLM model and processes inference requests from the queue.
        Each task can specify its own JSON schema and guided config for structured output.

        The worker handles both guided (with schema) and non-guided generation.
        """
        try:
            # Load the model
            llm_kwargs["gpu_num"] = gpu_num
            model = LLMProcessor.load_llm(**llm_kwargs)

            # Get default sampling params
            base_sampling_params = LLMProcessor.get_default_sampling_params()

            # Signal successful loading
            load_signal_queue.put((gpu_num, multiplicity_index))

        except Exception as e:
            load_signal_queue.put(
                f"Error loading model on GPU {gpu_num} (multiplicity {multiplicity_index}): {str(e)}"
            )
            return

        # Process requests until stop signal
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

                    # Legacy format without schema - use base sampling params
                    response = model.generate(
                        prompts=request,
                        sampling_params=base_sampling_params,
                        use_tqdm=False,
                    )
                else:
                    # New format with JSON schema and config
                    request_id, request, corr_id, json_schema, guided_config = task_data

                    # Create sampling params for this request
                    sampling_params = LLMProcessor.create_guided_sampling_params(
                        json_schema=json_schema,
                        guided_config=guided_config,
                        default_config=default_guided_config,
                    )

                    # Generate with schema constraints
                    response = model.generate(
                        prompts=request,
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )

                # Send response back
                response_queue.put((request_id, response, corr_id, request))

            except Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                if request_id is not None and corr_id is not None and request is not None:
                    response_queue.put((request_id, None, corr_id, request))

    # ============================================================================
    # Process Management
    # ============================================================================

    def prepare_processes(self) -> None:
        """
        Initialize worker processes for distributed inference.

        Creates and starts worker processes across specified GPUs. Supports both
        standard multi-GPU setup and tensor parallelism for large models.
        """
        ctx = get_context("spawn")
        tensor_parallel_size = self.llm_kwargs.get("tensor_parallel_size", 1)

        if tensor_parallel_size > 1:
            # Tensor parallelism: split GPUs into chunks
            # e.g., [0,1,2,3] with tp_size=2 → [[0,1], [2,3]]
            gpu_chunks = [
                self.gpu_list[i : i + tensor_parallel_size]
                for i in range(0, len(self.gpu_list), tensor_parallel_size)
            ]

            print(
                f"🔗 Tensor parallelism: Creating {len(gpu_chunks)} model(s) "
                f"with {tensor_parallel_size} GPUs each"
            )

            for i in range(self.multiplicity):
                for chunk_idx, gpu_chunk in enumerate(gpu_chunks):
                    # Set only this chunk visible to the worker
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_chunk))

                    process = ctx.Process(
                        target=LLMProcessor.worker,
                        args=(
                            self.llm_kwargs,
                            gpu_chunk[0],  # Use first GPU as identifier
                            self.task_queue,
                            self.response_queue,
                            self.stop_event,
                            self.load_signal_queue,
                            i,
                            self.default_guided_config,
                        ),
                    )
                    process.start()
                    self.processes.append(process)

                self.wait_for_models_to_load(expected_count=len(gpu_chunks))
        else:
            # Standard mode: one model per GPU
            for i in range(self.multiplicity):
                gpu_memory_utilization = min(self.gpu_memory_utilization + i * 0.2, 1.0)
                print(
                    f"Starting multiplicity round {i + 1} "
                    f"with GPU memory utilization: {gpu_memory_utilization}"
                )

                for gpu_num in self.gpu_list:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

                    process = ctx.Process(
                        target=LLMProcessor.worker,
                        args=(
                            self.llm_kwargs,
                            gpu_num,
                            self.task_queue,
                            self.response_queue,
                            self.stop_event,
                            self.load_signal_queue,
                            i,
                            self.default_guided_config,
                        ),
                    )
                    process.start()
                    self.processes.append(process)

                self.wait_for_models_to_load(expected_count=self.num_gpus)

    def wait_for_models_to_load(
        self, expected_count: int, timeout: Optional[float] = None
    ) -> bool:
        """
        Wait for a specific number of models to be loaded.

        Args:
            expected_count: Number of models to wait for
            timeout: Maximum time to wait in seconds (None = wait indefinitely)

        Returns:
            bool: True if all expected models loaded successfully
        """
        start_time = time.time()
        loaded_models = set()
        errors = []

        while len(loaded_models) < expected_count:
            try:
                result = self.load_signal_queue.get(timeout=1)
                if isinstance(result, tuple):
                    loaded_models.add(result)
                    print(f"✓ Model loaded on GPU {result[0]} (multiplicity {result[1]})")
                else:
                    # This is an error message
                    errors.append(result)
                    print(result)
            except Empty:
                pass

            if timeout is not None and time.time() - start_time > timeout:
                print("⏱️  Timeout waiting for models to load")
                return False

            if len(errors) + len(loaded_models) == expected_count:
                break

        if errors:
            print("❌ Some models failed to load")
            return False

        print(f"✅ All {expected_count} model(s) loaded successfully")
        return True

    # ============================================================================
    # Prompt Formatting and Batching
    # ============================================================================

    def format_prompt(self, prompt: str) -> str:
        """
        Format prompt using tokenizer chat template.

        If external tokenizer is not available, returns the prompt as-is.
        For models like Mistral, you should pre-format prompts before passing them in.
        """
        if self.tokenizer is None:
            # No external tokenizer - return prompt as-is
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

    # ============================================================================
    # Main Processing Methods
    # ============================================================================

    def process_with_schema(
        self,
        prompts: Union[str, List[str]],
        schema: Optional[Type[BaseModel]] = None,
        batch_size: int = 25,
        formatted: bool = False,
        guided_config: Optional[Dict] = None,
        on_batch_end=None,
        timeout: int = 10,
    ) -> List[RequestOutput]:
        """
        Process requests with optional schema for guided decoding.

        Args:
            prompts: Input prompts to process
            schema: Optional Pydantic schema for structured output
            batch_size: Number of prompts per batch
            formatted: Whether prompts are already formatted (recommended for Mistral)
            guided_config: Optional guided decoding configuration
            on_batch_end: Optional callback function(batch_prompts, indexes, response)
            timeout: Request timeout in seconds

        Returns:
            List[RequestOutput]: Generated responses

        Example:
            >>> results = processor.process_with_schema(
            ...     prompts=["What is AI?", "Define ML"],
            ...     schema=DefinitionSchema,
            ...     batch_size=2
            ... )
        """
        prompt_list = [prompts] if isinstance(prompts, str) else prompts

        # Format prompts if needed
        if formatted:
            formatted_prompt_list = prompt_list
        else:
            formatted_prompt_list = [
                self.format_prompt(prompt=prompt) for prompt in prompt_list
            ]

        # Create batches
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
            print(f"📋 Using schema: {schema.__name__}")

        # Submit tasks with JSON schema information
        for request_id, prompts_ in batch_prompts.items():
            self.task_queue.put(
                (request_id, prompts_, current_corr_id, json_schema, guided_config)
            )

        processed_responses = {}

        # Process responses
        schema_desc = schema.__name__ if schema else "No Schema"
        with tqdm(
            total=total_requests,
            colour="#B48EAD",
            leave=False,
            desc=f"Processing with {schema_desc}",
        ) as pbar:
            while response_counter < total_requests and not self.stop_event.is_set():
                try:
                    request_id, response, corr_id, prompts_ = self.response_queue.get(timeout=1)

                    if response is None:
                        print(f"⚠️  Failed on request_id {request_id}, retrying...")
                        self.task_queue.put(
                            (request_id, prompts_, current_corr_id, json_schema, guided_config)
                        )
                        continue

                    if current_corr_id != corr_id:
                        raise RuntimeError(
                            f"Correlation ID mismatch: {current_corr_id} != {corr_id}"
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
                    print(f"❌ Processing error: {e}")

        # Store responses in order
        self.responses = [
            processed_responses[request_id] for request_id in sorted(processed_responses.keys())
        ]

        return self.responses

    # ============================================================================
    # Result Extraction and Parsing
    # ============================================================================

    def extract_all_batch_outputs(self, response) -> List[str]:
        """
        Extract all text outputs from a single or batched inference response.

        Args:
            response: vLLM RequestOutput object or list of RequestOutput objects

        Returns:
            List[str]: All text outputs from the response(s)
        """
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

    def parse_results_with_schema(
        self,
        schema: Type[BaseModel],
        responses: Optional[List[RequestOutput]] = None,
        validate: bool = True,
    ) -> List[Union[BaseModel, Dict, str, None]]:
        """
        Parse results with a specific schema.

        Args:
            schema: Pydantic schema to use for parsing and validation
            responses: Optional specific responses to parse (defaults to last processed)
            validate: Whether to validate against the schema

        Returns:
            List of parsed results (BaseModel objects if validate=True, else dicts)

        Example:
            >>> results = processor.process_with_schema(prompts, MySchema)
            >>> parsed = processor.parse_results_with_schema(MySchema)
            >>> for item in parsed:
            ...     print(item.field_name)
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
                    print(f"❌ Failed to parse output: {text_output[:100]}...")
                    print(f"   Error: {e}")
                    parsed_results.append(None)

        return parsed_results

    # ============================================================================
    # Cleanup and Context Management
    # ============================================================================

    def terminate(self):
        """Terminate all worker processes and clean up resources."""
        if hasattr(self, "stop_event") and self.stop_event is not None:
            self.stop_event.set()

            # Send termination signals
            for _ in range(len(self.processes)):
                try:
                    self.task_queue.put((-1, "TERMINATE", None))
                except:
                    pass

            # Wait for processes to finish
            for process in self.processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                if process.is_alive():
                    process.kill()

            self.processes.clear()

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear references
        self.stop_event = None
        self.task_queue = None
        self.response_queue = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.terminate()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.terminate()
        except:
            pass


# ============================================================================
# Convenience Functions
# ============================================================================


def create_processor(
    gpu_list: List[int], llm: str = "meta-llama/Llama-3.2-3B-Instruct", **kwargs
) -> LLMProcessor:
    """
    Create an LLM processor with sensible defaults.

    Args:
        gpu_list: List of GPU indices to use
        llm: Model identifier
        **kwargs: Additional arguments for LLMProcessor

    Returns:
        LLMProcessor: Ready-to-use processor

    Example:
        >>> processor = create_processor([0, 1])
        >>> results = processor.process_with_schema(prompts, MySchema)
    """
    return LLMProcessor(gpu_list=gpu_list, llm=llm, **kwargs)
