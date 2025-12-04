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

from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.integrations.vllm import (
    build_vllm_logits_processor,
    build_vllm_token_enforcer_tokenizer_data,
)
from pydantic import BaseModel
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

multiprocessing.set_start_method("spawn", force=True)


# Constants
MISTRAL_MODEL = "solidrust/Mistral-7B-Instruct-v0.3-AWQ"
LLAMA_MODEL = "solidrust/Llama-3-8B-Instruct-v0.4-AWQ"
LLAMA_31_8B_INSTRUCT = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
LLAMA_31_70B_INSTRUCT = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
LLAMA_33_70B_INSTRUCT = "casperhansen/llama-3.3-70b-instruct-awq"
LLAMA_33_8B_INSTRUCT = "casperhansen/llama-3-8b-instruct-awq"
QWEN_5B_INSTRUCT = "casperhansen/qwen2-0.5b-instruct-awq"
DEEPSEEK_LLAMA_70B = "casperhansen/deepseek-r1-distill-llama-70b-awq"
QWEN_2_5_B_72B_INSTRUCT = "Qwen/Qwen2.5-72B-Instruct-AWQ"
LLAMA32_1B_INSTRUCT = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA32_3B_INSTRUCT = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA32_3B_INSTRUCT_AWQ = "casperhansen/llama-3.2-3b-instruct-awq"
PHI3_MINI_INSTRUCT = "microsoft/Phi-3-mini-4k-instruct"
PHI4_MINI_INSTRUCT = "microsoft/Phi-4-mini-instruct"
GEMMA3_4B_INSTRUCT = "google/gemma-3-4b-it"
QWEN3_4B = "Qwen/Qwen3-4B"
DOLPHIN_LLAMA32_3B = "cognitivecomputations/Dolphin3.0-Llama3.2-3B"
FT_DOLPHIN_LLAMA32_3B = "/data-fast/data3/common/halo/applications/fusion/finetuning/batch_2/v2/finetuning_3b_1e/save"
Q_DOLPHIN_LLAMA32_3B = (
    "/data-fast/data3/common/halo/data/fusion/finetuning/v3/finetune_3B/quantized/llama_3B_awq"
)
NEMO = "casperhansen/mistral-nemo-instruct-2407-awq"
QWEN_AWQ = "Qwen/Qwen2.5-14B-Instruct-AWQ"


# Setup logging
# logging.basicConfig(
#     level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
# )


def format_prompt(prompt: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


# Helper Functions
def create_batches(prompts: list, batch_size: int) -> list[list] | list:
    return [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]


def extract_text_output(
    prompt_responses: list[list[RequestOutput]] | list[RequestOutput],
) -> list[str]:
    flattened_prompt_responses = flatten(prompt_responses)
    return [response.outputs[0].text for response in flattened_prompt_responses]


def flatten(nested_list: List) -> Generator:
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def get_sampling_params():
    return SamplingParams(
        temperature=0.95,
        top_k=50,
        top_p=0.95,
        max_tokens=4098,
        frequency_penalty=2,
        repetition_penalty=1.1,
    )


def load_llm(**llm_kwargs) -> LLM:
    return LLM(
        model=llm_kwargs["model_path"],
        trust_remote_code=True,
        enforce_eager=True,
        tokenizer=llm_kwargs.get("tokenizer"),  # ADD THIS LINE
        gpu_memory_utilization=llm_kwargs.get("gpu_memory_utilization", 0.9),
        max_model_len=llm_kwargs.get("max_model_len", None),
        dtype=llm_kwargs.get("dtype", "auto"),
        enable_chunked_prefill=True,  # Enable continuous batching
        # Add more args here if needed
    )


# Model Class
class Model:
    def __init__(
        self,
        parser: Optional[CharacterLevelParser] = None,
        **llm_kwargs,
    ):
        self.parser = parser
        self.llm_kwargs = llm_kwargs
        self.model = load_llm(**self.llm_kwargs)
        self.sampling_params = get_sampling_params()

        if self.parser:
            tokenizer_data = build_vllm_token_enforcer_tokenizer_data(self.model)
            logits_processor = build_vllm_logits_processor(tokenizer_data, self.parser)
            self.sampling_params.logits_processors = [logits_processor]

    def generate(
        self, prompts: Union[str, list[str]], use_tqdm: bool = True
    ) -> list[RequestOutput]:
        prompt_list = [prompts] if isinstance(prompts, str) else prompts
        return self.model.generate(
            prompts=prompt_list,
            sampling_params=self.sampling_params,
            use_tqdm=use_tqdm,
        )


# LLM Processor Class
class LLMProcessor:
    def __init__(
        self,
        gpu_list: list[int],
        llm: str = MISTRAL_MODEL,
        multiplicity: int = 1,
        use_tqdm: bool = False,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int | None = 1024,
        parser: Optional[CharacterLevelParser] | None = None,
        tokenizer: str | None = None,  # <-- ADD THIS LINE
        **extra_llm_args,
    ):
        self.gpu_list = gpu_list
        self.llm = llm
        self.multiplicity = multiplicity
        self.use_tqdm = use_tqdm
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.parser = parser
        self.task_queue: Queue = Queue()
        self.response_queue: Queue = Queue()
        self.load_signal_queue: Queue = Queue()
        self.processes = []
        self.stop_event: Event = Event()
        self.responses = []
        self.num_gpus = len(gpu_list)

        tokenizer_path = tokenizer or self.llm
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Assemble LLM initialization arguments
        self.llm_kwargs = {
            "model_path": self.llm,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            **extra_llm_args,
        }

        # ✅ ADD THIS SECTION
        # Pass tokenizer to vLLM if specified separately
        if tokenizer is not None:
            self.llm_kwargs["tokenizer"] = tokenizer

        self.prepare_processes()

    def prepare_processes(self) -> None:
        ctx = get_context("spawn")
        for i in range(self.multiplicity):
            gpu_memory_utilization = min(self.gpu_memory_utilization + i * 0.2, 1.0)
            print(
                f"Starting multiplicity round {i + 1} with GPU memory utilization: {gpu_memory_utilization}"
            )

            for gpu_num in self.gpu_list:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)  # Set it here
                process = ctx.Process(
                    target=LLMProcessor.worker,
                    args=(
                        self.llm_kwargs,
                        gpu_num,
                        # self.max_model_len,
                        # gpu_memory_utilization,
                        self.task_queue,
                        self.response_queue,
                        self.stop_event,
                        self.load_signal_queue,
                        i,  # multiplicity index
                        self.parser,
                    ),
                )
                process.start()
                self.processes.append(process)

            # Wait for all models in this multiplicity round to load before starting the next round
            self.wait_for_models_to_load(expected_count=self.num_gpus)

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
    ):
        try:
            llm_kwargs["gpu_num"] = gpu_num  # If you still want this tracked
            model = Model(parser=parser, **llm_kwargs)
            load_signal_queue.put((gpu_num, multiplicity_index))
        except Exception as e:
            load_signal_queue.put(
                f"Error loading model on GPU {gpu_num} (multiplicity {multiplicity_index}): {str(e)}"
            )
            return

        while not stop_event.is_set():
            try:
                request_id, request, corr_id = task_queue.get(timeout=1)
                response = model.generate(prompts=request, use_tqdm=False)
                response_queue.put((request_id, response, corr_id, request))
            except Empty:
                continue
            except Exception:
                response_queue.put((request_id, None, corr_id, request))

    def wait_for_models_to_load(self, expected_count, timeout=None):
        """
        Wait for a specific number of models to be loaded.

        :param expected_count: Number of models to wait for
        :param timeout: Maximum time to wait in seconds. If None, wait indefinitely.
        :return: True if all expected models loaded successfully, False otherwise.
        """
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

    def process_requests(
        self,
        prompts: Union[str, List[str]],
        batch_size: int = 25,
        formatted: bool = False,
        on_batch_end=None,
        timeout=10,
    ):
        prompt_list = [prompts] if isinstance(prompts, str) else prompts
        if formatted:
            formatted_prompt_list = prompt_list
        else:
            formatted_prompt_list = [
                format_prompt(prompt=prompt, tokenizer=self.tokenizer) for prompt in prompt_list
            ]

        batch_prompts = {
            request_id: batch
            for request_id, batch in enumerate(
                create_batches(prompts=formatted_prompt_list, batch_size=batch_size)
            )
        }
        book_keeping_indexes = {
            request_id: batch
            for request_id, batch in enumerate(
                create_batches(prompts=list(range(0, len(prompt_list))), batch_size=batch_size)
            )
        }
        assert len(batch_prompts) == len(book_keeping_indexes)

        total_requests = len(batch_prompts)
        response_counter = 0
        current_corr_id = uuid.uuid4()
        for request_id, prompts_ in batch_prompts.items():
            self.task_queue.put((request_id, prompts_, current_corr_id))

        processed_responses = {}
        # completed = True
        with tqdm(
            total=total_requests,
            colour="#B48EAD",
            leave=False,
            desc=f"Process requests {current_corr_id}",
        ) as pbar:
            datetime.now()
            while response_counter < total_requests and not self.stop_event.is_set():
                try:
                    request_id, response, corr_id, prompts_ = self.response_queue.get(timeout=1)
                    if response is None:
                        print(f"Failed on request_id {request_id}")
                        self.task_queue.put((request_id, prompts_, current_corr_id))
                        continue

                    if current_corr_id != corr_id:
                        # print(f"\n{current_corr_id} does not match {corr_id}")
                        raise RuntimeError(
                            f"\nCurrent correlation id {current_corr_id} does not match the result queue correlation id {corr_id}"
                        )
                    response_counter += 1

                    if batch_prompts[request_id] != prompts_:
                        for r_id, bp in batch_prompts.items():
                            if bp == prompts_:
                                print(f"\nMatch {r_id} versus expected {request_id}")
                        raise RuntimeError("Returned values does not match with expectations")

                    if on_batch_end:
                        on_batch_end(
                            batch_prompts[request_id],
                            book_keeping_indexes[request_id],
                            response,
                        )
                    processed_responses[request_id] = response
                    pbar.update(1)
                    datetime.now()

                except Empty:
                    continue
                    # if (datetime.now() - start).total_seconds() >= timeout:
                    # # completed = False
                    # # self.stop_event.set()
                    # break
                    # else:
                    # continue
                except Exception as e:
                    print(f"\nasdf: {e}")

        self.responses = [
            processed_responses[request_id] for request_id in sorted(processed_responses.keys())
        ]

    def terminate(self):
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


class GuidedModel(Model):
    """
    Enhanced Model class with guided decoding support for structured output generation.

    Extends the base Model class to support Pydantic schema-based guided decoding using
    vLLM's GuidedDecodingParams. This ensures generated outputs conform to specified
    JSON schemas, enabling reliable structured data extraction.

    Args:
        schema: Optional Pydantic BaseModel class defining the output structure
        guided_config: Optional dictionary with guided decoding configuration including
                      temperature, top_k, top_p, max_tokens, and other sampling parameters
        **llm_kwargs: Additional arguments passed to the base Model class

    Attributes:
        schema: The Pydantic schema used for guided decoding
        guided_config: Configuration dictionary for guided generation
        sampling_params: vLLM SamplingParams with guided decoding constraints applied

    Example:
        >>> class MySchema(BaseModel):
        ...     name: str
        ...     score: float
        >>> model = GuidedModel(schema=MySchema, guided_config={'temperature': 0.1})
    """

    def __init__(
        self,
        schema: Optional[Type[BaseModel]] = None,
        guided_config: Optional[Dict] = None,
        **llm_kwargs,
    ):
        # Store schema and config before calling parent init
        self.schema = schema
        self.guided_config = guided_config or {}

        # Initialize base model
        super().__init__(**llm_kwargs)

        # Override sampling params if we have guided decoding config
        if self.schema or self.guided_config:
            self.setup_guided_decoding()

    def setup_guided_decoding(self):
        """
        Configure sampling parameters with guided decoding constraints.

        Creates SamplingParams with the specified guided configuration and applies
        JSON schema-based guided decoding if a schema is provided. The guided decoding
        ensures all generated outputs conform to the specified Pydantic schema structure.

        Sets:
            - Base sampling parameters (temperature, top_k, top_p, max_tokens)
            - GuidedDecodingParams with JSON schema if schema is provided
            - Any additional guided decoding parameters from guided_config

        Note:
            This method is automatically called during initialization if schema
            or guided_config are provided.
        """  # Base sampling parameters
        sampling_params = SamplingParams(
            temperature=self.guided_config.get("temperature", 0.1),
            top_k=self.guided_config.get("top_k", 50),
            top_p=self.guided_config.get("top_p", 0.95),
            max_tokens=self.guided_config.get("max_tokens", 1000),
        )

        # Add guided decoding if schema is provided
        if self.schema:
            json_schema = self.schema.model_json_schema()
            guided_decoding_params = GuidedDecodingParams(json=json_schema)
            sampling_params.guided_decoding = guided_decoding_params
            print(
                f"✓ Guided decoding setup for schema with {len(json_schema.get('properties', {}))} fields"
            )

        # Apply any additional guided decoding params from config
        if "guided_decoding" in self.guided_config:
            sampling_params.guided_decoding = self.guided_config["guided_decoding"]

        self.sampling_params = sampling_params


class SchemaProcessor(LLMProcessor):
    """
    Enhanced LLMProcessor with schema-based guided decoding for structured output generation.

    Extends the base LLMProcessor to support multiprocessing inference with Pydantic schema
    constraints. Each worker process uses GuidedModel to ensure all generated outputs
    conform to the specified schema structure.

    Args:
        schema: Pydantic BaseModel class defining the required output structure
        gpu_list: List of GPU indices to use for distributed inference
        llm: Model identifier string (e.g., "meta-llama/Llama-3.2-3B-Instruct")
        multiplicity: Number of model instances per GPU for increased throughput
        use_tqdm: Whether to display progress bars during processing
        gpu_memory_utilization: GPU memory utilization fraction (0.0-1.0)
        max_model_len: Maximum sequence length for the model
        guided_config: Optional dictionary with guided decoding parameters
        **extra_llm_args: Additional arguments for LLM initialization

    Attributes:
        schema: The Pydantic schema used across all worker processes
        guided_config: Configuration dictionary for guided generation

    Example:
        >>> class LabelDef(BaseModel):
        ...     definition: str
        ...     scope: str
        >>> processor = SchemaProcessor(schema=LabelDef, gpu_list=[0,1])
        >>> processor.process_requests(prompts=["Define teamwork"])
        >>> results = processor.get_parsed_results()
    """

    def __init__(
        self,
        schema: Type[BaseModel],
        gpu_list: list[int],
        llm: str = LLAMA_33_70B_INSTRUCT,
        multiplicity: int = 1,
        use_tqdm: bool = False,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 2048,
        guided_config: Optional[Dict] = None,
        **extra_llm_args,
    ):
        self.schema = schema
        self.guided_config = guided_config or {}

        # Initialize parent class
        super().__init__(
            gpu_list=gpu_list,
            llm=llm,
            multiplicity=multiplicity,
            use_tqdm=use_tqdm,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            **extra_llm_args,
        )

        print(f"🎯 SchemaProcessor initialized for schema: {schema.__name__}")

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
        schema=None,
        guided_config=None,
    ):
        """
        Worker process function for distributed schema-based inference.

        Creates a GuidedModel instance in each worker process and processes inference
        requests from the task queue. Each generated response is constrained by the
        provided schema to ensure structured output.

        Args:
            llm_kwargs: Dictionary of LLM initialization parameters
            gpu_num: GPU index for this worker process
            task_queue: Queue containing inference requests
            response_queue: Queue for returning inference results
            stop_event: Event signal for graceful worker termination
            load_signal_queue: Queue for signaling successful model loading
            multiplicity_index: Index for this multiplicity instance
            parser: Optional text parser (legacy parameter, typically None)
            schema: Pydantic BaseModel class for output structure
            guided_config: Configuration dictionary for guided decoding

        Note:
            This method runs in separate processes and handles model loading,
            request processing, and error recovery autonomously.
        """
        try:
            llm_kwargs["gpu_num"] = gpu_num
            model = GuidedModel(schema=schema, guided_config=guided_config, **llm_kwargs)
            load_signal_queue.put((gpu_num, multiplicity_index))
        except Exception as e:
            load_signal_queue.put(
                f"Error loading model on GPU {gpu_num} (multiplicity {multiplicity_index}): {str(e)}"
            )
            return

        while not stop_event.is_set():
            try:
                request_id, request, corr_id = task_queue.get(timeout=1)
                response = model.generate(prompts=request, use_tqdm=False)
                response_queue.put((request_id, response, corr_id, request))
            except Empty:
                continue
            except Exception:
                response_queue.put((request_id, None, corr_id, request))

    def prepare_processes(self) -> None:
        """
        Initialize worker processes for distributed schema-based inference.

        Creates and starts worker processes across specified GPUs, with each process
        loading a GuidedModel configured for the target schema. Implements multiplicity
        support for increased throughput by running multiple model instances per GPU.

        Process creation follows these steps:
        1. Iterate through multiplicity rounds to stagger GPU memory usage
        2. Create worker processes for each GPU in each round
        3. Pass schema and guided_config to each worker
        4. Wait for model loading confirmation before proceeding

        Raises:
            RuntimeError: If model loading fails in any worker process

        Note:
            This method blocks until all models are successfully loaded
            across all worker processes.
        """
        ctx = get_context("spawn")

        for i in range(self.multiplicity):
            gpu_memory_utilization = min(self.gpu_memory_utilization + i * 0.2, 1.0)
            print(
                f"Starting multiplicity round {i + 1} with GPU memory utilization: {gpu_memory_utilization}"
            )

            for gpu_num in self.gpu_list:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
                process = ctx.Process(
                    target=SchemaProcessor.worker,
                    args=(
                        self.llm_kwargs,
                        gpu_num,
                        self.task_queue,
                        self.response_queue,
                        self.stop_event,
                        self.load_signal_queue,
                        i,
                        self.parser,
                        self.schema,
                        self.guided_config,
                    ),
                )
                process.start()
                self.processes.append(process)

            self.wait_for_models_to_load(expected_count=self.num_gpus)

    def extract_all_batch_outputs(self, response):
        """
        Extract all text outputs from a single or batched inference response.

        Handles both single responses and batched responses to ensure all generated
        texts are captured. This is crucial for batch processing where multiple
        prompts are processed together and return multiple outputs.

        Args:
            response: vLLM response object (RequestOutput or list of RequestOutput)
                     Can be a single response with multiple outputs or a list of responses

        Returns:
            List[str]: All text outputs extracted from the response(s)

        Example:
            >>> # Single response with multiple outputs (batch_size > 1)
            >>> texts = processor.extract_all_batch_outputs(response)
            >>> # Returns: ["output1", "output2", "output3"]

        Note:
            This method is designed to handle the response structure variations
            that occur with different batch sizes in vLLM.
        """
        all_texts = []

        if isinstance(response, list):
            # Multiple response objects
            for resp in response:
                if hasattr(resp, "outputs"):
                    for output in resp.outputs:
                        all_texts.append(output.text)
        else:
            # Single response object with multiple outputs
            if hasattr(response, "outputs"):
                for output in response.outputs:
                    all_texts.append(output.text)

        return all_texts

    def get_parsed_results(self, validate: bool = True) -> List[Union[BaseModel, Dict, str]]:
        """
        Parse and validate all inference results according to the schema.

        Processes all responses from the inference queue, extracts text outputs,
        parses JSON content, and validates against the Pydantic schema. Handles
        both single and batched responses automatically.

        Args:
            validate: Whether to validate outputs against the Pydantic schema.
                     If True, returns validated Pydantic objects.
                     If False, returns raw dictionaries.

        Returns:
            List[Union[BaseModel, Dict, str, None]]: Parsed results where:
                - BaseModel objects if validation succeeds
                - Dict objects if validate=False and JSON parsing succeeds
                - str if JSON parsing fails (raw text)
                - None if all parsing attempts fail

        Raises:
            json.JSONDecodeError: Logged but handled gracefully
            pydantic.ValidationError: Logged but handled gracefully

        Example:
            >>> results = processor.get_parsed_results()
            >>> for result in results:
            ...     if result:
            ...         print(f"Definition: {result.definition}")

        Note:
            Failed parsing attempts are logged with error details and the
            problematic text snippet for debugging purposes.
        """
        parsed_results = []

        for response in tqdm(self.responses):
            # Use custom extraction for batches
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

                    if validate and self.schema:
                        validated_obj = self.schema(**json_data)
                        parsed_results.append(validated_obj)
                    else:
                        parsed_results.append(json_data)

                except Exception as e:
                    print(f"Failed to parse output: {text_output[:100]}...")
                    print(f"Error: {e}")
                    parsed_results.append(None)

        return parsed_results


class FlexibleGuidedModel:
    """
    Enhanced Model class with runtime schema switching for guided decoding.

    Unlike GuidedModel which is locked to a single schema at initialization,
    this model can switch schemas at generation time, making it reusable
    for different structured output tasks.

    Args:
        default_guided_config: Optional default configuration for guided decoding
        **llm_kwargs: Additional arguments passed to the base Model class

    Example:
        >>> model = FlexibleGuidedModel()
        >>> # Use with PersonSchema
        >>> result1 = model.generate_with_schema(prompts, PersonSchema)
        >>> # Later use with ProductSchema
        >>> result2 = model.generate_with_schema(prompts, ProductSchema)
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
        """Load the LLM model."""
        return LLM(
            model=llm_kwargs["model_path"],
            tokenizer=llm_kwargs.get("tokenizer"),  # ADD THIS LINE
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=llm_kwargs.get("gpu_memory_utilization", 0.9),
            max_model_len=llm_kwargs.get("max_model_len", None),
            dtype=llm_kwargs.get("dtype", "auto"),
        )

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

        Args:
            json_schema: Optional JSON schema dictionary for output structure
            guided_config: Optional configuration overrides for this generation

        Returns:
            SamplingParams: Configured sampling parameters with or without guided decoding
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
        """
        Generate outputs with optional JSON schema constraints.

        Args:
            prompts: Input prompts for generation
            json_schema: Optional JSON schema dictionary for guided decoding
            guided_config: Optional configuration for this generation
            use_tqdm: Whether to show progress bar

        Returns:
            List[RequestOutput]: Generated responses
        """
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
    Enhanced LLMProcessor with runtime schema switching for structured output generation.

    Unlike SchemaProcessor which is bound to a single schema, this processor can work
    with different schemas for each request batch, making it highly reusable across
    different tasks and output formats.

    Args:
        gpu_list: List of GPU indices to use for distributed inference
        llm: Model identifier string
        multiplicity: Number of model instances per GPU
        use_tqdm: Whether to display progress bars
        gpu_memory_utilization: GPU memory utilization fraction
        max_model_len: Maximum sequence length
        default_guided_config: Default guided decoding configuration
        **extra_llm_args: Additional LLM initialization arguments

    Example:
        >>> processor = FlexibleSchemaProcessor(gpu_list=[0, 1])
        >>>
        >>> # Use with PersonSchema
        >>> results1 = processor.process_with_schema(prompts1, PersonSchema)
        >>>
        >>> # Later use with ProductSchema - same processor!
        >>> results2 = processor.process_with_schema(prompts2, ProductSchema)
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
        **extra_llm_args,
    ):
        self.gpu_list = gpu_list
        self.llm = llm
        self.multiplicity = multiplicity
        self.use_tqdm = use_tqdm
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.default_guided_config = default_guided_config or {}

        # Use provided tokenizer or default to model path
        tokenizer_path = tokenizer or self.llm
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

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
            **extra_llm_args,
        }

        # ADD THIS: Pass tokenizer to vLLM if specified separately

        if tokenizer is not None:
            self.llm_kwargs["tokenizer"] = tokenizer

        self.prepare_processes()

        print("🔄 FlexibleSchemaProcessor initialized - ready for runtime schema switching")

    def format_prompt(self, prompt: str) -> str:
        """Format prompt using tokenizer chat template."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )

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

        for i in range(self.multiplicity):
            gpu_memory_utilization = min(self.gpu_memory_utilization + i * 0.2, 1.0)
            print(
                f"Starting multiplicity round {i + 1} with GPU memory utilization: {gpu_memory_utilization}"
            )

            for gpu_num in self.gpu_list:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
                process = ctx.Process(
                    target=FlexibleSchemaProcessor.worker,
                    args=(
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
            formatted: Whether prompts are already formatted
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
            # for response in tqdm(responses_to_parse, desc=f"Parsing with {schema.__name__}"):
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


# Convenience function for easy usage
def create_flexible_processor(
    gpu_list: List[int], llm: str = "meta-llama/Llama-3.2-3B-Instruct", **kwargs
) -> FlexibleSchemaProcessor:
    """
    Create a flexible schema processor with sensible defaults.

    Args:
        gpu_list: List of GPU indices to use
        llm: Model identifier
        **kwargs: Additional arguments for FlexibleSchemaProcessor

    Returns:
        FlexibleSchemaProcessor: Ready-to-use processor

    Example:
        >>> processor = create_flexible_processor([0, 1])
        >>> # Use immediately with any schema
    """
    return FlexibleSchemaProcessor(gpu_list=gpu_list, llm=llm, **kwargs)


"""
FlexibleSchemaJsonProcessor - Process prompts with JSON schema dictionaries

This module extends the flexible schema processing capabilities to work directly
with JSON schema dictionaries, bypassing the need for Pydantic model classes.
"""


class FlexibleSchemaJsonProcessor:
    """
    Enhanced processor that works directly with JSON schema dictionaries.

    Unlike SchemaProcessor (bound to single Pydantic schema) or FlexibleSchemaProcessor
    (requires Pydantic classes at runtime), this processor accepts pre-generated JSON
    schemas, making it ideal for:
    - Working with schemas from external sources
    - Dynamic schema generation
    - Systems where Pydantic models aren't available
    - Integration with schema registries

    Args:
        gpu_list: List of GPU indices to use for distributed inference
        llm: Model identifier string
        multiplicity: Number of model instances per GPU
        use_tqdm: Whether to display progress bars
        gpu_memory_utilization: GPU memory utilization fraction (0.0-1.0)
        max_model_len: Maximum sequence length
        default_guided_config: Default guided decoding configuration
        **extra_llm_args: Additional LLM initialization arguments

    Example:
        >>> # Generate JSON schema from Pydantic model
        >>> user_schema = User.model_json_schema()
        >>>
        >>> # Create processor
        >>> processor = FlexibleSchemaJsonProcessor(gpu_list=[0, 1])
        >>>
        >>> # Process with JSON schema
        >>> results = processor.process_with_json_schema(
        ...     prompts=["What is the capital of Canada?"],
        ...     json_schema=user_schema
        ... )
        >>>
        >>> # Parse results
        >>> parsed = processor.parse_results()
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
        **extra_llm_args,
    ):
        self.gpu_list = gpu_list
        self.llm = llm
        self.multiplicity = multiplicity
        self.use_tqdm = use_tqdm
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.default_guided_config = default_guided_config or {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
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
            **extra_llm_args,
        }
        self.prepare_processes()

        print("🔄 FlexibleSchemaJsonProcessor initialized - ready for JSON schema processing")

    def format_prompt(self, prompt: str) -> str:
        """Format prompt using tokenizer chat template."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )

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
        default_guided_config=None,
    ):
        """
        Worker process with flexible JSON schema support.

        Creates a FlexibleGuidedModel that handles JSON schema-based guided decoding.
        Each task can specify its own JSON schema dictionary and guided config.
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
        """Initialize worker processes with flexible JSON schema support."""
        ctx = get_context("spawn")

        for i in range(self.multiplicity):
            gpu_memory_utilization = min(self.gpu_memory_utilization + i * 0.2, 1.0)
            print(
                f"Starting multiplicity round {i + 1} with GPU memory utilization: {gpu_memory_utilization}"
            )

            for gpu_num in self.gpu_list:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
                process = ctx.Process(
                    target=FlexibleSchemaJsonProcessor.worker,
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

    def process_with_json_schema(
        self,
        prompts: Union[str, List[str]],
        json_schema: Optional[Dict] = None,
        batch_size: int = 25,
        formatted: bool = False,
        guided_config: Optional[Dict] = None,
        on_batch_end=None,
        timeout=10,
    ) -> List[RequestOutput]:
        """
        Process requests with a JSON schema dictionary for guided decoding.

        Args:
            prompts: Input prompts to process
            json_schema: Optional JSON schema dictionary for structured output.
                        Should follow JSON Schema specification with properties,
                        required fields, types, etc.
            batch_size: Number of prompts per batch
            formatted: Whether prompts are already formatted with chat template
            guided_config: Optional guided decoding configuration (temperature, etc.)
            on_batch_end: Optional callback function(batch_prompts, indexes, response)
            timeout: Request timeout in seconds

        Returns:
            List[RequestOutput]: Generated responses constrained by the schema

        Example:
            >>> json_schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "city": {"type": "string"},
            ...         "country": {"type": "string"}
            ...     },
            ...     "required": ["city", "country"]
            ... }
            >>> results = processor.process_with_json_schema(
            ...     prompts=["What is the capital of Canada?"],
            ...     json_schema=json_schema
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

        # Validate JSON schema if provided
        if json_schema:
            schema_name = json_schema.get("title", "CustomSchema")
            print(f"📋 Processing with JSON schema: {schema_name}")
            if "properties" in json_schema:
                print(f"   Properties: {list(json_schema['properties'].keys())}")

        # Submit tasks with JSON schema information
        for request_id, prompts_ in batch_prompts.items():
            self.task_queue.put(
                (request_id, prompts_, current_corr_id, json_schema, guided_config)
            )

        processed_responses = {}

        schema_desc = json_schema.get("title", "JSON Schema") if json_schema else "No Schema"
        with tqdm(
            total=total_requests,
            colour="#B48EAD",
            leave=False,
            desc=f"Processing with {schema_desc} [{current_corr_id}]",
        ) as pbar:
            while response_counter < total_requests and not self.stop_event.is_set():
                try:
                    request_id, response, corr_id, prompts_ = self.response_queue.get(timeout=1)

                    if response is None:
                        print(f"Failed on request_id {request_id}, retrying...")
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
                    print(f"Processing error: {e}")

        self.responses = [
            processed_responses[request_id] for request_id in sorted(processed_responses.keys())
        ]

        return self.responses

    def parse_results(
        self, responses: Optional[List[RequestOutput]] = None, strict: bool = False
    ) -> List[Union[Dict, str, None]]:
        """
        Parse results from schema-guided generation.

        Since we work with JSON schemas directly (not Pydantic models),
        this method returns dictionaries rather than validated objects.

        Args:
            responses: Optional specific responses to parse (defaults to last processed)
            strict: If True, raise exceptions on parsing failures. If False, return None
                   for failed parses and continue.

        Returns:
            List[Union[Dict, str, None]]: Parsed results where:
                - Dict if JSON parsing succeeds
                - str if JSON parsing fails (returns raw text)
                - None if all parsing attempts fail

        Example:
            >>> results = processor.parse_results()
            >>> for result in results:
            ...     if isinstance(result, dict):
            ...         print(f"City: {result.get('city')}")
            ...     elif result is None:
            ...         print("Failed to parse")
        """
        responses_to_parse = responses or self.responses
        parsed_results = []

        for response in tqdm(responses_to_parse, desc="Parsing JSON results"):
            all_texts = self.extract_all_batch_outputs(response)

            for text_output in all_texts:
                try:
                    # Clean up text
                    text_output = text_output.strip()

                    # Remove markdown code blocks if present
                    if text_output.startswith("```json"):
                        text_output = (
                            text_output.replace("```json", "").replace("```", "").strip()
                        )
                    elif text_output.startswith("```"):
                        text_output = text_output.replace("```", "").strip()

                    # Parse JSON
                    json_data = json.loads(text_output)
                    parsed_results.append(json_data)

                except json.JSONDecodeError as e:
                    if strict:
                        raise
                    print(f"JSON decode error: {e}")
                    print(f"Failed text: {text_output[:200]}...")
                    parsed_results.append(text_output)  # Return raw text

                except Exception as e:
                    if strict:
                        raise
                    print(f"Unexpected parsing error: {e}")
                    print(f"Failed text: {text_output[:200]}...")
                    parsed_results.append(None)

        return parsed_results

    def extract_all_batch_outputs(self, response):
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

    def validate_against_schema(
        self, parsed_results: List[Dict], json_schema: Dict, strict: bool = False
    ) -> List[bool]:
        """
        Validate parsed results against a JSON schema.

        Note: Requires jsonschema library. Install with: pip install jsonschema

        Args:
            parsed_results: List of parsed dictionary results
            json_schema: JSON schema to validate against
            strict: If True, raise exception on first validation failure

        Returns:
            List[bool]: Validation status for each result

        Example:
            >>> results = processor.parse_results()
            >>> valid = processor.validate_against_schema(results, user_schema)
            >>> valid_results = [r for r, v in zip(results, valid) if v]
        """
        try:
            from jsonschema import ValidationError, validate
        except ImportError:
            raise ImportError(
                "jsonschema library required for validation. "
                "Install with: pip install jsonschema"
            )

        validation_results = []
        for i, result in enumerate(parsed_results):
            if not isinstance(result, dict):
                validation_results.append(False)
                continue

            try:
                validate(instance=result, schema=json_schema)
                validation_results.append(True)
            except ValidationError as e:
                if strict:
                    raise
                print(f"Validation failed for result {i}: {e.message}")
                validation_results.append(False)

        return validation_results

    def terminate(self):
        """Terminate all worker processes and clean up resources."""
        if hasattr(self, "stop_event") and self.stop_event is not None:
            self.stop_event.set()

        for _ in range(len(self.processes)):
            try:
                self.task_queue.put((-1, "TERMINATE", None))
            except:
                pass

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

        # Clear references
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
            pass


def create_json_schema_processor(
    gpu_list: List[int], llm: str = "meta-llama/Llama-3.2-3B-Instruct", **kwargs
) -> FlexibleSchemaJsonProcessor:
    """
    Create a flexible JSON schema processor with sensible defaults.

    Args:
        gpu_list: List of GPU indices to use
        llm: Model identifier
        **kwargs: Additional arguments for FlexibleSchemaJsonProcessor

    Returns:
        FlexibleSchemaJsonProcessor: Ready-to-use processor

    Example:
        >>> processor = create_json_schema_processor([0, 1])
        >>> # Ready to use with any JSON schema
    """
    return FlexibleSchemaJsonProcessor(gpu_list=gpu_list, llm=llm, **kwargs)


"""
Enhanced FlexibleSchemaJsonProcessor with Per-Prompt Schema Support

This module extends FlexibleSchemaJsonProcessor to support heterogeneous batches
where each prompt can have its own unique JSON schema.
"""


class HeterogeneousSchemaProcessor(FlexibleSchemaJsonProcessor):
    """
    Enhanced processor that supports different schemas for different prompts.

    Extends FlexibleSchemaJsonProcessor to handle heterogeneous batches where
    each prompt can have its own unique JSON schema. Automatically optimizes
    by grouping prompts with the same schema together.

    Example:
        >>> processor = HeterogeneousSchemaProcessor(gpu_list=[0])
        >>>
        >>> # Different schemas for different prompts
        >>> prompt_schema_pairs = [
        ...     ("Extract person info", person_schema),
        ...     ("Extract company info", company_schema),
        ...     ("Extract location info", location_schema),
        ...     ("Extract another person", person_schema),  # Same as first
        ... ]
        >>>
        >>> # Process all in one call - automatically batches by schema
        >>> results = processor.process_heterogeneous_batch(prompt_schema_pairs)
    """

    def process_heterogeneous_batch(
        self,
        prompt_schema_pairs: List[Tuple[str, Optional[Dict]]],
        batch_size: int = 25,
        formatted: bool = False,
        guided_config: Optional[Dict] = None,
        preserve_order: bool = True,
    ) -> List[Dict]:
        """
        Process prompts where each prompt has its own schema.

        This method automatically groups prompts by schema for efficient batch
        processing, then returns results in the original order (if preserve_order=True).

        Args:
            prompt_schema_pairs: List of (prompt, json_schema) tuples. Each prompt
                                can have a different schema.
            batch_size: Maximum batch size for processing each schema group
            formatted: Whether prompts are already formatted with chat template
            guided_config: Optional guided decoding configuration
            preserve_order: If True, return results in original prompt order.
                          If False, return grouped by schema (faster).

        Returns:
            List[Dict]: Parsed results in original order (or grouped if preserve_order=False)

        Example:
            >>> pairs = [
            ...     ("Describe John Doe", person_schema),
            ...     ("Info about Acme Corp", company_schema),
            ...     ("Tell me about Jane Smith", person_schema),
            ... ]
            >>> results = processor.process_heterogeneous_batch(pairs)
            >>> # Returns 3 results in original order
        """
        print(f"\n🔀 Processing heterogeneous batch of {len(prompt_schema_pairs)} prompts")

        # Group prompts by schema
        schema_groups = self._group_by_schema(prompt_schema_pairs)

        print(f"📊 Grouped into {len(schema_groups)} unique schema(s)")
        for i, (schema_key, group_data) in enumerate(schema_groups.items(), 1):
            print(f"   Group {i}: {len(group_data['prompts'])} prompts")

        # Process each schema group
        all_results = []
        for schema_key, group_data in schema_groups.items():
            schema = group_data["schema"]
            prompts = group_data["prompts"]
            original_indices = group_data["indices"]

            schema_name = schema.get("title", "Schema") if schema else "No Schema"
            print(f"\n🔄 Processing {len(prompts)} prompts with {schema_name}")

            # Process this group
            self.process_with_json_schema(
                prompts=prompts,
                json_schema=schema,
                batch_size=batch_size,
                formatted=formatted,
                guided_config=guided_config,
            )

            # Parse results for this group
            group_results = self.parse_results()

            # Store with original indices
            for idx, result in zip(original_indices, group_results):
                all_results.append((idx, result))

        # Sort by original order if requested
        if preserve_order:
            all_results.sort(key=lambda x: x[0])

        # Return just the results (not the indices)
        final_results = [result for _, result in all_results]

        print(f"\n✅ Completed processing {len(final_results)} prompts")
        return final_results

    def _group_by_schema(
        self, prompt_schema_pairs: List[Tuple[str, Optional[Dict]]]
    ) -> Dict[str, Dict]:
        """
        Group prompts by their schema for efficient batch processing.

        Args:
            prompt_schema_pairs: List of (prompt, schema) tuples

        Returns:
            Dictionary mapping schema_key -> {schema, prompts, indices}
        """
        schema_groups = defaultdict(lambda: {"schema": None, "prompts": [], "indices": []})

        for idx, (prompt, schema) in enumerate(prompt_schema_pairs):
            # Create a hashable key for the schema
            schema_key = self._schema_to_key(schema)

            if schema_groups[schema_key]["schema"] is None:
                schema_groups[schema_key]["schema"] = schema

            schema_groups[schema_key]["prompts"].append(prompt)
            schema_groups[schema_key]["indices"].append(idx)

        return dict(schema_groups)

    def _schema_to_key(self, schema: Optional[Dict]) -> str:
        """
        Convert a schema to a hashable key for grouping.

        Args:
            schema: JSON schema dictionary or None

        Returns:
            String key representing the schema
        """
        if schema is None:
            return "no_schema"

        # Use JSON string as key (sorted for consistency)
        return json.dumps(schema, sort_keys=True)

    def process_sequential(
        self,
        prompt_schema_pairs: List[Tuple[str, Optional[Dict]]],
        formatted: bool = False,
        guided_config: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Process prompts sequentially, one at a time with their own schema.

        This is less efficient than process_heterogeneous_batch but simpler.
        Use this when you have very few prompts or need strict sequential processing.

        Args:
            prompt_schema_pairs: List of (prompt, json_schema) tuples
            formatted: Whether prompts are already formatted
            guided_config: Optional guided decoding configuration

        Returns:
            List[Dict]: Parsed results in original order

        Example:
            >>> pairs = [
            ...     ("Extract person", person_schema),
            ...     ("Extract company", company_schema),
            ... ]
            >>> results = processor.process_sequential(pairs)
        """
        print(f"\n⏭️  Processing {len(prompt_schema_pairs)} prompts sequentially")

        all_results = []
        for i, (prompt, schema) in enumerate(prompt_schema_pairs, 1):
            schema_name = schema.get("title", "Schema") if schema else "No Schema"
            print(f"   [{i}/{len(prompt_schema_pairs)}] Processing with {schema_name}")

            self.process_with_json_schema(
                prompts=[prompt],
                json_schema=schema,
                batch_size=1,
                formatted=formatted,
                guided_config=guided_config,
            )

            result = self.parse_results()
            all_results.extend(result)

        print("✅ Completed sequential processing")
        return all_results


# Convenience function
def create_heterogeneous_processor(
    gpu_list: List[int], llm: str = "meta-llama/Llama-3.2-3B-Instruct", **kwargs
) -> HeterogeneousSchemaProcessor:
    """
    Create a heterogeneous schema processor with sensible defaults.

    Args:
        gpu_list: List of GPU indices to use
        llm: Model identifier
        **kwargs: Additional arguments for processor

    Returns:
        HeterogeneousSchemaProcessor: Ready-to-use processor

    Example:
        >>> processor = create_heterogeneous_processor([0, 1])
        >>> # Ready to process heterogeneous batches
    """
    return HeterogeneousSchemaProcessor(gpu_list=gpu_list, llm=llm, **kwargs)
