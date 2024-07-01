from transformers import set_seed
import logging


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("prompting_interface.py")


def is_within_context_length(tokenizer, conversation, context_length: int):
    """
    Check whether a conversation is within a context length
    """
    conv_len = len(
        tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True
        )
    )
    return conv_len <= context_length


def validate_generation_configs(generation_configs):
    if generation_configs["top_k"] == 0:
        del generation_configs["top_k"]
    if generation_configs["top_p"] == 1.0:
        del generation_configs["top_p"]
    if generation_configs["penalty_alpha"] == 0.0:
        del generation_configs["penalty_alpha"]
    if generation_configs["temperature"] == 0.0:
        del generation_configs["temperature"]


def prompt_pipeline(
    pipe,
    conversation,
    context_length=8192,
    max_new_tokens=512,
    do_sample=False,
    top_k=0,
    top_p=1.0,
    penalty_alpha=0.0,
    temperature=0.0,
):
    """
    Prompt the pipeline with a conversation

    ### Parameters:
    - pipe (TextGenerationPipeline): An initialized pipeline.
    - conversation (list[dict[str, str]]): The data type of the model
    - context_length (int): The LLM's context length
    - max_new_tokens (int): Max number of tokens generated for each prompt
    - do_sample (bool): Perform sampling or not
    - top_k (int): The number of tokens to consider when sampling
    - top_p (float): Minimum cumulative probability of tokens being considered
    - penalty_alpha (float): The amount of focus being put to ensure non-repetitiveness
    - temperature (float): Control how sharp the distribution (smaller means sharper)

    ### Returns:
    - conversation (list[dict[str, str]]): The conversation appended with the model's output
    """
    generation_configs = {
        "max_new_tokens": max_new_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "do_sample": do_sample,
        "penalty_alpha": penalty_alpha,
        "temperature": temperature,
        "pad_token_id": pipe.tokenizer.eos_token_id,
    }
    validate_generation_configs(generation_configs)
    try:
        if is_within_context_length(pipe.tokenizer, conversation, context_length):
            set_seed(42)  # Enhance reproducibility for when using sampling
            conversation = pipe(conversation, **generation_configs)[0]["generated_text"]
            return conversation
        else:
            logger.warning(
                "The conversation is more than what the model can handle. Skip processing."
            )
            return [{"role": "user", "content": ""}]
    except:
        logger.warning(
            "Something wrong is happening. Skip processing."
        )
        return [{"role": "user", "content": ""}]


if __name__ == "__main__":
    # Try to prompt a pipeline
    import torch
    from pipeline_initializer import initialize_pipeline

    pipe = initialize_pipeline("tiny_llama", torch.bfloat16)  # Adjust model path & data type
    conversation = [{"role": "user", "content": "Tell me about Illinois!"}]
    output = prompt_pipeline(pipe, conversation, 2048)

    assert len(output) == 2
    assert output[-1]["role"] == "assistant"
