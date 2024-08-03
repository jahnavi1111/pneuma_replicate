from transformers import pipeline


def initialize_pipeline(model_path: str, torch_dtype, context_length=8192, hf_token=""):
    """
    Initialize a text generation pipeline

    ### Parameters:
    - model_path (str): The path of a model and tokenizer's weights

    ### Returns:
    - pipe (TextGenerationPipeline): The pipeline for text generation
    - hf_token (str): HuggingFace token to access gated model
    """
    pipe = pipeline(
        "text-generation", model=model_path, device_map="auto", torch_dtype=torch_dtype, token=hf_token
    )
    pipe.tokenizer.model_max_length = context_length
    return pipe


if __name__ == "__main__":
    import torch
    from transformers.pipelines.text_generation import TextGenerationPipeline

    model_path = "tiny_llama"  # Adjust model path
    pipe = initialize_pipeline(model_path, torch.bfloat16)

    assert type(pipe) == TextGenerationPipeline
