from transformers import pipeline


def initialize_pipeline(model_path: str, torch_dtype):
    """
    Initialize a text generation pipeline

    ### Parameters:
    - model_path (str): The path of a model and tokenizer's weights.

    ### Returns:
    - pipe (TextGenerationPipeline): The pipeline for text generation.
    """

    pipe = pipeline(
        "text-generation",
        model=model_path,
        device_map="auto",
        torch_dtype=torch_dtype
    )

    return pipe


if __name__ == "__main__":
    # Try to initialize a pipeline (note: adjust the model_path).
    # These codes assume the model/ folder as the current working directory.
    import torch
    from transformers.pipelines.text_generation import TextGenerationPipeline

    model_path = "tiny_llama"
    pipe = initialize_pipeline(model_path, torch.bfloat16)

    assert type(pipe) == TextGenerationPipeline
