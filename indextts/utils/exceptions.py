class GenerationLengthExceededError(RuntimeError):
    """Raised when autoregressive speech generation does not stop in time."""

    def __init__(
        self,
        *,
        max_mel_tokens: int,
        input_text_tokens: int,
        max_text_tokens_per_segment: int,
    ) -> None:
        self.max_mel_tokens = max_mel_tokens
        self.input_text_tokens = input_text_tokens
        self.max_text_tokens_per_segment = max_text_tokens_per_segment
        super().__init__(
            f"Generation exceeded max_mel_tokens={max_mel_tokens} "
            f"for {input_text_tokens} input text tokens. Consider reducing "
            f"max_text_tokens_per_segment={max_text_tokens_per_segment} or "
            "increasing max_mel_tokens."
        )
