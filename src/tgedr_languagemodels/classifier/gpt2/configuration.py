from transformers import PretrainedConfig


class ClassifierConfiguration(PretrainedConfig):
    """Hugging Face compatible configuration for GPT2Classifier.

    This wraps the BaseClassifierConfig to be compatible with PreTrainedModel.
    """

    model_type = "classifier-gpt2"

    def __init__(
        self,
        vocabulary_size: int = 50257,
        embeddings_dimension: int = 768,
        context_length: int = 1024,
        n_layers: int = 12,
        drop_rate: float = 0.1,
        stride: int = 1,
        n_heads: int = 12,
        qkv_bias: bool = False,
        n_classes: int = 2,
        **kwargs,
    ):
        """Initialize GPT2 classifier configuration.

        Parameters
        ----------
        vocabulary_size : int
            Number of tokens in the vocabulary.
        embeddings_dimension : int
            Dimension of token and positional embeddings.
        context_length : int
            Maximum number of tokens in a sequence.
        n_layers : int
            Number of transformer blocks.
        drop_rate : float
            Dropout probability for regularization.
        stride : int
            Stride used for sliding window tokenization.
        n_heads : int
            Number of attention heads per transformer block.
        qkv_bias : bool
            Whether to use bias in query, key, and value projections.
        n_classes : int
            Number of output classes for classification.
        """
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.embeddings_dimension = embeddings_dimension
        self.context_length = context_length
        self.n_layers = n_layers
        self.drop_rate = drop_rate
        self.stride = stride
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        self.n_classes = n_classes
