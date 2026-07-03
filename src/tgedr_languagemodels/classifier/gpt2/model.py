"""GPT model definition and supporting layers for language modeling."""

import logging
import torch
from torch import nn
import numpy as np

from tgedr_languagemodels.classifier.gpt2.configuration import ClassifierConfiguration
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from tgedr_languagemodels.layers.blocks import TransformerBlock
from tgedr_languagemodels.layers.normalization import LayerNormalization

logger = logging.getLogger(__name__)


class GPT2Classifier(PreTrainedModel):
    """GPT-2 based classifier for text classification tasks.

    This class extends the GPT2Model to add a classification head for predicting
    class labels. It inherits the embedding layers and transformer blocks from
    GPT2Model and adds a final linear layer for classification.

    Attributes
    ----------
    tok_emb : nn.Embedding
        Token embedding layer.
    pos_emb : nn.Embedding
        Positional embedding layer.
    drop_emb : nn.Dropout
        Dropout layer applied to embeddings.
    trf_blocks : nn.Sequential
        Sequential container of transformer blocks.
    final_norm : LayerNormalization
        Final layer normalization.
    out_head : nn.Linear
        Output linear layer for classification.
    """

    def __init__(self, cfg: ClassifierConfiguration) -> None:
        """Initialize GPT-2 model layers from the given configuration.

        Parameters
        ----------
        cfg : ClassifierConfiguration
            Model configuration containing vocabulary size, context length,
            embedding dimension, dropout rate, and number of transformer layers.
        """
        super().__init__(cfg)
        self.tok_emb = nn.Embedding(cfg.vocabulary_size, cfg.embeddings_dimension)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.embeddings_dimension)

        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.final_norm = LayerNormalization(cfg.embeddings_dimension)
        self.out_head = nn.Linear(cfg.embeddings_dimension, cfg.n_classes, bias=False)

    @property
    def device(self) -> torch.device:
        """Return the device to run the model on.

        Returns
        -------
        torch.device

            The device to run the model on.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute classifier logits for a batch of token ids."""
        _, seq_len = input_ids.shape
        input_ids = input_ids.to(self.device)
        tok_embeds = self.tok_emb(input_ids)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=self.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

    def calculate_batch_loss(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Calculate cross-entropy loss for a batch of inputs and targets.

        Parameters
        ----------
        input_batch : torch.Tensor
            The input tensor batch.
        target_batch : torch.Tensor
            The target tensor batch.

        Returns
        -------
        torch.Tensor
            The cross-entropy loss for the batch.
        """
        logger.debug(f"[calculate_batch_loss|in] ({input_batch}, {target_batch})")
        target_device = self.device if device is None else device
        input_batch = input_batch.to(target_device)
        target_batch = target_batch.to(target_device)
        logits = self._compute_logits(input_batch)[:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        logger.debug(f"[calculate_batch_loss|out] => {loss}")
        return loss

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """Forward pass compatible with Hugging Face Trainer.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token indices of shape (batch_size, sequence_length).
        attention_mask : torch.Tensor, optional
            Attention mask (not currently used by GPT2Classifier).
        labels : torch.Tensor, optional
            Target class labels for computing loss.
        **kwargs
            Additional arguments (ignored for compatibility).

        Returns
        -------
        SequenceClassifierOutput
            Hugging Face classifier output containing logits and optionally loss.
        """
        del attention_mask, kwargs
        logits = self._compute_logits(input_ids)
        loss = None

        if labels is not None:
            labels = labels.to(self.device)
            loss = torch.nn.functional.cross_entropy(logits[:, -1, :], labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def _assign(self, left, right) -> torch.nn.Parameter:
        """Validate shapes and return right as a new Parameter with the same shape as left.

        Parameters
        ----------
        left : torch.Tensor
            Reference tensor whose shape must match right.
        right : array-like
            Source data to wrap as a parameter.

        Returns
        -------
        torch.nn.Parameter
            Parameter wrapping the values of right.

        Raises
        ------
        ValueError
            If left and right have different shapes.
        """
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")  # noqa: EM102, TRY003
        return torch.nn.Parameter(torch.tensor(right))

    def _load_weights(self, params: dict) -> None:
        """Load pretrained weights into the model from a parameters dictionary.

        Parameters
        ----------
        params : dict
            Dictionary containing pretrained weights for token embeddings, positional
            embeddings, transformer blocks, and output layer.
        """
        # 1 Sets the model's positional and token embedding weights to those specified in params.

        self.pos_emb.weight = self._assign(self.pos_emb.weight, params["wpe"])
        self.tok_emb.weight = self._assign(self.tok_emb.weight, params["wte"])

        for b in range(len(params["blocks"])):  # 2 Iterates over each transformer block in the model
            # 3 The np.split function is used to divide the attention and bias weights
            # into three equal parts for the query, key, and value components.
            q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            self.trf_blocks[b].att.W_query.weight = self._assign(self.trf_blocks[b].att.W_query.weight, q_w.T)
            self.trf_blocks[b].att.W_key.weight = self._assign(self.trf_blocks[b].att.W_key.weight, k_w.T)
            self.trf_blocks[b].att.W_value.weight = self._assign(self.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            self.trf_blocks[b].att.W_query.bias = self._assign(self.trf_blocks[b].att.W_query.bias, q_b)
            self.trf_blocks[b].att.W_key.bias = self._assign(self.trf_blocks[b].att.W_key.bias, k_b)
            self.trf_blocks[b].att.W_value.bias = self._assign(self.trf_blocks[b].att.W_value.bias, v_b)

            self.trf_blocks[b].att.out_projection.weight = self._assign(
                self.trf_blocks[b].att.out_projection.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T
            )
            self.trf_blocks[b].att.out_projection.bias = self._assign(
                self.trf_blocks[b].att.out_projection.bias, params["blocks"][b]["attn"]["c_proj"]["b"]
            )

            self.trf_blocks[b].ff.layers[0].weight = self._assign(
                self.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T
            )
            self.trf_blocks[b].ff.layers[0].bias = self._assign(
                self.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
            )
            self.trf_blocks[b].ff.layers[2].weight = self._assign(
                self.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T
            )
            self.trf_blocks[b].ff.layers[2].bias = self._assign(
                self.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"]
            )

            self.trf_blocks[b].norm1.scale = self._assign(
                self.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
            )
            self.trf_blocks[b].norm1.shift = self._assign(
                self.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
            )
            self.trf_blocks[b].norm2.scale = self._assign(
                self.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
            )
            self.trf_blocks[b].norm2.shift = self._assign(
                self.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
            )

        self.final_norm.scale = self._assign(self.final_norm.scale, params["g"])
        self.final_norm.shift = self._assign(self.final_norm.shift, params["b"])

    def pretrain(self, weights: dict) -> None:
        """Prepare the model for fine-tuning by loading pretrained weights.

        Parameters
        ----------
        weights : dict
            Dictionary containing pretrained weights to load into the model.
        """
        self._load_weights(weights)
        for param in self.parameters():
            param.requires_grad = False
        # fine-tuning additional layers can noticeably improve the predictive performance of the model
        # We also configure the last transformer block and the final LayerNorm module,
        # which connects this block to the output layer, to be trainable
        for param in self.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in self.final_norm.parameters():
            param.requires_grad = True
        for param in self.out_head.parameters():
            param.requires_grad = True

    def infer(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model to obtain class logits.

        Parameters
        ----------
        in_idx : torch.Tensor
            Input tensor of shape (batch_size, sequence_length) containing token indices.

        Returns
        -------
        torch.Tensor
            Predicted class indices.
        """
        in_idx = in_idx.to(self.device)
        was_training = self.training
        if was_training:
            self.eval()
        with torch.no_grad():  # Models inference without gradient tracking
            logits = self._compute_logits(in_idx)[:, -1, :]
        if was_training:
            self.train()
        return torch.argmax(logits, dim=-1)
