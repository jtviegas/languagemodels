"""GPT model definition and supporting layers for language modeling."""

import logging
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader

from tgedr_languagemodels.configuration import BaseClassifierConfig
from tgedr_languagemodels.evaluations import CrossEntropyModelEvaluatorMixin
from tgedr_languagemodels.gpt2.model import GPT2Model
from tgedr_languagemodels.layers.blocks import TransformerBlock
from tgedr_languagemodels.layers.normalization import LayerNormalization

logger = logging.getLogger(__name__)


class GPT2Classifier(CrossEntropyModelEvaluatorMixin, GPT2Model):
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

    def __init__(self, cfg: BaseClassifierConfig) -> None:
        """Initialize GPT-2 model layers from the given configuration.

        Parameters
        ----------
        cfg : BaseModelConfig
            Model configuration containing vocabulary size, context length,
            embedding dimension, dropout rate, and number of transformer layers.
        """
        super().__init__(cfg)
        self._training = False
        self.tok_emb = nn.Embedding(cfg.vocabulary_size, cfg.embeddings_dimension)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.embeddings_dimension)

        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.final_norm = LayerNormalization(cfg.embeddings_dimension)
        self.out_head = nn.Linear(cfg.embeddings_dimension, cfg.n_classes, bias=False)

    def load_weights(self, params: dict) -> None:
        """Load pretrained weights into the model from a parameters dictionary.

        Parameters
        ----------
        params : dict
            Dictionary containing pretrained weights for token embeddings, positional
            embeddings, transformer blocks, and output layer.
        """
        # 1 Sets the model's positional and token embedding weights to those specified in params.

        self.pos_emb.weight = self.assign(self.pos_emb.weight, params["wpe"])
        self.tok_emb.weight = self.assign(self.tok_emb.weight, params["wte"])

        for b in range(len(params["blocks"])):  # 2 Iterates over each transformer block in the model
            # 3 The np.split function is used to divide the attention and bias weights
            # into three equal parts for the query, key, and value components.
            q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            self.trf_blocks[b].att.W_query.weight = self.assign(self.trf_blocks[b].att.W_query.weight, q_w.T)
            self.trf_blocks[b].att.W_key.weight = self.assign(self.trf_blocks[b].att.W_key.weight, k_w.T)
            self.trf_blocks[b].att.W_value.weight = self.assign(self.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            self.trf_blocks[b].att.W_query.bias = self.assign(self.trf_blocks[b].att.W_query.bias, q_b)
            self.trf_blocks[b].att.W_key.bias = self.assign(self.trf_blocks[b].att.W_key.bias, k_b)
            self.trf_blocks[b].att.W_value.bias = self.assign(self.trf_blocks[b].att.W_value.bias, v_b)

            self.trf_blocks[b].att.out_projection.weight = self.assign(
                self.trf_blocks[b].att.out_projection.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T
            )
            self.trf_blocks[b].att.out_projection.bias = self.assign(
                self.trf_blocks[b].att.out_projection.bias, params["blocks"][b]["attn"]["c_proj"]["b"]
            )

            self.trf_blocks[b].ff.layers[0].weight = self.assign(
                self.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T
            )
            self.trf_blocks[b].ff.layers[0].bias = self.assign(
                self.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
            )
            self.trf_blocks[b].ff.layers[2].weight = self.assign(
                self.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T
            )
            self.trf_blocks[b].ff.layers[2].bias = self.assign(
                self.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"]
            )

            self.trf_blocks[b].norm1.scale = self.assign(
                self.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
            )
            self.trf_blocks[b].norm1.shift = self.assign(
                self.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
            )
            self.trf_blocks[b].norm2.scale = self.assign(
                self.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
            )
            self.trf_blocks[b].norm2.shift = self.assign(
                self.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
            )

        self.final_norm.scale = self.assign(self.final_norm.scale, params["g"])
        self.final_norm.shift = self.assign(self.final_norm.shift, params["b"])

    def setup_for_tuning(self, weights: dict) -> None:
        """Prepare the model for fine-tuning by loading pretrained weights.

        Parameters
        ----------
        weights : dict
            Dictionary containing pretrained weights to load into the model.
        """
        self.load_weights(weights)
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

    def classify(self, in_idx: torch.Tensor) -> torch.Tensor:
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
        device = self.get_device()
        in_idx = in_idx.to(device)
        was_training = self._training
        if was_training:
            self.eval()
        with torch.no_grad():  # Models inference without gradient tracking
            logits = self(in_idx)[:, -1, :]
        if was_training:
            self.train()
        return torch.argmax(logits, dim=-1)

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Return an optimizer configured to update only the trainable parameters.

        Returns
        -------
        torch.optim.Optimizer
            An optimizer instance for updating model parameters during training.
        """
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=3e-5, weight_decay=0.1)

    def get_device(self) -> torch.device:
        """Return the device to run the model on.

        Returns
        -------
        torch.device

            The device to run the model on.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_classifier(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        eval_batches: int | None,
    ) -> tuple:
        """Train the classifier on the provided training data.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for training data.
        val_loader : DataLoader
            DataLoader for validation data.
        num_epochs : int
            Number of epochs to train for.
        eval_batches : int | None
            Number of batches to use for evaluation. If None, evaluate all batches.

        Returns
        -------
        tuple
            A tuple containing (train_accs, val_accs, examples_seen).
        """
        # 1 Initialize lists to track losses and examples seen
        train_accs, val_accs = [], []
        examples_seen = 0

        device = self.get_device()
        self.to(device)
        optimizer = self.get_optimizer()

        for _ in range(num_epochs):  # 2 Main training loop
            if not self._training:
                self.train()  # 3 Sets model to training mode
                self._training = True

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()  # 4 Resets loss gradients from the previous batch iteration
                loss = self.calculate_batch_loss(input_batch, target_batch, device)
                loss.backward()  # 5 Calculates loss gradients
                optimizer.step()  # 6 Updates model weights using loss gradients
                examples_seen += input_batch.shape[0]  # 7 New: tracks examples instead of tokens

            # 9 Calculates accuracy after each epoch
            if self._training:
                self.eval()  # 10 Sets model to evaluation mode
            train_accuracy = self.calculate_loader_accuracy(train_loader, device, num_batches=eval_batches)
            val_accuracy = self.calculate_loader_accuracy(val_loader, device, num_batches=eval_batches)

            logger.info(
                "Training accuracy: %.2f%% | Validation accuracy: %.2f%%",
                train_accuracy * 100,
                val_accuracy * 100,
            )
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)

        return train_accs, val_accs, examples_seen
