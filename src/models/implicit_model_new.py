import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F

from src.data.multi_task_logistic_regression import Y_PADDING
from src.models.gpt2 import GPTConfig, Block as GPTBlock, LayerNorm as GPTLayerNorm

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal absolute positional encoding implementation.
    Expects input shape (T, B, d_model)
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ImplicitInContextLearner(pl.LightningModule):
    """
    A PyTorch Lightning module for the implicit in-context learning model.
    """
    def __init__(self,
                 model_type: str = 'transformer',
                 x_dim: int = 1,
                 y_dim: int = 1,
                 d_model: int = 64,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-4,
                 nhead: int = 4,
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 128,
                 gru_hidden_dim: int = 64,
                 gru_num_layers: int = 2,
                 consistency_beta: float = 0.0, # Weight for the regularizer
                 task_type: str = 'regression',
                 identity_dim: int = 0,
                 rope: bool = True,
                 **kwargs): # 'regression' or 'classification'
        super().__init__()
        self.save_hyperparameters()
        assert model_type in ['transformer', 'gru']
        assert task_type in ['regression', 'classification']
        # Input projection
        self.input_proj = nn.Linear(x_dim + y_dim + identity_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        if not rope and model_type == 'transformer':
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, dropout)
        else:
            self.pos_encoder = None
        print(f"RoPE: {rope}, Sinusoidal Positional Encoding: {self.pos_encoder is not None}")
        # Sequence processor
        if self.hparams.model_type == 'transformer':
            self.transformer_config = GPTConfig(
                n_layer=num_encoder_layers,
                n_head=nhead,
                n_embd=d_model,
                dropout=dropout,
                bias=True,
            )
            self.transformer_blocks = nn.ModuleList([
                GPTBlock(self.transformer_config, rope=rope) for _ in range(num_encoder_layers)
            ])
            self.transformer_norm = GPTLayerNorm(d_model, bias=True)
        elif self.hparams.model_type == 'gru':
            self.sequence_processor = nn.GRU(
                input_size=d_model, hidden_size=gru_hidden_dim,
                num_layers=gru_num_layers, batch_first=True,
                dropout=dropout if gru_num_layers > 1 else 0
            )
        # Output projection
        output_dim = gru_hidden_dim if model_type == 'gru' else d_model
        self.output_dim = output_dim
        if task_type == 'regression':
            # For regression: predict mean and log-variance (2 * y_dim)
            self.output_proj = nn.Linear(output_dim, y_dim * 2)
        else:
            # For classification: predict logits (y_dim for binary classification)
            self.output_proj = nn.Linear(output_dim, y_dim)

    def _prepare_input_sequence(self, x: torch.Tensor, y: torch.Tensor, seg_ids: torch.Tensor = None, task_ids: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, _ = x.shape
        y_0 = torch.full((batch_size, 1, self.hparams.y_dim), Y_PADDING, dtype=torch.float32, device=self.device)
        y_shifted = torch.cat([y_0, y[:, :-1, :]], dim=1)
        model_input = torch.cat([x, y_shifted], dim=-1)
        if seg_ids is not None:
            # (batch_size, seq_len) -> (batch_size, seq_len, 1)
            seg_ids = seg_ids.unsqueeze(-1)
            model_input = torch.cat([model_input, seg_ids], dim=-1)
        if task_ids is not None:
            # (batch_size, seq_len) -> (batch_size, seq_len, 1)
            task_ids = task_ids.unsqueeze(-1)
            model_input = torch.cat([model_input, task_ids], dim=-1)
        targets = y
        return model_input, targets

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def _encode_sequence(self, src: torch.Tensor, hidden_state: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the shared encoder stack and returns the hidden representations before the output head.

        This helper allows subclasses to reuse the encoder while attaching custom decoding heads.
        """
        src = self.input_proj(src)
        if self.hparams.model_type == 'transformer':
            if len(self.transformer_blocks) > 0:
                encoder_param_dtype = next(self.transformer_blocks[0].parameters()).dtype
            else:
                encoder_param_dtype = self.transformer_norm.weight.dtype
            if src.dtype != encoder_param_dtype:
                src = src.to(dtype=encoder_param_dtype)
            src = src * math.sqrt(self.hparams.d_model)
            if self.pos_encoder is not None:
                # src shape: (B, T, d_model) -> (T, B, d_model)
                src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
            x = self.input_dropout(src)
            for block in self.transformer_blocks:
                x = block(x)
            output = self.transformer_norm(x)
            hidden_state = None # Transformer is stateless in this forward pass
        elif self.hparams.model_type == 'gru':
            output, hidden_state = self.sequence_processor(src, hidden_state)

        return output, hidden_state

    def forward(self, src: torch.Tensor, hidden_state: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, hidden_state = self._encode_sequence(src, hidden_state)
        predictions = self.output_proj(output)
        
        if self.hparams.task_type == 'regression':
            mu, logvar = predictions.chunk(2, dim=-1)
            return mu, logvar, hidden_state
        else:  # classification
            # For classification, return logits and None for logvar
            logits = predictions
            return logits, None, hidden_state

    def gaussian_nll_loss(self, mu, logvar, y_true):
        """Calculates the negative log-likelihood for a Gaussian distribution."""
        # Clamp logvar for numerical stability
        # Note: we do not include the constant term in the Gaussian NLL here 
        # since we only use this for optimizing the model and this won't be reported and compared to other methods
        # for era5 experiments, we include the constant term in the KL divergence for a precise Gaussian NLL loss
        logvar = torch.clamp(logvar, min=-10, max=10)
        term1 = 0.5 * logvar
        term2 = 0.5 * ((y_true - mu) ** 2) / torch.exp(logvar)
        return (term1 + term2).mean()

    def kl_divergence_gaussians(self, mu1, logvar1, mu2, logvar2):
        """Calculates KL(N(mu1, var1) || N(mu2, var2))."""
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        term1 = (var1 + (mu1 - mu2)**2) / (2 * var2)
        term2 = 0.5 * (logvar2 - logvar1 - 1)
        return (term1 + term2).mean()

    def binary_cross_entropy_loss(self, logits, y_true):
        """Calculates binary cross-entropy loss for classification."""
        return F.binary_cross_entropy_with_logits(logits, y_true)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        batch_size, seq_len, _ = x.shape
        
        # --- Pass 1: Standard Prediction ---
        model_input, targets = self._prepare_input_sequence(x, y)
        output1, output2, _ = self(model_input)
        
        if self.hparams.task_type == 'regression':
            # For regression: output1=mu, output2=logvar
            primary_loss = self.gaussian_nll_loss(output1, output2, targets)
            self.log('train_nll_loss', primary_loss, on_step=True, on_epoch=True, prog_bar=True)
        else:  # classification
            # For classification: output1=logits, output2=None
            primary_loss = self.binary_cross_entropy_loss(output1, targets)
            self.log('train_bce_loss', primary_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        total_loss = primary_loss

        self.log('train_total_loss', total_loss)
        return total_loss

    def validation_step(self, batch: tuple, batch_idx: int):
        x, y = batch
        model_input, targets = self._prepare_input_sequence(x, y)
        print(model_input.type(), targets.type())
        output1, output2, _ = self(model_input)
        
        if self.hparams.task_type == 'regression':
            loss = self.gaussian_nll_loss(output1, output2, targets)
        else:  # classification
            loss = self.binary_cross_entropy_loss(output1, targets)
            
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
            Added linear warmup and cosine annealing scheduler.
        """
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.get('warmup_ratio', 0.03) * total_steps)
        warmup_steps = max(1, warmup_steps)
        

        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps - warmup_steps
        )
        sch = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}


    def predict_autoregressive(self, x_context: torch.Tensor, y_context: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
        self.eval()
        x_context, y_context, x_query = x_context.to(self.device), y_context.to(self.device), x_query.to(self.device)
        y_query_predictions = []

        with torch.no_grad():
            if self.hparams.model_type == 'gru':
                # --- Efficient GRU Path ---
                hidden_state = None
                if len(x_context) > 0:
                    context_input, _ = self._prepare_input_sequence(x_context.unsqueeze(0), y_context.unsqueeze(0))
                    _, _, hidden_state = self(context_input)
                
                last_y = y_context[-1:] if len(y_context) > 0 else torch.zeros(1, self.hparams.y_dim, device=self.device)

                for i in range(len(x_query)):
                    current_x = x_query[i:i+1]
                    step_input = torch.cat([current_x, last_y], dim=-1).unsqueeze(0)
                    output1, _, hidden_state = self(step_input, hidden_state)
                    
                    if self.hparams.task_type == 'regression':
                        next_y_pred = output1.squeeze(0)  # Use mean prediction
                    else:  # classification
                        # Convert logits to probabilities and sample
                        probs = torch.sigmoid(output1.squeeze(0))
                        next_y_pred = torch.bernoulli(probs)
                    
                    y_query_predictions.append(next_y_pred)
                    last_y = next_y_pred

            elif self.hparams.model_type == 'transformer':
                # --- Slower Transformer Path ---
                for i in range(len(x_query)):
                    x_seq = torch.cat([x_context, x_query[:i+1]], dim=0)
                    y_known = torch.cat([y_context] + y_query_predictions, dim=0) if y_query_predictions else y_context
                    
                    y_placeholder = torch.full(
                        (1, self.hparams.y_dim),
                        float(Y_PADDING),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    y_seq = torch.cat([y_known, y_placeholder], dim=0)

                    # x_seq and y_seq now have the same length
                    model_input, _ = self._prepare_input_sequence(x_seq.unsqueeze(0), y_seq.unsqueeze(0))
                    
                    output1, _, _ = self(model_input)
                    
                    if self.hparams.task_type == 'regression':
                        next_y_pred = output1[:, -1:, :].squeeze(0)  # Use mean prediction
                    else:  # classification
                        # Convert logits to probabilities and sample
                        logits = output1[:, -1:, :].squeeze(0)
                        probs = torch.sigmoid(logits)
                        next_y_pred = torch.bernoulli(probs)
                    
                    y_query_predictions.append(next_y_pred)

        return torch.cat(y_query_predictions, dim=0)

    def predict_independent(self, x_context: torch.Tensor, y_context: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
        self.eval()
        x_context, y_context, x_query = x_context.to(self.device), y_context.to(self.device), x_query.to(self.device)
        
        with torch.no_grad():
            batch_size = len(x_query)
            context_len = len(x_context)
            
            # Repeat context for each query point: [batch_size, context_len, x_dim]
            x_context_batch = x_context.unsqueeze(0).expand(batch_size, -1, -1)
            y_context_batch = y_context.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Add query points: [batch_size, context_len + 1, x_dim]
            x_query_expanded = x_query.unsqueeze(1)  # [batch_size, 1, x_dim]
            x_seq_batch = torch.cat([x_context_batch, x_query_expanded], dim=1)
            
            # Add placeholder y values for query points: [batch_size, context_len + 1, y_dim]
            # Use Y_PADDING to match training-time special-token padding semantics.
            y_placeholder_batch = torch.full(
                (batch_size, 1, self.hparams.y_dim),
                float(Y_PADDING),
                dtype=torch.float32,
                device=self.device,
            )
            y_seq_batch = torch.cat([y_context_batch, y_placeholder_batch], dim=1)
            
            # Prepare input and predict in one forward pass
            model_input, _ = self._prepare_input_sequence(x_seq_batch, y_seq_batch)
            output1, _, _ = self(model_input)
            
            # Extract predictions for the last position (query prediction)
            if self.hparams.task_type == 'regression':
                y_query_predictions = output1[:, -1, :]  # [batch_size, y_dim]
            else:  # classification: we return logits instead of y
                # Convert logits to probabilities and sample
                logits = output1[:, -1, :]  # [batch_size, y_dim]
                y_query_predictions = logits

        return y_query_predictions
