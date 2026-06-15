# ridge.py
"""
Note this script should implement a generalized ridge regression model which 
can provide the Bayesian optimal solution even when the mean of prior and noise are non-zero.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Ridge(nn.Module):
    def __init__(self, mu_w: float = 0.0, std_w: float = 1.0, 
                 mu_eps: float = 0.0, std_eps: float = 0.5, 
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.mu_w = mu_w
        self.std_w = std_w
        self.mu_eps = mu_eps
        self.std_eps = std_eps
        self.dtype = dtype
        if std_w == 0.0:
            self.lam = None  # forces w = mu_w
        elif std_eps == 0.0:
            self.lam = 1e-6  # minimal ridge penalty to prevent singular matrix
        else:
            self.lam = float(std_eps ** 2 / std_w ** 2)

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, return_var: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        X_train : (B, N, D)
        y_train : (B, N) or (B, N, 1)
        X_test  : (B, T, D) or (B, D) for a single test point

        Returns
        -------
        preds   : (B, T) predictions for X_test
        pred_var: (B, T) predictive variances for X_test
        """
        device = X_train.device
        dtype = self.dtype

        # Ensure desired dtype without extra copies when possible
        X = X_train.to(dtype)                      # (B, N, D)
        Y = y_train.to(dtype)
        if Y.dim() == 2:                           # (B, N) -> (B, N, 1)
            Y = Y.unsqueeze(-1)

        # Normalize X_test shape to (B, T, D)
        if X_test.dim() == 2:                      # (B, D) -> (B, 1, D)
            Xq = X_test.to(dtype).unsqueeze(1)
        else:
            Xq = X_test.to(dtype)                  # (B, T, D)

        B, _, D = X.shape

        eye = torch.eye(D, device=device, dtype=dtype).unsqueeze(0)  # (1, D, D)
        mu_eps_tensor = torch.tensor(self.mu_eps, dtype=dtype, device=device)
        mu_w_tensor = torch.full((1, D, 1), self.mu_w, dtype=dtype, device=device)  # (1, D, 1)

        if self.lam is not None:
            # Batched ridge closed-form: w = (XᵀX + λI)⁻¹ [Xᵀ(y - μ_ε) + λ μ_w]
            XT   = X.transpose(1, 2)                              # (B, D, N)
            XT_X = torch.bmm(XT, X)                               # (B, D, D)
            ridge_mat = XT_X + self.lam * eye                     # (B, D, D)
            XT_y = torch.bmm(XT, (Y - mu_eps_tensor))             # (B, D, 1)
            rhs = XT_y + (self.lam * mu_w_tensor).expand_as(XT_y) # (B, D, 1)
            w = torch.linalg.solve(ridge_mat, rhs)                # (B, D, 1)
        else:
            w = mu_w_tensor.expand(B, D, 1)                       # (B, D, 1)

        # Predict on X_test
        preds = torch.bmm(Xq, w).squeeze(-1) + self.mu_eps                      # (B, T)
        
        if not return_var:
            return preds

        # 2. Calculate Predictive Standard Deviation
        if self.lam is not None:
            # We need to compute: x_q^T * (X^T X + lambda I)^-1 * x_q
            # We solve: (X^T X + lambda I) v = x_q^T
            # Then: x_q^T * v = result
            
            # Xq is (B, T, D), we need it as (B, D, T) for the solve
            Xq_T = Xq.transpose(1, 2) 
            
            # v: (B, D, T)
            v = torch.linalg.solve(ridge_mat, Xq_T)
            
            # Compute quadratic form (element-wise dot product across D)
            # quad_form: (B, T)
            quad_form = torch.sum(Xq_T * v, dim=1)
            
            # Predictive Variance: sigma_eps^2 * (1 + quad_form)
            # Use max(0) or clamp to prevent tiny numerical negatives
            pred_var = (self.std_eps ** 2) * (1.0 + quad_form)
        else:
            # If std_w is 0, w is fixed to mu_w, weight uncertainty is 0
            pred_var = torch.full_as(preds, self.std_eps ** 2)

        return preds, pred_var
