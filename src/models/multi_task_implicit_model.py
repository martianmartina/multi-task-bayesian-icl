import torch
import pytorch_lightning as pl
from src.models.implicit_model_new import ImplicitInContextLearner
from src.data.multi_task_logistic_regression import ROLE_PREDICTION
class MultiTaskImplicitInContextLearner(ImplicitInContextLearner):
    """
    A multi-task version of the implicit in-context learning model.
    """
    def __init__(self, *args, **kwargs):
        if 'learning_rate' in kwargs:
            kwargs['learning_rate'] = float(kwargs['learning_rate'])
        super().__init__(*args, **kwargs)
    
    def _unpack_batch(self, batch: tuple):
        """
        Minimal helper so we can handle (x,y) or (x,y,seg_ids,task_ids) seamlessly.
        Returns (x, y, seg_ids, task_ids) where seg_ids/task_ids may be None.
        """
        x, y, seg_ids, task_ids, role_ids = None, None, None, None, None
        if len(batch) == 3:
            x, y, role_ids = batch
        elif len(batch) == 5:
            x, y, seg_ids, task_ids, role_ids = batch
        else:
            raise ValueError(f"Unexpected batch format of length {len(batch)}")
        return x, y, seg_ids, task_ids, role_ids

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y, seg_ids, task_ids, role_ids = self._unpack_batch(batch)

        model_input, targets = self._prepare_input_sequence(x, y, seg_ids, task_ids)

        output1, output2, _ = self(model_input)

        loss_mask = (role_ids == ROLE_PREDICTION)
        loss_mask = loss_mask.unsqueeze(-1).expand_as(targets)
        
        output1_masked = output1[loss_mask]
        targets_masked = targets[loss_mask]

        if output1_masked.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if self.hparams.task_type == 'regression':
            output2_masked = output2[loss_mask]
            loss = self.gaussian_nll_loss(output1_masked, output2_masked, targets_masked)
            self.log('train_nll_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        else:  # classification
            loss = self.binary_cross_entropy_loss(output1_masked, targets_masked)
            self.log('train_bce_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        x, y, seg_ids, task_ids, role_ids = self._unpack_batch(batch)

        model_input, targets = self._prepare_input_sequence(x, y, seg_ids, task_ids)
        output1, output2, _ = self(model_input)

        loss_mask = (role_ids == ROLE_PREDICTION)
        loss_mask = loss_mask.unsqueeze(-1).expand_as(targets)

        output1_masked = output1[loss_mask]
        targets_masked = targets[loss_mask]

        if output1_masked.numel() > 0:
            if self.hparams.task_type == 'regression':
                output2_masked = output2[loss_mask]
                loss = self.gaussian_nll_loss(output1_masked, output2_masked, targets_masked)
            else:  # classification
                loss = self.binary_cross_entropy_loss(output1_masked, targets_masked)

            self.log('val_loss', loss, on_epoch=True, prog_bar=True)
