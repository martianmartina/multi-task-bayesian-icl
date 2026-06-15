import yaml
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data.multi_task_linear import MultiTaskLinearDataModule
from data.multi_task_logistic_regression import MultiTaskLogisticDataModule
from models.multi_task_implicit_model import MultiTaskImplicitInContextLearner

class GradNormLogger(pl.Callback):
    def __init__(self, norm_type: float = 2.0, log_name: str = "grad_norm/pre_clip"):
        self.norm_type = norm_type
        self.log_name = log_name

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        norms = []
        for p in pl_module.parameters():
            if p.grad is not None:
                norms.append(p.grad.detach().data.norm(self.norm_type))
        if not norms:
            return
        total_norm = torch.norm(torch.stack(norms), self.norm_type)
        pl_module.log(self.log_name, total_norm, on_step=True, logger=True, prog_bar=False)


def main():
    parser = argparse.ArgumentParser(description="Run a multi-task training experiment.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--lr', type=float, default=None, help='Overwrite the learning rate in the config file.')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, 
                       help='Path to checkpoint file to resume training from.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    resume_checkpoint = args.resume_from_checkpoint or config.get('resume_from_checkpoint', None)

    # Instantiate the data module
    print("Instantiating DataModule...")
    data_config = config['data']

    data_config.pop('_target_', None)
    name = data_config.pop('name', None)
    if name == "linear":
        data_module = MultiTaskLinearDataModule(**data_config)
    elif name == "logistic":
        data_module = MultiTaskLogisticDataModule(**data_config)
    else:
        raise ValueError(f"Invalid data module name: {name}")
    
    print("Instantiating Model...")
    model_config = config['model']
    if args.lr is not None:
        model_config['learning_rate'] = args.lr
    
    model_config.pop('_target_', None)
    
    # Load model from checkpoint if resuming, otherwise create new model
    if resume_checkpoint:
        print(f"Loading model from checkpoint: {resume_checkpoint}")
        try:
            model = MultiTaskImplicitInContextLearner.load_from_checkpoint(resume_checkpoint)
            print("Model loaded successfully from checkpoint!")
        except FileNotFoundError:
            print(f"ERROR: Checkpoint file not found at '{resume_checkpoint}'")
            print("Creating new model instead...")
            model = MultiTaskImplicitInContextLearner(**model_config)
        except Exception as e:
            print(f"ERROR loading checkpoint: {e}")
            print("Creating new model instead...")
            model = MultiTaskImplicitInContextLearner(**model_config)
    else:
        model = MultiTaskImplicitInContextLearner(**model_config)

    print("Instantiating Logger...")
    logger_config = config['logger']
    logger_config.pop('_target_', None)
    logger_config['name'] = config['experiment_name']
    logger_config['project'] = config['project_name']
    logger = WandbLogger(**logger_config)
    logger.log_hyperparams(config)

    print("Instantiating Checkpoint Callback...")
    checkpoint_config = config['checkpoint']
    checkpoint_config.pop('_target_', None)
    checkpoint_config['dirpath'] = checkpoint_config['dirpath'].replace('${experiment_name}', config['experiment_name'])
    checkpoint_callback = ModelCheckpoint(**checkpoint_config)

    print("Instantiating Early Stopping Callback...")
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=config.get('early_stopping_patience', 100),
        verbose=True,
        mode='min'
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        accelerator=config['accelerator'],
        devices=config['devices'],
        max_epochs=config['max_epochs'],
        precision='bf16-mixed' if config['precision'] == 16 else config['precision'],
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, GradNormLogger()],
        log_every_n_steps=50,
        strategy='ddp',
        gradient_clip_val=1.0, 
        gradient_clip_algorithm="norm"
    )

    # Start training
    print(f"--- Starting training for experiment: {config['experiment_name']} ---")
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_checkpoint)
    else:
        print("Starting training from scratch")
        trainer.fit(model, datamodule=data_module)
    print("--- Training finished ---")

if __name__ == '__main__':
    main()
