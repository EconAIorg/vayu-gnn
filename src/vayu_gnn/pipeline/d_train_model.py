import os
import torch
import io
import pickle
from vayu_gnn.dbx import dbx_helper  # Ensure this is correctly imported in your project
from tsl.metrics.torch import MaskedMAE
from tsl.engines import Predictor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from tsl.nn.models import RNNEncGCNDecModel  # Replace with the correct import


class GNNPipeline:
    """
    Pipeline for setting up, training, testing, and saving a GNN model predictor.

    The pipeline loads a data module from Dropbox, extracts parameters,
    builds a model and predictor, trains and tests using PyTorch Lightning, 
    and then saves the trained predictor to Dropbox.

    Attributes
    ----------
    dm : object
        The data module loaded from Dropbox.
    city : str
        The city (or folder name) used to store outputs in Dropbox.
    """

    def __init__(self, city: str):
        """
        Initialize the pipeline by loading the data module from Dropbox.

        Parameters
        ----------
        city : str
            The city (or folder name) where the data module and predictor are stored.
        """
        self.city = city
        self.dm = dbx_helper.load_torch(dbx_helper.output_path, city, 'data_module.pt')
        if self.dm is None:
            raise ValueError("Failed to load data module from Dropbox.")

    def _setup_data_module(self, stage: str = 'fit'):
        """
        Set up the data module for the given stage.

        Parameters
        ----------
        stage : str, optional
            The stage to set up the data module for (e.g., 'fit' or 'predict').
            Default is 'fit'.
        """
        self.dm.setup(stage=stage)

    def _get_parameters(self) -> dict:
        """
        Extract model parameters from the data module.

        Returns
        -------
        dict
            A dictionary containing parameters such as input_size, output_size, 
            number of nodes, horizon, exogenous size, and input window size.
        """
        input_size = self.dm.torch_dataset.n_channels
        output_size = self.dm.torch_dataset.n_channels
        n_nodes = self.dm.torch_dataset.n_nodes
        horizon = self.dm.torch_dataset.horizon
        exog_size = self.dm.torch_dataset.n_covariates
        input_window_size = self.dm.window

        return {
            'input_size': input_size,
            'output_size': output_size,
            'n_nodes': n_nodes,
            'horizon': horizon,
            'exog_size': exog_size,
            'input_window_size': input_window_size,
        }

    def _build_model(self, params: dict):
        """
        Build and return the GNN model.

        Parameters
        ----------
        params : dict
            Dictionary of parameters extracted from the data module.

        Returns
        -------
        RNNEncGCNDecModel
            The initialized GNN model.
        """
        model = RNNEncGCNDecModel(
            output_size=params['output_size'],
            horizon=params['horizon'],
            input_size=params['input_size'],
            exog_size=params['exog_size'] * params['n_nodes'],
            hidden_size=64,
            rnn_layers=1,
            gcn_layers=1,
            rnn_dropout=0.3,
            gcn_dropout=0.3
        )
        return model

    def _setup_predictor(self, model):
        """
        Set up the Predictor for training and testing.

        Parameters
        ----------
        model : torch.nn.Module
            The initialized model.

        Returns
        -------
        Predictor
            The Predictor object with optimizer, loss function, and metrics configured.
        """
        loss_fn = MaskedMAE()
        metrics = {'mae': MaskedMAE()}

        predictor = Predictor(
            model=model,
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.001},
            loss_fn=loss_fn,
            metrics=metrics
        )
        return predictor

    def _setup_trainer(self):
        """
        Set up the PyTorch Lightning Trainer and checkpoint callback.

        Returns
        -------
        tuple
            A tuple (trainer, checkpoint_callback) where trainer is a PyTorch Lightning Trainer
            and checkpoint_callback is the ModelCheckpoint callback.
        """
        logger = TensorBoardLogger(save_dir="logs", name="tsl", version=2)
        checkpoint_callback = ModelCheckpoint(
            dirpath='logs',
            save_top_k=1,
            monitor='val_mae',
            mode='min',
        )

        trainer = Trainer(
            max_epochs=100,
            logger=logger,
            accelerator="gpu",
            limit_train_batches=100,  # End an epoch after 100 updates
            callbacks=[checkpoint_callback]
        )
        return trainer, checkpoint_callback

    def execute(self):
        """
        Execute the entire pipeline: setup data module, build model, train, test, and save the predictor.

        The method performs the following steps:
          1. Set up the data module for training.
          2. Extract relevant parameters.
          3. Build the model.
          4. Configure the predictor.
          5. Create the trainer.
          6. Train the model.
          7. Load the best model, freeze it, and test.
          8. Set up the data module for prediction.
          9. Save the predictor to Dropbox using dbx_helper.save_torch.
        """
        # 1. Setup data module for 'fit'
        self._setup_data_module(stage='fit')

        # 2. Extract parameters from the data module
        params = self._get_parameters()

        # 3. Build the model using extracted parameters
        model = self._build_model(params)

        # 4. Setup predictor with the model
        predictor = self._setup_predictor(model)

        # 5. Setup trainer and checkpoint callback
        trainer, checkpoint_callback = self._setup_trainer()

        # 6. Train the model
        trainer.fit(predictor, datamodule=self.dm)

        # 7. Load the best model, freeze it, and test
        predictor.load_model(checkpoint_callback.best_model_path)
        predictor.freeze()
        trainer.test(predictor, datamodule=self.dm)

        # 8. Set up data module for prediction stage
        self._setup_data_module(stage='predict')

        # 9. Save the predictor to Dropbox using dbx_helper.save_torch
        dbx_helper.save_torch(predictor, dbx_helper.output_path, self.city, 'predictor.pt')

