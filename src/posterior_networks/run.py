import logging
import torch

import sys
sys.path.append("../")
from src.dataset_manager.ClassificationDataset import ClassificationDataset
from src.posterior_networks.PosteriorNetwork import PosteriorNetwork
from src.posterior_networks.train import train, train_sequential
from src.posterior_networks.test import test

def run(
        # Dataset parameters
        seed_dataset,  # Seed to shuffle dataset. int
        directory_dataset,  # Path to dataset. string
        dataset_name,  # Dataset name. string
        ood_dataset_names,  # OOD dataset names.  list of strings
        unscaled_ood,  # If true consider also unscaled versions of ood datasets. boolean
        split,  # Split for train/val/test sets. list of floats
        transform_min,  # Minimum value for rescaling input data. float
        transform_max,  # Maximum value for rescaling input data. float

        # Architecture parameters
        seed_model,  # Seed to init model. int
        directory_model,  # Path to save model. string
        architecture,  # Encoder architecture name. string
        input_dims,  # Input dimension. List of ints
        output_dim,  # Output dimension. int
        hidden_dims,  # Hidden dimensions. list of ints
        kernel_dim,  # Input dimension. int
        latent_dim,  # Latent dimension. int
        no_density,  # Use density estimation or not. boolean
        density_type,  # Density type. string
        n_density,  # Number of density components. int
        k_lipschitz,  # Lipschitz constant. float or None (if no lipschitz)
        budget_function,  # Budget function name applied on class count. name

        # Training parameters
        directory_results,  # Path to save resutls. string
        max_epochs,  # Maximum number of epochs for training
        patience,  # Patience for early stopping. int
        frequency,  # Frequency for early stopping test. int
        batch_size,  # Batch size. int
        lr,  # Learning rate. float
        loss,  # Loss name. string
        training_mode,  # 'joint' or 'sequential' training. string
        regr):  # Regularization factor in Bayesian loss. float

    logging.info('Received the following configuration:')
    logging.info(f'DATASET | '
                 f'seed_dataset {seed_dataset} - '
                 f'dataset_name {dataset_name} - '
                 f'ood_dataset_names {ood_dataset_names} - '
                 f'split {split} - '
                 f'transform_min {transform_min} - '
                 f'transform_max {transform_max}')
    logging.info(f'ARCHITECTURE | '
                 f' seed_model {seed_model} - '
                 f' architecture {architecture} - '
                 f' input_dims {input_dims} - '
                 f' output_dim {output_dim} - '
                 f' hidden_dims {hidden_dims} - '
                 f' kernel_dim {kernel_dim} - '
                 f' latent_dim {latent_dim} - '
                 f' no_density {no_density} - '
                 f' density_type {density_type} - '
                 f' n_density {n_density} - '
                 f' k_lipschitz {k_lipschitz} - '
                 f' budget_function {budget_function}')
    logging.info(f'TRAINING | '
                 f' max_epochs {max_epochs} - '
                 f' patience {patience} - '
                 f' frequency {frequency} - '
                 f' batch_size {batch_size} - '
                 f' lr {lr} - '
                 f' loss {loss} - '
                 f' training_mode {training_mode} - '
                 f' regr {regr}')

    ##################
    ## Load dataset ##
    ##################
    dataset = ClassificationDataset(f'{directory_dataset}/{dataset_name}.csv',
                                    input_dims=input_dims, output_dim=output_dim,
                                    transform_min=transform_min, transform_max=transform_max,
                                    seed=seed_dataset)
    train_loader, val_loader, test_loader, N = dataset.split(batch_size=batch_size, split=split, num_workers=4)

    #################
    ## Train model ##
    #################
    model = PosteriorNetwork(N=N,
                             input_dims=input_dims,
                             output_dim=output_dim,
                             hidden_dims=hidden_dims,
                             kernel_dim=kernel_dim,
                             latent_dim=latent_dim,
                             architecture=architecture,
                             k_lipschitz=k_lipschitz,
                             no_density=no_density,
                             density_type=density_type,
                             n_density=n_density,
                             budget_function=budget_function,
                             batch_size=batch_size,
                             lr=lr,
                             loss=loss,
                             regr=regr,
                             seed=seed_model)
    full_config_dict = {'seed_dataset': seed_dataset,
                        'dataset_name': dataset_name,
                        'split': split,
                        'transform_min': transform_min,
                        'transform_max': transform_max,
                        'seed_model': seed_model,
                        'architecture': architecture,
                        # 'N': N,
                        'input_dims': input_dims,
                        'output_dim': output_dim,
                        'hidden_dims': hidden_dims,
                        'kernel_dim': kernel_dim,
                        'latent_dim': latent_dim,
                        'no_density': no_density,
                        'density_type': density_type,
                        'n_density': n_density,
                        'k_lipschitz': k_lipschitz,
                        'budget_function': budget_function,
                        'max_epochs': max_epochs,
                        'patience': patience,
                        'frequency': frequency,
                        'batch_size': batch_size,
                        'lr': lr,
                        'loss': loss,
                        'training_mode': training_mode,
                        'regr': regr}
    full_config_name = ''
    for k, v in full_config_dict.items():
        full_config_name += str(v) + '-'
    full_config_name = full_config_name[:-1]
    model_path = f'{directory_model}/model-dpn-{full_config_name}'
    if training_mode == 'joint':
        train_losses, val_losses, train_accuracies, val_accuracies = train(model,
                                                                      train_loader,
                                                                      val_loader,
                                                                      max_epochs=max_epochs,
                                                                      frequency=frequency,
                                                                      patience=patience,
                                                                      model_path=model_path,
                                                                      full_config_dict=full_config_dict)
    elif training_mode == 'sequential':
        assert not no_density
        train_losses, val_losses, train_accuracies, val_accuracies = train_sequential(model,
                                                                                       train_loader,
                                                                                       val_loader,
                                                                                       max_epochs=max_epochs,
                                                                                       frequency=frequency,
                                                                                       patience=patience,
                                                                                       model_path=model_path,
                                                                                       full_config_dict=full_config_dict)
    else:
        raise NotImplementedError

    ################
    ## Test model ##
    ################
    ood_dataset_loaders = {}
    for ood_dataset_name in ood_dataset_names:
        dataset = ClassificationDataset(f'{directory_dataset}/{ood_dataset_name}.csv',
                                        input_dims=input_dims, output_dim=output_dim,
                                        transform_min=transform_min, transform_max=transform_max,
                                        seed=None)
        ood_dataset_loaders[ood_dataset_name] = torch.utils.data.DataLoader(dataset, batch_size=2 * batch_size, num_workers=4, pin_memory=True)
        if unscaled_ood:
            dataset = ClassificationDataset(f'{directory_dataset}/{ood_dataset_name}.csv',
                                            input_dims=input_dims, output_dim=output_dim,
                                            seed=None)
            ood_dataset_loaders[ood_dataset_name + '_unscaled'] = torch.utils.data.DataLoader(dataset, batch_size=2 * batch_size, num_workers=4, pin_memory=True)
    result_path = f'{directory_results}/results-dpn-{full_config_name}'
    model.load_state_dict(torch.load(f'{model_path}')['model_state_dict'])
    metrics = test(model, test_loader, ood_dataset_loaders, result_path)

    results = {
        'model_path': model_path,
        'result_path': result_path,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }

    return {**results, **metrics}
