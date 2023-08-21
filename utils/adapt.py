import torch
import numpy as np


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt_maml(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 4] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy

def fast_adapt_mlti(batch, lam, mix_flag, learner, loss, adaptation_steps, shots, ways, device):
    data, label1, label2 = batch
    data, label1, label2 = data.to(device), label1.to(device), label2.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 4] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_label1, adaptation_label2 = data[adaptation_indices], label1[adaptation_indices], label2[adaptation_indices]
    evaluation_data, evaluation_label1, evaluation_label2 = data[evaluation_indices], label1[evaluation_indices], label2[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        if mix_flag == 'cross':
             adaptation_error = loss(learner(adaptation_data), adaptation_label1)
        elif mix_flag == 'inner':
            adaptation_error = lam*loss(learner(adaptation_data), adaptation_label1) + (1-lam)*loss(learner(adaptation_data), adaptation_label2)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    if mix_flag == 'cross':
        evaluation_error = loss(predictions, evaluation_label1)
        evaluation_accuracy = accuracy(predictions, evaluation_label1)
    elif mix_flag == 'inner':
        evaluation_error = lam*loss(predictions, evaluation_label1) + (1-lam)*loss(predictions, evaluation_label2)
        evaluation_accuracy = lam*accuracy(predictions, evaluation_label1) + (1-lam)*accuracy(predictions, evaluation_label2)

    return evaluation_error, evaluation_accuracy
