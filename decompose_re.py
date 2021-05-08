import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import distance


# Algorithms 1-3 based on paper
def alg1_FC(weight, ind, threshold, model_type):
    '''
    weight - 2D matrix (n_{i+1}, n_i), torch.Tensor
    ind - chosen indices to remain, list
    threshold - cosine similarity threshold
    '''
    # Y_i == weight_chosen
    # Z_i == scaling_mat

    Y_i = weight[ind, :]
    Z_i = torch.zeros(weight.shape[0], Y_i.shape[0], dtype=torch.float32)
    for i in range(weight.shape[0]):
        if i in ind: # selected neuron
            p = ind.index(i)
            Z_i[i, p] = 1
        else: # pruned neuron
            if model_type == 'prune':
                continue

            w_n = weight[i, :]
            w_star_n, p_star, sim, scale = alg2(w_n, Y_i)
            if sim >= threshold:
                Z_i[i, p_star] = scale

    return Y_i, Z_i

def alg2(w_n, Y_i):
    '''
    w_n - weight vector (n_i), torch.Tensor
    Y_i - selected neurons (p_{i+1}, n_i), torch.Tensor
    '''

    cosine_sim = []
    cos = nn.CosineSimilarity(dim=0, eps=1e-8)
    for i in range(Y_i.shape[0]):
        w = Y_i[i,:]
        cosine_sim.append(cos(w_n, w))
    sim = max(cosine_sim)
    max_ind = cosine_sim.index(sim)
    w_star_n = Y_i[max_ind, :]
    w_n_norm = torch.norm(w_n, p='fro')
    w_star_n_norm = torch.norm(w_star_n, p='fro')
    scale = w_n_norm / w_star_n_norm
    return w_star_n, max_ind, sim, scale

def alg1_conv(weight, ind, threshold, bn_weight, bn_bias, bn_mean, bn_var, lam, model_type):
    '''
    weight - 4D tensor (N_{i+1}, N_i, K, K), torch.Tensor
    ind - chosen indices to remain, list
    threshold - cosine similarity threshold
    bn_weight, bn_bias - parameters of batch norm layer right after the conv layer
    bn_mean, bn_var - running_mean, running_var of BN (for inference)
    lam - how much to consider cosine sim over bias, float value between 0 and 1
    '''
    # Y_i == weight_chosen
    # Z_i == scaling_mat

    # Reshaping the conv filters into 1D tensors
    weight = weight.reshape(weight.shape[0], -1)
    Y_i = weight[ind, :]
    Z_i = torch.zeros(weight.shape[0], Y_i.shape[0], dtype=torch.float32)

    for i in range(weight.shape[0]):
        if i in ind: # selected neuron
            p = ind.index(i)
            Z_i[i, p] = 1
        else: # pruned neuron
            if model_type == 'prune':
                continue

            F_n = weight[i, :]
            F_star_n, p_star, sim, scale = alg3(F_n, i, Y_i, ind, bn_weight, bn_bias, bn_mean, bn_var, lam)

            if threshold and sim >= threshold:
                Z_i[i, p_star] = scale

    return Y_i, Z_i

def alg3(F_n, F_n_ind, Y_i, ind, gamma_i, beta_i, mu_i, sigma_i, lam):
    '''
    F_n - 3D weight tensor (N_i, K, K), torch.Tensor
    Y_i - 2D weight tensor (P_{i+1}, N_i x K x K), torch.Tensor
    ind - indices of selected neurons
    mu_i - running_mean (for inference)
    sigma_i - running_var (for inference)
    gamma_i - batch norm weight
    beta_i - batch norm bias
    lam -  how much to consider cosine sim over bias, float value between 0 and 1
    bn_weight, bn_bias - parameters of batch norm layer right after the conv layer
    bn_mean, bn_var - running_mean, running_var of BN (for inference)
    lam -
    '''
    # F_n == weight
    # Y_i == weight_chosen
    # gamma_i == bn_weight
    # beta_i == bn_bias
    # mu_i == bn_mean
    # sigma_i == bn_var
    cos_list = []
    bias_list = []
    scale_list = []

    cos = nn.CosineSimilarity(dim=0, eps=1e-8)

    gamma_1 = gamma_i[F_n_ind]
    beta_1 = beta_i[F_n_ind]
    mu_1 = mu_i[F_n_ind]
    sigma_1 = sigma_i[F_n_ind]
    x_1_norm = torch.norm(F_n, p='fro')

    assert Y_i.shape[0] == len(ind)
    for m in range(Y_i.shape[0]):
        F_m = Y_i[m]
        cos_dist = 1 - cos(F_n, F_m)
        cos_list.append(cos_dist)

        # The BN parameters contain values for all of the neurons.
        # ind[m] returns the original index of neuron m
        gamma_2 = gamma_i[ind[m]]
        beta_2 = beta_i[ind[m]]
        mu_2 = mu_i[ind[m]]
        sigma_2 = sigma_i[ind[m]]
        x_2_norm = torch.norm(F_m, p='fro')

        s = x_1_norm / x_2_norm
        S = s * (gamma_2 / gamma_1) * (sigma_1 / sigma_2)
        scale_list.append(S)

        B = (gamma_2 / sigma_2) * (s * ((-(sigma_1 * beta_1) / gamma_1) + mu_1) - mu_2) + beta_2 # eq 8

        bias_list.append(abs(B) / S)

    bias_list_normalised = normalise_bias(bias_list)
    dist_list = get_dist_list(cos_list, bias_list_normalised, lam)
    min_dist = min(dist_list)
    min_ind = dist_list.index(min_dist)
    F_star_n = Y_i[min_ind,:]
    sim = 1 - cos_list[min_ind] # convert cos dist to cos sim
    scale = scale_list[min_ind]

    return F_star_n, min_ind, sim, scale


def normalise_bias(bias_list):
    min_bias = min(bias_list)
    max_bias = max(bias_list)
    bias_list_normalised = [(b - min_bias) / (max_bias - min_bias) for b in bias_list]

    return bias_list_normalised


def get_dist_list(cos_list, bias_list, lam):
    dist_list = []

    for (c, b) in zip(cos_list, bias_list):
        dist_list.append(c * lam + b * (1-lam))

    return dist_list


# This class reimplementation is based on the original code.
# Function names and variables closely match to allow better comparison.
class DecomposeRe:
    def __init__(self, arch, param_dict, criterion, threshold, lamda, model_type, cfg, cuda):
        self.param_dict = param_dict
        self.arch = arch
        self.criterion = criterion
        self.threshold = threshold
        self.lamda = lamda
        self.model_type = model_type
        self.cfg = cfg
        self.cuda = cuda
        self.output_channel_index = {}
        self.decompose_weight = []

    def main(self):
        if not self.cuda:
            for layer in self.param_dict:
                self.param_dict[layer] = self.param_dict[layer].cpu()

        self.get_decompose_weight()

        return self.decompose_weight

    def get_decompose_weight(self):
        # scale matrix
        z = None

        # copy original weight
        self.decompose_weight = list(self.param_dict.values())

        # cfg index
        layer_id = -1

        for index, layer in enumerate(self.param_dict):
            original = self.param_dict[layer]

            # VGG
            if self.arch == 'VGG':
                # feature
                if 'feature' in layer:
                    # conv
                    if len(self.param_dict[layer].shape) == 4:
                        layer_id += 1

                        # get index
                        self.output_channel_index[index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                        # Merge scale matrix
                        if z != None:
                            original = original[:,input_channel_index,:,:]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0],-1)
                                o = torch.mm(z,o)
                                o = o.view(z.shape[0],f.shape[1],f.shape[2])
                                original[i,:,:,:] = o


                        # make scale matrix with batchNorm
                        bn = list(self.param_dict.values())

                        # Reprod
                        bn_weight_tensor = bn[index+1].cpu().detach()
                        bn_bias_tensor = bn[index+2].cpu().detach()
                        bn_mean_tensor = bn[index+3].cpu().detach()
                        bn_var_tensor = bn[index+4].cpu().detach()

                        _, z = alg1_conv(self.param_dict[layer].cpu().detach(),
                                         self.output_channel_index[index],
                                         self.threshold,
                                         bn_weight_tensor, bn_bias_tensor, bn_mean_tensor, bn_var_tensor,
                                         self.lamda, self.model_type)

                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        # pruned
                        pruned = original[self.output_channel_index[index],:,:,:]

                        # update next input channel
                        input_channel_index = self.output_channel_index[index]

                        # update decompose weight
                        self.decompose_weight[index] = pruned

                    # batchNorm
                    elif len(self.param_dict[layer].shape):
                        # pruned
                        pruned = self.param_dict[layer][input_channel_index]

                        # update decompose weight
                        self.decompose_weight[index] = pruned

                # first classifier
                else:
                    pruned = torch.zeros(original.shape[0], z.shape[0])

                    if self.cuda:
                        pruned = pruned.cuda()

                    for i, f in enumerate(original):
                        o_old = f.view(z.shape[1], -1)
                        o = torch.mm(z, o_old).view(-1)
                        pruned[i, :] = o

                    self.decompose_weight[index] = pruned

                    break

            # ResNet
            elif self.arch == 'ResNet':
                # block
                if 'layer' in layer:
                    # last layer each block
                    if '0.conv1.weight' in layer:
                        layer_id += 1

                    # Pruning
                    if 'conv1' in layer:
                        # get index
                        self.output_channel_index[index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                        # make scale matrix with batchNorm
                        bn = list(self.param_dict.values())

                        # Reprod
                        bn_weight_tensor = bn[index + 1].cpu().detach()
                        bn_bias_tensor = bn[index + 2].cpu().detach()
                        bn_mean_tensor = bn[index + 3].cpu().detach()
                        bn_var_tensor = bn[index + 4].cpu().detach()

                        _, z = alg1_conv(self.param_dict[layer].cpu().detach(),
                                         self.output_channel_index[index],
                                         self.threshold,
                                         bn_weight_tensor, bn_bias_tensor, bn_mean_tensor, bn_var_tensor,
                                         self.lamda, self.model_type)

                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        # pruned
                        pruned = original[self.output_channel_index[index], :, :, :]

                        # update next input channel
                        input_channel_index = self.output_channel_index[index]

                        # update decompose weight
                        self.decompose_weight[index] = pruned

                    # batchNorm
                    elif 'bn1' in layer:
                        if len(self.param_dict[layer].shape):
                            # pruned
                            pruned = self.param_dict[layer][input_channel_index]

                            # update decompose weight
                            self.decompose_weight[index] = pruned

                    # Merge scale matrix
                    elif 'conv2' in layer:
                        if z != None:
                            original = original[:, input_channel_index, :, :]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0], -1)
                                o = torch.mm(z, o)
                                o = o.view(z.shape[0], f.shape[1], f.shape[2])
                                original[i, :, :, :] = o

                        scaled = original

                        # update decompose weight
                        self.decompose_weight[index] = scaled

            # WideResNet
            elif self.arch == 'WideResNet':
                # block
                if 'block' in layer:
                    # last layer each block
                    if '0.conv1.weight' in layer:
                        layer_id += 1

                    # Pruning
                    if 'conv1' in layer:
                        # get index
                        self.output_channel_index[index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                        # make scale matrix with batchNorm
                        bn = list(self.param_dict.values())

                        # Reprod
                        bn_weight_tensor = bn[index + 1].cpu().detach()
                        bn_bias_tensor = bn[index + 2].cpu().detach()
                        bn_mean_tensor = bn[index + 3].cpu().detach()
                        bn_var_tensor = bn[index + 4].cpu().detach()

                        _, z = alg1_conv(self.param_dict[layer].cpu().detach(),
                                         self.output_channel_index[index],
                                         self.threshold,
                                         bn_weight_tensor, bn_bias_tensor, bn_mean_tensor, bn_var_tensor,
                                         self.lamda, self.model_type)

                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        # pruned
                        pruned = original[self.output_channel_index[index], :, :, :]

                        # update next input channel
                        input_channel_index = self.output_channel_index[index]

                        # update decompose weight
                        self.decompose_weight[index] = pruned

                    # BatchNorm
                    elif 'bn2' in layer:
                        if len(self.param_dict[layer].shape):
                            # pruned
                            pruned = self.param_dict[layer][input_channel_index]

                            # update decompose weight
                            self.decompose_weight[index] = pruned

                    # Merge scale matrix
                    elif 'conv2' in layer:
                        # scale
                        if z != None:
                            original = original[:, input_channel_index, :, :]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0], -1)
                                o = torch.mm(z, o)
                                o = o.view(z.shape[0], f.shape[1], f.shape[2])
                                original[i, :, :, :] = o

                        scaled = original

                        # update decompose weight
                        self.decompose_weight[index] = scaled

            # LeNet_300_100
            elif self.arch == 'LeNet_300_100':
                # ip
                if layer in ['ip1.weight', 'ip2.weight']:
                    # Merge scale matrix
                    if z != None:
                        original = torch.mm(original, z)

                    layer_id += 1

                    # concatenate weight and bias
                    if layer in 'ip1.weight':
                        # Reprod
                        weight_tensor = self.param_dict['ip1.weight'].detach()
                        bias_tensor = self.param_dict['ip1.bias'].detach()

                    elif layer in 'ip2.weight':
                        # Reprod
                        weight_tensor = self.param_dict['ip2.weight'].detach()
                        bias_tensor = self.param_dict['ip2.bias'].detach()

                    # Reprod
                    bias_reshaped_tensor = bias_tensor.view(bias_tensor.shape[0], -1)
                    concat_weight_tensor = torch.cat((weight_tensor, bias_reshaped_tensor), 1)

                    # get index
                    self.output_channel_index[index] = self.get_output_channel_index(concat_weight_tensor, layer_id)

                    _, z = alg1_FC(concat_weight_tensor, self.output_channel_index[index], self.threshold, self.model_type)

                    if self.cuda:
                        z = z.cuda()

                    # pruned
                    pruned = original[self.output_channel_index[index], :]

                    # update next input channel
                    input_channel_index = self.output_channel_index[index]

                    # update decompose weight
                    self.decompose_weight[index] = pruned

                elif layer in 'ip3.weight':

                    original = torch.mm(original, z)

                    # update decompose weight
                    self.decompose_weight[index] = original

                # update bias
                elif layer in ['ip1.bias', 'ip2.bias']:
                    self.decompose_weight[index] = original[input_channel_index]

                else:
                    pass
            # AlexNet
            elif self.arch in ['AlexNet_CIFAR100', 'AlexNet_ImageNet']:
                if layer in ['classifier.1.weight', 'classifier.4.weight']:

                    # Merge scale matrix
                    if z != None:
                        if self.cuda:
                            z = z.cuda()
                            original = original.cuda()
                        original = torch.mm(original, z)
                    layer_id += 1

                    # concatenate weight and bias
                    if layer in 'classifier.1.weight':
                        weight_tensor = self.param_dict['classifier.1.weight'].cpu().detach()
                        bias_tensor = self.param_dict['classifier.1.bias'].cpu().detach()

                    elif layer in 'classifier.4.weight':
                        weight_tensor = self.param_dict['classifier.4.weight'].cpu().detach()
                        bias_tensor = self.param_dict['classifier.4.bias'].cpu().detach()

                    bias_reshaped_tensor = bias_tensor.reshape(bias_tensor.shape[0], -1)
                    concat_weight_tensor = torch.cat([weight_tensor, bias_reshaped_tensor], 1)

                    # get index
                    self.output_channel_index[index] = self.get_output_channel_index(concat_weight_tensor, layer_id)

                    _, z = alg1_FC(concat_weight_tensor, self.output_channel_index[index], self.threshold, self.model_type)

                    if self.cuda:
                        z = z.cuda()
                    # pruned
                    pruned = original[self.output_channel_index[index], :]
                    # update next input channel
                    input_channel_index = self.output_channel_index[index]
                    # update decompose weight
                    self.decompose_weight[index] = pruned

                elif layer in 'classifier.6.weight':
                    if self.cuda:
                        z = z.cuda()
                        original = original.cuda()
                    original = torch.mm(original, z)
                    # update decompose weight
                    self.decompose_weight[index] = original

                # update bias
                elif layer in ['classifier.1.bias', 'classifier.4.bias']:
                    self.decompose_weight[index] = original[input_channel_index]

    def get_output_channel_index(self, value, layer_id):
        output_channel_index = []

        if len(value.size()):
            weight_vec = value.view(value.size()[0], -1)
            weight_vec = weight_vec.cuda()

            # l1-norm
            if self.criterion == 'l1-norm':
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().detach().numpy()
                arg_max = np.argsort(norm_np)
                arg_max_rev = arg_max[::-1][:self.cfg[layer_id]]
                output_channel_index = sorted(arg_max_rev.tolist())

            # l2-norm
            elif self.criterion == 'l2-norm':
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().detach().numpy()
                arg_max = np.argsort(norm_np)
                arg_max_rev = arg_max[::-1][:self.cfg[layer_id]]
                output_channel_index = sorted(arg_max_rev.tolist())

            # l2-GM
            elif self.criterion == 'l2-GM':
                weight_vec = weight_vec.cpu().detach().numpy()
                matrix = distance.cdist(weight_vec, weight_vec, 'euclidean')
                similar_sum = np.sum(np.abs(matrix), axis=0)

                output_channel_index = np.argpartition(similar_sum, -self.cfg[layer_id])[-self.cfg[layer_id]:]
                output_channel_index = output_channel_index.tolist()

        return output_channel_index
