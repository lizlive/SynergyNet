import torch
from torch.nn import *
from torch import nn
import numpy as np
from scipy.ndimage import gaussian_filter1d
EPS = 1e-12

class Multiply(nn.Module):
    def __init__(self, factor):
        super(Multiply, self).__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor

class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class SynergyNet_my_regression(nn.Module):
    def __init__(self, config):
        super(SynergyNet_my_regression, self).__init__()
        self.input_dim = config.input_dim
        self.intermediate_dim = config.intermediate_dim
        self.timeConvLen = config.timeConvLen
        self.embedding_dim = config.embedding_dim
        self.output_dim = config.output_dim
        self.n_step = config.n_step # The number of time steps
        self.step_len = config.step_len
        self.kin_intermediate_dim = config.kin_intermediate_dim
        self.kin_dictionary_num = config.kin_dictionary_num # The number of kinematic dictionary
        self.kin_dictionary_timeNum = config.kin_dictionary_timeNum # The number of time steps for kinematic dictionary
        self.alpha = config.alpha # tradeoff of decoding and reconstruction
        self.beta_smooth = config.beta_smooth
        self.beta_diversity = config.beta_diversity

        self.window_len = self.step_len * (self.n_step-1) / 1000 # The time window length in ms
        self.tpi = Parameter(torch.from_numpy(np.array([2.0 * np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(
            torch.from_numpy(np.linspace(-self.window_len, 0, self.n_step, dtype=np.float32)),
            requires_grad=False) # delta time for each bin
        self.freqs = Parameter(torch.fft.rfftfreq(self.n_step)[1:] * self.n_step / self.window_len, requires_grad=False)  # Remove DC frequency

        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, self.intermediate_dim, self.timeConvLen, stride=1,
                      padding=int((self.timeConvLen- 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros'),
            LN_v2(self.n_step),
            nn.ReLU(),
            nn.Conv1d(self.intermediate_dim, self.embedding_dim, self.timeConvLen, stride=1,
                      padding=int((self.timeConvLen - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        )

        self.fc_1 = nn.Conv1d(self.embedding_dim, self.embedding_dim, self.n_step, stride=1,
                              padding=0, dilation=1, groups=self.embedding_dim,
                              bias=True)
        self.fc_2 = nn.Conv1d(self.embedding_dim, self.embedding_dim, self.n_step, stride=1,
                              padding=0, dilation=1, groups=self.embedding_dim,
                              bias=True)

        self.decoder = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.intermediate_dim, self.timeConvLen, stride=1,
                      padding=int((self.timeConvLen - 1) / 2), dilation=1, groups=1, bias=True,
                      padding_mode='zeros'),
            nn.ReLU(),
            LN_v2(self.n_step),
            nn.Conv1d(self.intermediate_dim, self.input_dim,self.timeConvLen, stride=1,
                     padding=int((self.timeConvLen - 1) / 2), dilation=1, groups=1, bias=True,
                     padding_mode='zeros')
        )

        self.kin_amp = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.kin_intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.kin_intermediate_dim, self.kin_dictionary_num),
            nn.ReLU())
        self.kin_phase = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.kin_intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.kin_intermediate_dim, self.kin_dictionary_num),
            nn.Tanh(),
            Multiply(0.5))

        self.kin_dictionary = Parameter(torch.empty(self.kin_dictionary_num, self.output_dim, self.kin_dictionary_timeNum), requires_grad=True)

        self.loss_function = torch.nn.MSELoss()

        # 初始化权重
        self.init_weights()

    def compute_entropy(self, cov_matrix):
        """
        使用协方差矩阵近似计算信息熵
        """
        cov_matrix += torch.eye(cov_matrix.size(0)) * 1e-6  # 避免数值不稳定
        entropy = 0.5 * torch.logdet(cov_matrix)
        return entropy

    def compute_smoothness_loss(self, kin_dictionary):
        """
        计算一阶差分平滑性损失，并对时间步长度进行标准化
        :param dictionary: 字典矩阵 (n_atoms, n_features, n_time)
        """
        _, _, n_time = kin_dictionary.size()
        # 一阶差分平滑性约束
        smooth_loss = torch.sum((kin_dictionary[:, :, 1:] - kin_dictionary[:, :, :-1]) ** 2)

        # 时间步标准化
        smooth_loss /= (n_time - 1)
        return smooth_loss

    def compute_diversity_loss(self, kin_dictionary):
        # 多样性约束
        # 基于正交性的约束
        n_atoms, n_features, n_time = kin_dictionary.size()
        # 将所有时间步的字典基向量拼接
        D_pooled = kin_dictionary.reshape(n_atoms, n_features*n_time)

        gram_matrix = D_pooled @ D_pooled.T
        orth_loss = torch.norm(gram_matrix - torch.eye(n_atoms, device=kin_dictionary.device)) ** 2
        orth_loss /= n_atoms * (n_atoms - 1) / 2  # 归一化
        return orth_loss


    def random_idx_class(self, labels_dis):
        idx_shuffle = torch.arange(labels_dis.shape[0]).to(self.fc_1.weight.device)
        labels_dis_list = torch.unique(labels_dis)
        for label in labels_dis_list:
            idx_label_tmp = (labels_dis == label).nonzero()[:,0]
            idx = torch.randperm(idx_label_tmp.nelement())
            idx_label_tmp_shuffle = idx_label_tmp[idx]
            idx_shuffle[idx_label_tmp] = idx_label_tmp_shuffle
        return idx_shuffle
    #Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] #Spectrum without DC component
        power = spectrum**2

        #Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

        #Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.n_step

        #Offset
        offset = rfft.real[:,:,0] / self.n_step #DC component

        return freq, amp, offset

    def kin_decoder(self, amp, phase):

        # 将 phase 从 [-0.5, 0.5] 映射到 [0, kin_dictionary_timeNum - 1]
        phase_mapped = (phase + 0.5) * (self.kin_dictionary_timeNum - 1)

        # 计算下边界和上边界的整数索引
        index_floor = torch.floor(phase_mapped).long()  # 下边界索引
        index_ceil = torch.ceil(phase_mapped).long()  # 上边界索引

        # 计算插值权重
        weight_ceil = phase_mapped - index_floor.float()
        weight_floor = 1 - weight_ceil

        index_floor = index_floor.unsqueeze(-1).repeat(1,1,self.kin_dictionary.size(1)).unsqueeze(-1) # [B, hidden, out, 1]
        index_ceil = index_ceil.unsqueeze(-1).repeat(1,1,self.kin_dictionary.size(1)).unsqueeze(-1) # [B, hidden, out, 1]

        kin_dictionary = self.kin_dictionary.unsqueeze(0).repeat(phase_mapped.size(0),1,1,1) # [B, hidden, out, T]
        # 获取 kin_dictionary 在两个相邻时间步的切片
        dictionary_slice_floor = torch.gather(kin_dictionary, dim=3, index=index_floor).squeeze(-1) # 下边界切片
        dictionary_slice_ceil = torch.gather(kin_dictionary, dim=3, index=index_ceil).squeeze(-1)   # 上边界切片

        # 在两个时间步之间进行线性插值
        dictionary_slice = weight_floor.unsqueeze(-1) * dictionary_slice_floor + weight_ceil.unsqueeze(-1) * dictionary_slice_ceil

        # 用 amp 激活 dictionary_slice
        output = (amp.unsqueeze(-1) * dictionary_slice).sum(dim=1)  #  [B, output_dim]

        return output


    def kin_decoder_dics(self, amp, phase):

        # 将 phase 从 [-0.5, 0.5] 映射到 [0, kin_dictionary_timeNum - 1]
        phase_mapped = (phase + 0.5) * (self.kin_dictionary_timeNum - 1)

        # 计算下边界和上边界的整数索引
        index_floor = torch.floor(phase_mapped).long()  # 下边界索引
        index_ceil = torch.ceil(phase_mapped).long()  # 上边界索引

        # 计算插值权重
        weight_ceil = phase_mapped - index_floor.float()
        weight_floor = 1 - weight_ceil

        index_floor = index_floor.unsqueeze(-1).repeat(1,1,self.kin_dictionary.size(1)).unsqueeze(-1) # [B, hidden, out, 1]
        index_ceil = index_ceil.unsqueeze(-1).repeat(1,1,self.kin_dictionary.size(1)).unsqueeze(-1) # [B, hidden, out, 1]

        kin_dictionary = self.kin_dictionary.unsqueeze(0).repeat(phase_mapped.size(0),1,1,1) # [B, hidden, out, T]
        # 获取 kin_dictionary 在两个相邻时间步的切片
        dictionary_slice_floor = torch.gather(kin_dictionary, dim=3, index=index_floor).squeeze(-1) # 下边界切片
        dictionary_slice_ceil = torch.gather(kin_dictionary, dim=3, index=index_ceil).squeeze(-1)   # 上边界切片

        # 在两个时间步之间进行线性插值
        dictionary_slice = weight_floor.unsqueeze(-1) * dictionary_slice_floor + weight_ceil.unsqueeze(-1) * dictionary_slice_ceil

        # 用 amp 激活 dictionary_slice
        output = (amp.unsqueeze(-1) * dictionary_slice)  #  [B, n_dim, output_dim]

        return output
    def forward_noLabel(self, x):
        latent = self.encoder(x)  # B, C, T to B, C1, T
        # Frequency, Amplitude, Offset
        f, a, b = self.FFT(latent, dim=2)
        # Phase
        v1 = torch.squeeze(self.fc_1(latent), -1)
        v2 = torch.squeeze(self.fc_2(latent), -1)
        p = torch.atan2(v1, v2) / self.tpi

        m_x = a * torch.sin(self.tpi * p) + b
        m_y = a * torch.cos(self.tpi * p) + b
        phase_manifold = torch.cat((m_x, m_y), dim=1)

        kin_a = self.kin_amp(phase_manifold)
        kin_p = self.kin_phase(phase_manifold)
        kin_pre = self.kin_decoder(kin_a, kin_p)
        kin_pre_dics = self.kin_decoder_dics(kin_a, kin_p)


        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)

        # Latent Reconstruction
        latent_rec = a * torch.sin(self.tpi * (f * self.args + p)) + b

        x_rec = self.decoder(latent_rec)

        loss_rec = self.loss_function(x_rec.reshape(x_rec.shape[0], self.input_dim * self.n_step),
                                      x.reshape(x.shape[0], self.input_dim * self.n_step))
        return latent, p, f, a, b, latent_rec, x_rec, kin_pre, loss_rec, phase_manifold, kin_a, kin_p, kin_pre_dics
    def forward(self, x, labels=None, labels_dis = None, labels_dis_process = None):
        latent = self.encoder(x)  # B, C, T to B, C1, T
        # Frequency, Amplitude, Offset
        f, a, b = self.FFT(latent, dim=2)
        # Phase
        v1 = torch.squeeze(self.fc_1(latent), -1)
        v2 = torch.squeeze(self.fc_2(latent), -1)
        p = torch.atan2(v1, v2) / self.tpi

        m_x = a * torch.sin(self.tpi * p) + b
        m_y = a * torch.cos(self.tpi * p) + b
        phase_manifold = torch.cat((m_x, m_y), dim=1)

        kin_a = self.kin_amp(phase_manifold)
        kin_p = self.kin_phase(phase_manifold)
        kin_pre = self.kin_decoder(kin_a, kin_p)

        # if self.training with labels input
        if labels is not None:
            idx_shuffle_label = self.random_idx_class(labels_dis)
            kin_a_swap = kin_a[idx_shuffle_label, :]
            kin_pre_swap_a = self.kin_decoder(kin_a_swap, kin_p)

            idx_shuffle_label = self.random_idx_class(labels_dis_process)
            kin_p_swap = kin_p[idx_shuffle_label, :]
            kin_pre_swap_p = self.kin_decoder(kin_a, kin_p_swap)


        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)

        # Latent Reconstruction
        latent_rec = a * torch.sin(self.tpi * (f * self.args + p)) + b

        x_rec = self.decoder(latent_rec)

        loss_rec = self.loss_function(x_rec.reshape(x_rec.shape[0], self.input_dim * self.n_step),
                                      x.reshape(x.shape[0], self.input_dim * self.n_step))
        # 如果传入真实标签，就直接计算损失值
        if labels is not None:
            loss_reg = self.loss_function(kin_pre.view(-1, self.output_dim), labels.contiguous().float())
            loss_reg_swap_a = self.loss_function(kin_pre_swap_a.view(-1, self.output_dim), labels.contiguous().float())
            loss_reg_swap_p = self.loss_function(kin_pre_swap_p.view(-1, self.output_dim), labels.contiguous().float())
            loss_dic_smooth = self.compute_smoothness_loss(self.kin_dictionary)
            loss_dic_diversity = self.compute_diversity_loss(self.kin_dictionary)
            loss = (1-self.alpha) * loss_rec + self.alpha * (loss_reg+loss_reg_swap_a+loss_reg_swap_p)+self.beta_smooth*loss_dic_smooth+self.beta_diversity*loss_dic_diversity
            return loss
        else:
            return kin_pre, x_rec


    def reset_parameters(self):
        pass

    def init_weights(self):
        """
        initial kin_dictionary，
        """
        raw_dictionary = torch.randn(self.kin_dictionary_num, self.output_dim,
                                     self.kin_dictionary_timeNum)
        for i in range(self.kin_dictionary_num):
            for j in range(self.output_dim):
                raw_dictionary[i, j, :] = torch.tensor(
                    gaussian_filter1d(raw_dictionary[i, j, :].numpy(), sigma=self.kin_dictionary_timeNum/3)
                )

        with torch.no_grad():
            self.kin_dictionary.copy_(raw_dictionary)



