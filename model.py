import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

class Model(nn.Module):

    def __init__(self, n_items, n_layers=1, n_heads=1, hidden_size=64, dropout_prob=0.5, max_insert_size=5):
        super(Model, self).__init__()

        # load parameters info
        self.n_items = n_items  # note that include padding and EOS and mask
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.inner_size = self.hidden_size * 4
        self.full_layer = nn.Linear(self.hidden_size, 3)
        self.dropout_prob = dropout_prob
        self.max_insert_size = max_insert_size
        self.hidden_act = 'relu'

        self.alpha = 0.1
        self.beta = 0.001
        self.tau = 1
        self.sim = 'dot'

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(100, self.hidden_size)
        self.net = nn.ModuleList(TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_heads, dim_feedforward=self.inner_size,
                                                                 dropout=self.dropout_prob, activation=self.hidden_act) for _ in range(self.n_layers))

        self.recommender = TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_heads, dim_feedforward=self.inner_size,
                                                                 dropout=self.dropout_prob, activation=self.hidden_act)
                                                                 # note that d_model%nhead=0 and the final layer of TransformerEncoderLayer is dropout

        self.insert_net = TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_heads, dim_feedforward=self.inner_size,
                                                                dropout=self.dropout_prob, activation='relu')
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.crossEntropy = torch.nn.CrossEntropyLoss(reduction='none')

        self.batch_size = 256
        self.mask = self.mask_correlated_samples(self.batch_size)

        self.nce_fct = nn.CrossEntropyLoss(reduction='mean')


    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)  # mask self sim
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.（对应的另一个变换作为正例）
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)  # [2*batch_size, emb_size]

        if sim == 'cos':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp  # [2*batch_size, 2*batch_size]

        sim_i_j = torch.diag(sim, batch_size)  # [batch_size]
        sim_j_i = torch.diag(sim, -batch_size)  # [batch_size]

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.mask = self.mask_correlated_samples(batch_size)
        mask = self.mask

        negative_samples = sim[mask].reshape(N, -1)  # neg_sim，[2*batch_size, 2*batch_size-2]

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        return logits, labels
    
    def seq2tensor(self, seqs):
        """
        Get item embeddings
        """
        seqs_emb = self.item_embedding(seqs)

        positions = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0], 1])

        positions = torch.tensor(positions, device=seqs_emb.device).long()

        seqs_emb += self.position_embedding(positions)

        return seqs_emb

    def encoder(self, seqs, padding_mask):
        """
        Encoder part
        """
        seqs_emb = self.seq2tensor(seqs)

        seqs_emb = self.dropout(seqs_emb)

        seqs_emb = torch.transpose(seqs_emb, 0, 1)

        encoder_output = seqs_emb   #[seq_len, batch_size,  emb_size]

        src_mask = (1-torch.tril(torch.ones(encoder_output.shape[0], encoder_output.shape[0], device=padding_mask.device))).bool()  # [seq_len, seq_len]

        for mod in self.net:

            encoder_output = mod(encoder_output, src_key_padding_mask=padding_mask, src_mask=src_mask)

        return encoder_output  #[seq_len, batch_size,  emb_size]
    
    def forward(self, input_session_ids, item_seq_len):
        padding_mask = (input_session_ids == 0)
        encoder_output = self.encoder(input_session_ids, padding_mask)
        src_mask = (1-torch.tril(torch.ones(encoder_output.shape[0], encoder_output.shape[0], device=padding_mask.device))).bool()  # [seq_len, seq_len]
        output = self.recommender(encoder_output, src_key_padding_mask=padding_mask, src_mask=src_mask)  # [seq_len, batch_size, emb_size]
        output = output.permute(1, 0, 2) 
        output = self.gather_indexes(output, item_seq_len - 1)
        return output
        
    def cal_distance(self, feature1, feature2):

        #difference = feature1 - feature2
        #squared_distance = torch.sum(torch.pow(difference, 2))
        #euclidean_distance = torch.sqrt(squared_distance)
        
        distance = torch.sum(feature1 * feature2)
        return distance

    def tri_cl_loss(self, raw_emb, aug_emb_1, aug_emb_2):
        '''
        :param raw_emb: [batch_size, emb_size]
        :param aug_emb_1: [batch_size, emb_size]
        :param aug_emb_2: [batch_size, emb_size]
        :return: bpr loss
        '''
        negative_sims = torch.sum(raw_emb * aug_emb_1, dim=1)
        positive_sims = torch.sum(raw_emb * aug_emb_2, dim=1)
        positive_exp = torch.exp(positive_sims)   # [batch_size]
        negative_exp = torch.exp(negative_sims)   # [batch_size]

        x = positive_exp + negative_exp
        x[x == 0] = 0.00001
        temp = positive_exp / x     # [batch_size]
        loss = -1 * torch.sum(torch.log(temp))

        return loss

    def train_forward(self, input_session_ids, aug_item_seq1, aug_item_seq2):
        item_seq_len = (input_session_ids > 0).sum(-1)  # [batch_size]
        output = self.forward(input_session_ids, item_seq_len)

        aug_len1 = (aug_item_seq1 > 0).sum(-1)
        seq_output1 = self.forward(aug_item_seq1, aug_len1)

        aug_len2 = (aug_item_seq2 > 0).sum(-1)
        seq_output2 = self.forward(aug_item_seq2, aug_len2)
                
        distance1 = self.cal_distance(output, seq_output1)
        distance2 = self.cal_distance(output, seq_output2)
        
        nce_logits, nce_labels = self.info_nce(seq_output1, seq_output2, temp=self.tau, batch_size=aug_len1.shape[0], sim=self.sim)
  
        nce_loss = self.alpha * self.nce_fct(nce_logits, nce_labels)
        tri_cl_loss = self.beta * self.tri_cl_loss(output, seq_output1, seq_output2)
  
        return output, nce_loss, tri_cl_loss, distance1, distance2


    def test_forward(self, input_session_ids):
        item_seq_len = (input_session_ids > 0).sum(-1)  # [batch_size]
        output = self.forward(input_session_ids, item_seq_len)
        return output

    def rec_loss(self, output, targets, nce_loss, tri_cl_loss):
        output = torch.matmul(output, self.item_embedding.weight.T)  # [batch_size, item_num]
        targets = targets.unsqueeze(-1)  # [batch_size, 1]
        rec_loss = -output.log_softmax(dim=-1).gather(dim=-1, index=targets).squeeze(-1)  # [batch_size]
        loss = rec_loss.mean()
        loss += nce_loss + tri_cl_loss
        return loss


    #train augmentation mudule
    def insert(self, encoder_output, insert_seqs,
               insert_padding_mask):
        """
        At each time step, regardless of whether the model predicts the need for insertion, the insert model needs to be
        substituted
        """
        insert_seqs_emb = self.item_embedding(insert_seqs)  # (batch_size,seqs_len,max_insert_size-1,emb_size)

        encoder_output = torch.transpose(encoder_output, 0, 1)  # (batch_size,seqs_len,emb_size)

        encoder_output = encoder_output.unsqueeze(2)  # (batch_size,seqs_len,1,emb_size)

        insert_seqs_emb = torch.cat([encoder_output, insert_seqs_emb], dim=2)  # (batch, seqs_len, max_insert_size, emb_size)

        insert_seqs_emb = torch.reshape(insert_seqs_emb, (
            insert_seqs_emb.shape[0] * insert_seqs_emb.shape[1], insert_seqs_emb.shape[2], insert_seqs_emb.shape[3]))  # batch_size*seqs_len,max_insert,emb_size）

        positions = np.tile(np.array(range(insert_seqs_emb.shape[1])), [insert_seqs_emb.shape[0], 1])  # (batch_size*seqs_len,max_insert_size)

        position_emb = self.position_embedding(torch.tensor(positions, device=insert_seqs_emb.device).long())  # (batch_size*seqs_len,max_insert_size,emb_size)

        insert_seqs_emb += position_emb

        insert_seqs_emb = self.dropout(insert_seqs_emb)

        insert_seqs_emb = torch.transpose(insert_seqs_emb, 0, 1)  # (max_insert_size, batch_size*seqs_len, emb_size)

        src_mask = (1 - torch.tril(
            torch.ones(insert_seqs_emb.shape[0], insert_seqs_emb.shape[0], device=insert_seqs_emb.device))).bool()  # previous time steps are not visible to subsequent time steps

        insert_padding_mask[:, 0] = False  # if the whole sequence is padding, nan error will occur

        insert_net_output = self.insert_net(insert_seqs_emb, src_key_padding_mask=insert_padding_mask, src_mask=src_mask)
        # (max_insert_size, batch_size*seqs_len, emb_size)

        insert_net_output = torch.matmul(insert_net_output, self.item_embedding.weight.T)  # (max_insert_size, batch_size*seqs_len, item_num)

        return insert_net_output

    def augmentor_forward(self, input_seqs, input_insert_seqs):
        """
        Item-wise augmentor part
        :param input_seqs: (batch_size,seqs_len)
        :param input_insert_seqs: (batch_size,seqs_len,max_insert_size-1)
        """
        padding_mask = (input_seqs == 0)  # [batch, seq_len]

        encoder_output = self.encoder(input_seqs, padding_mask)  # (seqs_len,batch_size,emb_size)

        full_layer_output = self.full_layer(encoder_output)  # (seqs_len,batch_size,3), in addition, 3 refers to keep, delete, insert

        temp_input_seqs = input_seqs.unsqueeze(-1)  # (batch_size,seqs_len,1)

        temp_input_insert_mask = torch.cat([temp_input_seqs, input_insert_seqs],
                                           dim=-1)  # (batch_size,seqs_len,max_insert_size)

        insert_padding_mask = (temp_input_insert_mask == 0)  # (batch_size,seqs_len,max_insert_size)

        insert_padding_mask = torch.reshape(insert_padding_mask, (
            insert_padding_mask.shape[0] * insert_padding_mask.shape[1],
            insert_padding_mask.shape[2]))  # (batch_size * seqs_len, max_insert_size)

        insert_net_output = self.insert(encoder_output, input_insert_seqs, insert_padding_mask)

        full_layer_output = torch.transpose(full_layer_output, 0, 1)  # (batch_size,seqs_len,3)

        insert_net_output = torch.transpose(insert_net_output, 0, 1)  # (batch_size*seqs_len,max_insert_size,item_num)

        insert_net_output = torch.reshape(insert_net_output, (
            temp_input_insert_mask.shape[0], temp_input_insert_mask.shape[1], temp_input_insert_mask.shape[2],
            insert_net_output.shape[2]))  # (batch_size,seqs_len,max_insert_size,item_num)

        return full_layer_output, insert_net_output, padding_mask

    def l1_loss(self, l1_ground_truth, full_layer_output,
                padding_mask):
        """
        As for the time step of padding, l1_loss equals to 0
        """
        padding_mask = padding_mask.float()  # (batch_size,seqs_len)

        full_layer_output = torch.transpose(full_layer_output, 1, 2)  # (batch_size,3,seqs_len)

        cross_entropy_l1 = self.crossEntropy(full_layer_output, l1_ground_truth)  # (batch_size,seqs_len)

        input_padding = 1 - padding_mask

        cross_entropy_l1 *= input_padding

        return cross_entropy_l1

    def l2_loss(self, insert_net_output, l1_ground_truth, l2_ground_truth):
        """
        Calculating l2_loss needs two types of mask
        The first one is:
        Exclude time steps correspond to keep operation and delete operation
        The second one is:
        Exclude the position that corresponds to EOS token in the input
        For example,
        0 1 2 3 4 -> 1 2 3 4 5  all of five positions calculate the loss
        0 1 2 3 4 -> 1 2 3 4 EOS  all of five positions calculate the loss
        0 1 2 3 eos -> 1 2 3 eos padding   all of the positions calculate the loss except for the position corresponds to EOS, that is, first position, second position, third position, forth position
        Generally, as for sequence 0 1 2 3 4, 0 corresponds to input_seqs, 1 2 3 4 corresponds to input_insert_seqs
        """
        insert_net_output = insert_net_output.permute(0, 3, 1, 2)  # (batch,item_num,seqs_len,max_insert_size)

        cross_entropy_l2 = self.crossEntropy(insert_net_output, l2_ground_truth)  # (batch_size,seqs_len,max_insert_size)

        # first mask

        insert_mask = (l1_ground_truth == 2).float()  # (batch_size,seqs_len)

        insert_mask = insert_mask.unsqueeze(-1)  # (batch_size,seqs_len,1)

        cross_entropy_l2 *= insert_mask

        # second mask

        insert_seq_mask = (l2_ground_truth != 0).float()  # (batch_size, seqs_len,max_insert_size)

        cross_entropy_l2 *= insert_seq_mask

        return cross_entropy_l2

    def augmentor_loss(self, full_layer_output, insert_net_output, l1_ground_truth, l2_ground_truth, padding_mask):
        """
        l1_ground_truth corresponds to the actual operation of each time step (keep, delete or insert)
        That is 0 for keep, 1 for delete, 2 for insert
        l2_ground_truth corresponds to the sequence that should be inserted ahead for each time step
        l1_loss refers to the loss of the keep delete insert operation predicted by the model
        l2_loss calculates the loss doing insert operation
        """
        l1_loss_entropy = self.l1_loss(l1_ground_truth, full_layer_output, padding_mask)

        l2_loss_entropy = self.l2_loss(insert_net_output, l1_ground_truth, l2_ground_truth)

        return l1_loss_entropy, l2_loss_entropy

    def augmentor_inference(self, input_seqs):
        """
        Augment the original sequence
        return: augmented sequence
        """
        #
        padding_mask = (input_seqs == 0)

        encoder_output = self.encoder(input_seqs, padding_mask)  # [seq_len, batch, emb_size]

        full_layer_output = self.full_layer(encoder_output)  # [seq_len, batch, 3]

        random_decisions = torch.rand(full_layer_output.shape[0], full_layer_output.shape[1], 3) # [seq_len, batch, 3]
        
        device = full_layer_output.device
        
        random_decisions = random_decisions.to(device)

        full_layer_output = torch.mul(full_layer_output, random_decisions)

        # random_decisions = torch.rand(full_layer_output.shape[0], full_layer_output.shape[1])  # [seq_len, batch]
        #
        # for i in full_layer_output.shape[0]:
        #     for j in full_layer_output.shape[1]:
        #         if random_decisions[i, j] <= full_layer_output[i, j, 0]:
        #             full_layer_output[i, j, 0] = 0.999
        #         elif full_layer_output[i, j, 0] < random_decisions[i, j] <= (full_layer_output[i, j, 0] + full_layer_output[i, j, 1]):
        #             full_layer_output[i, j, 1] = 0.999
        #         else:
        #             full_layer_output[i, j, 2] = 0.999

        decisions = full_layer_output.argmax(-1)  # (seqs_len, batch_size)
        decisions = torch.transpose(decisions, 0, 1)  # (batch_size, seqs_len, 3)

        encoder_output = torch.transpose(encoder_output, 0, 1)  # (batch_size, seqs_len, emb_size)

        input_insert_embedding = torch.reshape(encoder_output, (1, encoder_output.shape[0] * encoder_output.shape[1], encoder_output.shape[2]))  # (1, batch_size*seqs_len, emb_size)

        for i in range(self.max_insert_size):

            positions = np.tile(np.array(range(input_insert_embedding.shape[0])), [input_insert_embedding.shape[1], 1])

            positions = torch.tensor(positions, device=input_insert_embedding.device).long()

            positions = torch.transpose(positions, 0, 1)  # (1, batch_size*seqs_len, 1)

            input_insert_embedding += self.position_embedding(positions)  # [1, batch*seq_len, emb_size]

            if i == 0:
                tgt_mask = (1 - torch.tril(torch.ones(i + 1, i + 1, device=input_insert_embedding.device))).bool()

                insert_output = self.insert_net(input_insert_embedding, src_mask=tgt_mask)  # (1, batch_size*seqs_len, emb_size)

                insert_output = torch.matmul(insert_output[-1, :, :],
                                             self.item_embedding.weight.T)  # (batch_size*seqs_len, item_num)

                i_insert_seqs = insert_output.argmax(-1, keepdim=True)   # (batch_size*seqs_len, 1)

                insert_seqs = i_insert_seqs  # (batch_size*seqs_len, 1)

            else:
                tgt_mask = (1 - torch.tril(torch.ones(i + 1, i + 1, device=input_insert_embedding.device))).bool()

                insert_output = self.insert_net(input_insert_embedding, src_mask=tgt_mask)  # (i+1, batch_size*seqs_len, emb_size)

                insert_output = torch.matmul(insert_output[-1, :, :], self.item_embedding.weight.T)  # (batch_size*seqs_len, item_num)

                i_insert_seqs = insert_output.argmax(-1, keepdim=True)  # (batch_size*seqs_len, 1)

                insert_seqs = torch.cat([insert_seqs, i_insert_seqs], -1)  # (batch_size*seqs_len, i+1)

            i_insert_seqs_embedding = self.item_embedding(i_insert_seqs)  # (batch_size*seqs_len, 1, emb_size)
            
            i_insert_seqs_embedding = torch.transpose(i_insert_seqs_embedding, 0, 1)  # (1, batch_size*seqs_len, emb_size)
            
            input_insert_embedding = torch.cat([input_insert_embedding, i_insert_seqs_embedding])  # (i+1+1, batch_size*seqs_len, emb_size)

        insert_seqs = torch.reshape(insert_seqs, (decisions.shape[0], decisions.shape[1], -1))  # (batch_size,seqs_len,max_insert_size)

        return decisions, insert_seqs

    def seqs_augmentation(self, decisions, insert_seqs, input_seqs):
        """
        Through function augmentor_inference, we can get the modified prediction for every time step of each sequence,
        this function augments the original sequence via the result from the function augmentor_inference
        input_seqs: [batch, seq_len]
        decisions: [batch, seq_len]
        insert_seqs: [batch, seq_len, max_insert_size]
        """
        modified_seqs = input_seqs.clone()

        decisions[modified_seqs == 0] = 0

        modified_seqs[decisions == 1] = 0

        modified_seqs = modified_seqs.tolist()

        # batch             seq_pos
        dim_0_index_insert, dim_1_index_insert = torch.where(decisions == 2)

        insert_seqs_corrector = insert_seqs[dim_0_index_insert, dim_1_index_insert].tolist()  # max_insert_size

        i = 0

        pre_dim_0 = 0

        k = 0

        while i < len(insert_seqs_corrector):

            j = 0

            if pre_dim_0 != dim_0_index_insert[i]:

                k = 0

                pre_dim_0 = dim_0_index_insert[i]

            while j < len(insert_seqs_corrector[i]):

                if insert_seqs_corrector[i][j] == self.n_items - 2:

                    break

                modified_seqs[dim_0_index_insert[i]].insert((dim_1_index_insert[i] + k), insert_seqs_corrector[i][j])

                j += 1

            k += j

            i += 1

        batch_size = len(modified_seqs)

        for i in range(batch_size):

            modified_seqs[i] = list(filter(lambda x: (x != 0 and x != self.n_items - 2 and x != self.n_items - 1), modified_seqs[i]))   # filter padding (and deleted items), EOS, mask token

        return modified_seqs
