import os
import pickle

from transformers import (
    T5ForConditionalGeneration,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    HammingDiversityLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    BeamSearchScorer,
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import BaseModelOutput
import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp

import torch.nn.functional as F

def align(teacher_emb, student_emb, temperature=3.0):
    teacher_probs = F.softmax(teacher_emb / temperature, dim=-1)
    student_probs = F.log_softmax(student_emb / temperature, dim=-1)
    return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

class Contrastive_Loss2(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau

    def forward(self, X, Y):
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = nn.functional.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (Y_loss + X_loss) / 2.0  # shape: (batch_size)
        return loss.mean()
    def cross_entropy(self, preds, targets, reduction='none'):

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

class Solomon(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def __create_sparse_bipartite_adjacency(self, records, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        n_nodes = self.user_num + self.item_num + 1
        row_idx = [pair['user'] for pair in records]
        col_idx = [pair['item'] for pair in records]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes), dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def create_user_user_sparse_adjacency(self, sim_user_file, self_connection=False):
      
        with open(sim_user_file, 'rb') as f:
            sim_user_data = pickle.load(f)  
        user_num, top_k = sim_user_data.shape
        row_idx = np.repeat(np.arange(user_num), top_k) + 1
        col_idx = sim_user_data.flatten() + 1
        weights = np.ones_like(row_idx, dtype=np.float32)
        adj_mat = sp.csr_matrix((weights, (row_idx, col_idx)), shape=(user_num + 1, user_num + 1), dtype=np.float32)
        if self_connection:
            adj_mat += sp.eye(user_num + 1, dtype=np.float32)
        return adj_mat

    def normalize_graph_mat(self, adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        rowsum[rowsum==0] = 1e-9
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_sparse_mat_to_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor(np.array([coo.row, coo.col]))
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape)

    def init_graph_embeddings(self, alpha, sigma, L,L_user, training_exp_data, user_num, item_num, dataset_dir, k):
        '''llm'''
        llm_item_emb = pickle.load(open(dataset_dir + 'pca_item_feature.pkl', "rb"))
        self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        self.item_emb.weight.requires_grad = False 
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int((llm_item_emb.shape[1] + self.shared.weight.size(1)) / 2)),
            nn.Linear(int((llm_item_emb.shape[1] + self.shared.weight.size(1)) / 2), self.shared.weight.size(1))
        )
        '''llm'''
        # pass
        self.alpha = alpha
        self.L = L
        self.L_user = L_user
        self.user_num, self.item_num = user_num, item_num
        self.user_embeddings = nn.Embedding(user_num+1, self.shared.weight.size(1), padding_idx=0)
        torch.nn.init.normal_(self.user_embeddings.weight, mean=0, std=sigma)
 

        ui_adj = self.__create_sparse_bipartite_adjacency(training_exp_data)
        norm_adj = self.normalize_graph_mat(ui_adj)
        self.sparse_norm_adj = self.convert_sparse_mat_to_tensor(norm_adj).cuda()

        sim_user_file = dataset_dir + "user_preference_{}.pkl".format(k)
        sparse_user_user_adj = self.create_user_user_sparse_adjacency(sim_user_file, self_connection=True)
        sparse_user_user_adj = self.normalize_graph_mat(sparse_user_user_adj)
        self.sparse_user_adj = self.convert_sparse_mat_to_tensor(sparse_user_user_adj).cuda()


        self.user_embs = nn.Embedding(user_num+1, self.shared.weight.size(1), padding_idx=0)
        torch.nn.init.normal_(self.user_embs.weight, mean=0, std=sigma)

        
        self.align = Contrastive_Loss2()
    def graph_convolution(self, num_layer):
        user_embeddings = self.user_embeddings.weight
        item_embeddings = self.item_emb.weight
        item_seq_emb = self.adapter(item_embeddings)
        ego_embeddings = torch.concat([user_embeddings, item_seq_emb], dim=0)

        all_embeddings = [ego_embeddings]
        for k in range(num_layer):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        # return torch.concat([self.whole_word_embeddings.weight, all_embeddings], dim=0)
        return all_embeddings
 
    def graph_convolution_llm(self, num_layer):
        user_embeddings = self.user_embs.weight
        ego_embeddings = user_embeddings
        all_embeddings = [ego_embeddings]
        for k in range(num_layer):
            ego_embeddings = torch.sparse.mm(self.sparse_user_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        # return torch.concat([self.whole_word_embeddings.weight, all_embeddings], dim=0)
        return all_embeddings


    def graph_convolution_embeddings(self, num_layer):
        user_embeddings = self.user_embeddings.weight
        item_embeddings = self.item_emb.weight
        item_seq_emb = self.adapter(item_embeddings)
        ego_embeddings = torch.concat([user_embeddings, item_seq_emb], dim=0)
        for k in range(num_layer):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
        return ego_embeddings

    def init_prompt(self, task_num, prompts_per_task, device):
        emsize = self.shared.weight.size(1)
        self.prompts_per_task = prompts_per_task
        self.model_device = device
        self.prompt_embeddings = nn.Embedding(task_num * prompts_per_task, emsize)
        self.whole_word_embeddings = nn.Embedding(self.config.n_positions, emsize)  # sequence length
        # initrange = 0.1
        # self.prompt_embeddings.weight.data.uniform_(-initrange, initrange)
        self.prompt_offset = torch.arange(prompts_per_task).to(self.model_device)

    def input_plus_whole_word(self, input_ids, whole_word_ids, whole_word_ids_users):
        text_emb = self.shared(input_ids)  # (batch_size, src_len, emsize)
        whole_word_emb = self.whole_word_embeddings(whole_word_ids)
        text_emb_plus = text_emb + whole_word_emb * self.alpha

        return text_emb_plus

    def input_plus_graph_aware_whole_word(self, input_ids, whole_word_ids):
        text_emb = self.shared(input_ids)  # (batch_size, src_len, emsize)
        graph_aware_whole_word = self.graph_convolution(self.L)
        user_emb = graph_aware_whole_word[:self.user_num + 1]
        graph_aware_whole_word = graph_aware_whole_word[whole_word_ids]
        text_emb_plus = text_emb + graph_aware_whole_word * self.alpha
        return text_emb_plus, user_emb

    def append_prompt(self, task_id, input_ids, whole_word_ids, attention_mask, users=None):
        # prompt
        batch_size = task_id.size(0)
 
        task_ids = (task_id * self.prompts_per_task).unsqueeze(1) + self.prompt_offset.repeat(batch_size, 1)
        prompt = self.prompt_embeddings(task_ids)  # (batch_size, prompts_per_task, input_size)

        if task_id[0] == 1:
            text_emb_plus = self.input_plus_whole_word(input_ids, whole_word_ids, None)
        else:
            text_emb_plus, user_emb = self.input_plus_graph_aware_whole_word(input_ids, whole_word_ids)
        input_emb = torch.cat([prompt, text_emb_plus], 1)  # (batch_size, src_total_len, emsize)
        # mask
        prompt_pad = torch.ones((batch_size, self.prompts_per_task), dtype=torch.int64).to(self.model_device)
        input_mask = torch.cat([prompt_pad, attention_mask], 1)  # (batch_size, src_total_len)

        if users is not None and task_id[0] != 1:
            ''' align_loss'''
            user_emb_llm = self.graph_convolution_llm(self.L_user)[users]
            align_loss = self.align(user_emb_llm, user_emb[users].detach())  # Calculate align loss

            ''' align_loss'''
            return input_emb, input_mask, align_loss

        if users is not None and task_id[0] == 1:
            ''' align_loss'''
            user_emb_llm = self.graph_convolution_llm(self.L_user)[users]
            user_embs_seq = torch.mean(self.whole_word_embeddings(whole_word_ids), dim=1)
            
            # Align Loss
            align_loss =self.align(user_emb_llm, user_embs_seq)  # Calculate align loss

            ''' align_loss'''
            return input_emb, input_mask, align_loss
        # prompt_user = user_emb_llm
        # prompt_user = prompt_user.unsqueeze(1)  # (batch_size, 1, input_size)
        # input_emb = torch.cat([prompt_user, input_emb], 1)  # (batch_size, src_total_len, emsize)
        # prompt_pad = torch.ones((batch_size, 1), dtype=torch.int64).to(self.model_device)
        # input_mask = torch.cat([prompt_pad, input_mask], 1)  # (batch_size, src_total_len)
    
        return input_emb, input_mask, None
    def forward(
        self,
        task_id=None,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        users=None,
    ):
        if encoder_outputs is None:
            if task_id is None:
                input_emb = self.input_plus_whole_word(input_ids, whole_word_ids)
            else:
                if users is not None:
                    input_emb, attention_mask, align_loss = self.append_prompt(task_id, input_ids, whole_word_ids, attention_mask,
                                                                   users)
                else:
                    input_emb, attention_mask, _ = self.append_prompt(task_id, input_ids, whole_word_ids, attention_mask,
                                                                   users)
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                #input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=input_emb,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if users is None:
            return super().forward(
                # input_ids=input_ids,
                # attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                # inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return super().forward(
                # input_ids=input_ids,
                # attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                # inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ), align_loss

    def beam_search(
        self,
        task_id=None,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        max_length=50,
        num_beams=20,
        num_beam_groups=1,
        early_stopping=True,
        min_length=1,
        diversity_penalty=0.0,
        repetition_penalty=1.0,
        num_return_sequences=20,
        bad_words_ids=None,
    ):
        # define decoder start token ids
        batch_size = input_ids.size(0)
        decoder_input_ids = torch.ones((num_beams * batch_size, 1), dtype=torch.int64).to(self.model_device)
        decoder_input_ids = decoder_input_ids * self.config.decoder_start_token_id

        # add encoder_outputs to model keyword arguments
        if task_id is None:
            input_emb = self.input_plus_whole_word(input_ids, whole_word_ids)
        else:
            input_emb, attention_mask, _  = self.append_prompt(task_id, input_ids, whole_word_ids, attention_mask)
        model_kwargs = {
            "encoder_outputs": self.encoder(
                attention_mask=attention_mask.repeat_interleave(num_beams, dim=0),
                inputs_embeds=input_emb.repeat_interleave(num_beams, dim=0),
                return_dict=True,
            )
        }

        # instantiate beam scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=self.model_device,
            num_beam_groups=num_beam_groups,
            num_beam_hyps_to_keep=num_return_sequences,
            do_early_stopping=early_stopping,
        )

        criteria = StoppingCriteriaList()
        criteria.append(MaxLengthCriteria(max_length=max_length))

        # instantiate logits processors
        logits_processor = LogitsProcessorList()
        logits_processor.append(MinLengthLogitsProcessor(min_length, eos_token_id=self.config.eos_token_id))
        if bad_words_ids is not None:
            logits_processor.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id=self.config.eos_token_id))

        if num_beam_groups == 1:
            return super().beam_search(
                decoder_input_ids,
                beam_scorer,
                stopping_criteria=criteria,
                logits_processor=logits_processor,
                **model_kwargs)
        else:
            if diversity_penalty > 0.0:
                logits_processor.append(
                    HammingDiversityLogitsProcessor(
                        diversity_penalty,
                        num_beams=num_beams,
                        num_beam_groups=num_beam_groups,
                    )
                )
            if repetition_penalty != 1.0:
                logits_processor.append(
                    RepetitionPenaltyLogitsProcessor(
                        penalty=repetition_penalty,
                    )
                )

            return super().group_beam_search(
                decoder_input_ids,
                beam_scorer,
                stopping_criteria=criteria,
                logits_processor=logits_processor,
                **model_kwargs)
