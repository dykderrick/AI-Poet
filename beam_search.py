# -*- coding: utf-8 -*-
# @Time    : 2019/11/23 2:14 下午
# @Author  : Yingke Ding
# @FileName: beam_search.py
# @Software: PyCharm

import torch
import torch.nn.functional as F


class BeamSearch(object):
    def __init__(self, model, context, tokenizer, genre, temperature=1, beam_size=0, mode=0, top_k=0, device='cpu'):
        self.temperature = temperature
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size  # beam_search的唯一参数
        self.top_k = top_k
        self.mode = mode  # mode为1是整首诗的beam_search, mode为2是单句话的beam_search
        self.genre = genre

        self.beam_seqs = [self.get_context_tensor(context) for i in range(self.beam_size)]  # beam_size个标题, 待喂给模型

    def beam_sequence(self):
        with torch.no_grad():
            # 第一步应先给标题找第一个字, 粘到每一个beam_seq上, 并得到初始概率分布
            top_values, top_indices = self.get_top_beam(self.beam_seqs[0])

            # 将top_k = beam_size个初始token粘贴到每一个beam_seq上, 使用unsqueeze(0).unsqueeze(0)可以增加两次维度
            for i in range(self.beam_size):
                self.beam_seqs[i] = torch.cat((self.beam_seqs[i], top_indices[i].unsqueeze(0).unsqueeze(0)), dim=1)

            # 将top_values归一化, 得到下次beam_search计算时的概率
            beam_probability = F.softmax(top_values, dim=-1)

            # beam_search从这里开始
            while True:
                beam_values = []  # 相当于是一个beam_size * beam_size的Tensor列, 存放每一次search的字概率
                beam_indexes = []  # 相当于是一个beam_size * beam_size的Tensor列, 存放每一次search的字索引

                # 对beam_size个文本进行投喂
                for index, beam_seq in enumerate(self.beam_seqs):
                    top_values, top_indices = self.get_top_beam(beam_seq)

                    top_values *= beam_probability[index]  # 更新条件概率值
                    beam_values.append(top_values)
                    beam_indexes.append(top_indices)

                # While-True循环结束的条件: 搜索到的最高概率字是[PAD]
                if top_indices[0].tolist() == 0:
                    break

                # beam_search核心算法: 首先用stack和view将所有的概率值叠在一起(stack可以堆叠, view改变维度),
                # 然后再搜索topk=beam_size, 得到新的最有可能值
                top_values_new, top_indices_new = torch.topk(torch.stack(beam_values).view(self.beam_size ** 2),
                                                             self.beam_size, dim=-1)

                beam_probability = F.softmax(top_values_new, dim=-1)  # 概率归一化

                # 通过概率值找到新的seqs
                next_tokens = []
                beam_seqs_new = []

                for top_index_new in top_indices_new:
                    next_tokens.append(
                        beam_indexes[int(top_index_new / self.beam_size)][top_index_new % self.beam_size])
                    beam_seq_new = torch.cat((self.beam_seqs[int(top_index_new / self.beam_size)],
                                              next_tokens[-1].unsqueeze(0).unsqueeze(0)), dim=1)
                    beam_seqs_new.append(beam_seq_new)

                # 整首诗的beam_search
                if self.mode == 1:
                    self.beam_seqs = beam_seqs_new  # 更新到beam_seqs中, 进入下一次循环

                # 单句话的beam_search
                elif self.mode == 2:
                    if next_tokens[0].tolist() == 3946:  # 遇到逗号, 单句话beam结束
                        top_value_seq, top_index_seq = torch.topk(beam_probability, 1, dim=-1)
                        seq_seleted = beam_seqs_new[top_index_seq.tolist()[0]]

                        # 更新self.beam_seqs, 使得所有beam_size个序列都为相同的文本
                        for i in range(self.beam_size):
                            self.beam_seqs[i] = seq_seleted

                        # 计算一次beam, 以得到新的概率分布
                        top_values, top_indices = self.get_top_beam(self.beam_seqs[0])

                        # 寻找下一个字, 并更新概率分布, 再循环
                        for i in range(self.beam_size):
                            self.beam_seqs[i] = torch.cat((self.beam_seqs[i], top_indices[i].unsqueeze(0).unsqueeze(0)),
                                                          dim=1)
                        beam_probability = F.softmax(top_values, dim=-1)

                    else:  # 单句话内部
                        self.beam_seqs = beam_seqs_new

        probs = beam_probability.tolist()
        return self.beam_seqs[probs.index(max(probs))]  # 返回最高概率seq

    def get_top_beam(self, beam_seq):
        """
        给定一个序列, 寻找最可能的下beam_size个字
        :param beam_seq: 输入序列
        :return: top_values, top_indices
        """
        inputs = {'input_ids': beam_seq}
        outputs = self.model(**inputs)
        next_token_logits = outputs[0][0, -1, :] / self.temperature
        filtered_logits = self.filtering(logits=next_token_logits, index=0)

        tokens_probability = F.softmax(filtered_logits, dim=-1)
        top_values, top_indices = torch.topk(tokens_probability, self.beam_size, dim=-1)

        return top_values, top_indices

    def filtering(self, logits, index, filter_value=-float('Inf')):
        assert logits.dim() == 1
        if self.top_k > 0:
            top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
            top_values, top_indices = torch.topk(logits, 30, dim=-1)
            generated = self.beam_seqs[index][0].tolist()
            generated = [index for index in generated if index != 0 and index != 3946]
            length = len(generated)
            # if generated[-1] != generated[-2]:
            # generated = generated[:-1]
            for index in generated:
                logits[index] = filter_value
        return logits

    def get_context_tensor(self, context):
        context = torch.tensor(context, dtype=torch.long, device=self.device)
        context = context.unsqueeze(0)
        return context
