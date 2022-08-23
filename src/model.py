import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KGIC(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(KGIC, self).__init__()
        self._parse_args(args, n_entity, n_relation)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.attention = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.Sigmoid(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
                )
        # parameter for contrastive learning
        self.ssl_temp = 0.2      # 0.1  movie0.5
        self.ssl_reg = 1e-6      # 1e-6 movie-7
        self.ssl_reg_inter = 1e-6  # 1e-6 movie-7
        self._init_weight()

    def forward(
        self,
        items: torch.LongTensor,
        user_triple_set: list,
        item_triple_set: list,
        user_potential_triple_set: list,
        item_origin_triple_set: list,
    ):

        user_embeddings = []
        # [batch_size, triple_set_size, dim]
        user_emb_0 = self.entity_emb(user_triple_set[0][0])
        # [batch_size, dim]
        user_intial_embedding = user_emb_0.mean(dim=1)
        user_embeddings.append(user_intial_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(user_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(user_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_triple_set[2][i])
            # [batch_size, dim]
            user_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            user_embeddings.append(user_emb_i)


        item_embeddings = []
        # [batch size, dim]
        item_emb_origin = self.entity_emb(items)
        # item_embeddings.append(item_emb_origin)
        item_emb_0 = self.entity_emb(item_triple_set[0][0])
        item_intial_embedding = item_emb_origin
        item_embeddings.append(item_intial_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(item_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(item_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_triple_set[2][i])
            # [batch_size, dim]
            item_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            item_embeddings.append(item_emb_i)


        user_potential_embeddings = []
        user_potential_embeddings_0 = self.entity_emb(user_potential_triple_set[0][0])
        user_intial_potential_embedding = user_potential_embeddings_0.mean(dim=1)
        user_potential_embeddings.append(user_intial_potential_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(user_potential_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(user_potential_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_potential_triple_set[2][i])
            # [batch_size, dim]
            user_potential_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            user_potential_embeddings.append(user_potential_emb_i)


        item_origin_embeddings = []
        item_origin_embeddings_0 = self.entity_emb(item_origin_triple_set[0][0])
        item_intial_origin_embedding = item_origin_embeddings_0.mean(dim=1)
        item_origin_embeddings.append(item_intial_origin_embedding)
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(item_origin_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(item_origin_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_origin_triple_set[2][i])
            # [batch_size, dim]
            item_origin_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            item_origin_embeddings.append(item_origin_emb_i)

        if self.n_layer > 0 and (self.agg == 'sum' or self.agg == 'pool'):
             # [batch_size, triple_set_size, dim]
            item_emb_0 = self.entity_emb(item_triple_set[0][0])
            # [batch_size, dim]
            item_embeddings.append(item_emb_0.mean(dim=1))
            
        scores = self.predict(user_embeddings, item_embeddings, user_potential_embeddings, item_origin_embeddings)
        return scores

    def predict(self, user_embeddings, item_embeddings, user_potential_embeddings, item_origin_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
        e_p_u = user_potential_embeddings[0]
        e_o_v = item_origin_embeddings[0]
        loss = 0
        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            # intra-level contrastive learning
            loss += self.ssl_layer_loss(user_embeddings, user_potential_embeddings)
            loss += self.ssl_layer_loss(item_embeddings, item_origin_embeddings)

            # inter-level contrastive learning
            for i in range(0,  len(user_embeddings)):
                loss += self.ssl_layer_loss_inter(user_embeddings[i],  user_potential_embeddings[i], user_embeddings, user_potential_embeddings)
                loss += self.ssl_layer_loss_inter(item_embeddings[i], item_origin_embeddings[i], item_embeddings, item_origin_embeddings)

            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u), dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v), dim=-1)
            for i in range(1, len(user_potential_embeddings)):
                e_p_u = torch.cat((user_potential_embeddings[i], e_p_u), dim=-1)
            for i in range(1, len(item_origin_embeddings)):
                e_o_v = torch.cat((item_origin_embeddings[i], e_o_v), dim=-1)

            e_u = torch.cat((e_u, e_p_u), dim=-1)
            e_v = torch.cat((e_o_v, e_v), dim=-1)
            
        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores, loss

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg

    def _init_weight(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        # [batch_size, triple_set_size]
        att_weights = self.attention(torch.cat((h_emb, r_emb), dim=-1)).squeeze(-1)
        att_weights_norm = F.softmax(att_weights, dim=-1)
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        emb_i = emb_i.sum(dim=1)
        return emb_i

    def ssl_layer_loss(self, user_embedding, user_embedding_h):
        # pos and neg for the first graph
        current_user_embeddings = user_embedding[0]
        previous_user_embeddings = user_embedding[1]
        if self.n_layer > 1:
            for i in range(2, self.n_layer):
                previous_user_embeddings = torch.cat((previous_user_embeddings, user_embedding[i]), dim=0)
        previous_user_embeddings_all = self.entity_emb.weight

        # pos and neg for the second graph
        current_item_embeddings = user_embedding_h[0]
        previous_item_embeddings = user_embedding_h[1]
        if self.n_layer > 1:
            for i in range(2, self.n_layer):
                previous_item_embeddings = torch.cat((previous_item_embeddings, user_embedding_h[i]), dim=0)
        previous_item_embeddings_all = self.entity_emb.weight

        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.matmul(norm_user_emb1, norm_user_emb2.transpose(0, 1))
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp).sum(dim=1)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.matmul(norm_item_emb1, norm_item_emb2.transpose(0, 1))
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp).sum(dim=1)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item)

        return ssl_loss

    def ssl_layer_loss_inter(self,current_user_embeddings, current_item_embeddings, user_embedding, user_embedding_h):
        previous_user_embeddings_all = user_embedding_h[0]
        previous_item_embeddings_all = user_embedding[0]
        for i in range(1, len(user_embedding_h)):
            previous_user_embeddings_all = torch.cat((user_embedding_h[i], previous_user_embeddings_all), dim=0)
        for i in range(1, len(user_embedding)):
            previous_item_embeddings_all = torch.cat((user_embedding[i], previous_item_embeddings_all), dim=0)

        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(current_item_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        norm_item_emb1 = norm_user_emb2
        norm_item_emb2 = norm_user_emb1
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg_inter * (ssl_loss_user + ssl_loss_item)

        return ssl_loss

    # def sim(self, z1: torch.Tensor, z2: torch.Tensor):
    #     z1 = F.normalize(z1)
    #     z2 = F.normalize(z2)
    #     return torch.mm(z1, z2.t())
    #
    # def CL_loss(self, A_embedding, B_embedding):
    #     # first calculate the sim rec
    #     tau = 0.6  # default = 0.8
    #     f = lambda x: torch.exp(x / tau)
    #     # A_embedding = self.fc1(A_embedding)
    #     # B_embedding = self.fc1(B_embedding)
    #     refl_sim = f(self.sim(A_embedding, A_embedding))
    #     between_sim = f(self.sim(A_embedding, B_embedding))
    #
    #     loss_1 = -torch.log(
    #         between_sim.diag()
    #         / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    #     refl_sim_1 = f(self.sim(B_embedding, B_embedding))
    #     between_sim_1 = f(self.sim(B_embedding, A_embedding))
    #     loss_2 = -torch.log(
    #         between_sim_1.diag()
    #         / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
    #     ret = (loss_1 + loss_2) * 0.5
    #     # ret = loss_1
    #     ret = ret.mean()
    #     return ret