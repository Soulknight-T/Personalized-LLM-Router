import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticAttention(nn.Module):
    def __init__(self, in_channel: int, hidden_dim: int = 128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_channel, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.project(z) # [N, P, 1]
        beta = torch.softmax(w, dim=1)  # [N, P, 1]
        return (beta * z).sum(dim=1) # [N, Emb_Dim]

class HAN_GNN(nn.Module):
    def __init__(self, 
                 metapath_list: list[tuple[str, str, str]], 
                 in_dim: int, 
                 hidden_dim: int, 
                 out_dim: int, 
                 heads: int = 4, 
                 dropout: float = 0.1):
        super().__init__()
        self.metapath_list = metapath_list
        self.dropout = dropout
        
        from torch_geometric.nn import GATConv
        self.gat_layers = nn.ModuleList()
        for _ in metapath_list:
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
            )
        
        self.semantic_in_dim = hidden_dim * heads
        
        self.semantic_attn = SemanticAttention(self.semantic_in_dim, hidden_dim=128)
        
        self.out_proj = nn.Linear(self.semantic_in_dim, out_dim)

    def forward(self, x_dict: dict, metapath_adj_dict: dict) -> dict:
        """
        Args:
            x_dict: {'user': [N, 768], 'llm': [M, 768], ...}
            metapath_adj_dict: { ('llm','meta_LQU','user'): AdjMatrix, ... }
        """
        # We specifically want to update 'user' embeddings
        target_node_type = "user" 
        
        semantic_embeddings = []
        
        for i, metapath_tuple in enumerate(self.metapath_list):
            # metapath_tuple format: (Source_Type, Relation, Target_Type)
            # Example: ('llm', 'meta_LQU', 'user')
            src_type, _, dst_type = metapath_tuple
            
            if metapath_tuple not in metapath_adj_dict:
                continue
            
            # Retrieve the Sparse Adjacency Matrix
            adj = metapath_adj_dict[metapath_tuple]
            
            # [CRITICAL FIX] Bipartite GAT Input
            # We must pass (Source_Features, Target_Features) to GATConv
            # This tells PyG that indices in 'adj' row 0 refer to src_type
            # and indices in 'adj' row 1 refer to dst_type.
            x_src = x_dict[src_type]
            x_dst = x_dict[dst_type]
            
            # GATConv will aggregate info FROM x_src TO x_dst
            # Output shape will be [Num_Dst_Nodes, Hidden * Heads]
            z_i = self.gat_layers[i]((x_src, x_dst), adj) 
            z_i = F.elu(z_i)
            
            semantic_embeddings.append(z_i)
            
        if not semantic_embeddings:
            # Fallback if no paths exist
            target_x = x_dict[target_node_type]
            return {target_node_type: self.out_proj(torch.zeros_like(target_x).repeat(1, self.out_proj.out_features // target_x.shape[1]))}

        # Stack: [N_Users, Num_Paths, Hidden*Heads]
        z_stack = torch.stack(semantic_embeddings, dim=1)
        
        # Semantic Attention
        z_final = self.semantic_attn(z_stack)
        
        # Final Projection
        out = self.out_proj(z_final)
        
        new_x_dict = x_dict.copy()
        new_x_dict[target_node_type] = out
        return new_x_dict

class PreferencePredictor(nn.Module):
    def __init__(self, user_dim: int, query_dim: int, llm_dim: int, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
    
        self.user_proj = nn.Linear(user_dim, hidden_dim)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.llm_proj = nn.Linear(llm_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, user, query, llm):
        if user.dim() == 1: user = user.unsqueeze(0)
        if query.dim() == 1: query = query.unsqueeze(0)
        if llm.dim() == 1: llm = llm.unsqueeze(0)

        u = self.user_proj(user)     # [B, H]
        q = self.query_proj(query)   # [B, H]
        l = self.llm_proj(llm)       # [B, H]

        seq = torch.stack([u, q, l], dim=1)
        
        attn_out, _ = self.attn(seq, seq, seq)
        
        final_vec = attn_out[:, 2, :] 
        

        score = self.out(final_vec) # [B, 1]
        return score