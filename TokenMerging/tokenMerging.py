import torch

class TokenMerging(torch.nn.Module):

    def __init__(self, r):
        """
        Intializes the TokenMerging layer with specifed number of tokens to merge. 
        Args:
            r (int): The number of tokens to merge at each iteration
        """
        super().__init__()
        self.r = r
    
    def forward(self, x):
        """
        Defines the computation performed at each call of the token merger
        Parameters:
            x (torch.Tensor): The input tensor with size [batch_size, seq_len, d_model]
        
        Returns:
            torch.Tensor: The output tensor with the same shape as the input tensor, after
                          token merging has occurred. 
        """
        out = None
        
        # First, ensure that r is not more than 50% of the total number of tokens in x
        r = min(self.r, x.shape[1] // 2)

        # Return x if number of tokens to merge is <= 0
        if r <= 0:
            return x
        
        with torch.no_grad():
            # Separate tokens in x into two groups: A and B
            a = x[:, ::2, :]
            b = x[:, 1::2, :]

            scores = a @ b.transpose(-1, -2)

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

            # merge 
            n, t1, c = a.shape
            unm = a.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            a = a.gather(dim=-2, index=src_idx.expand(n, r, c))
            b = b.scatter_reduce(-2, dst_idx.expand(n, r, c), a, reduce="mean")

            out = torch.cat([unm, b], dim=1)

        return out