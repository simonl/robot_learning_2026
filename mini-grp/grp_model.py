import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_patches_fast(images, cfg):
    from einops import rearrange
    batch_size, height, width, channels = images.shape
    patch_size = cfg.patch_size ## n_patches = 8

    patches = rearrange(images[:,:,:,:3], 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
    if channels > 3:
        ## History stacking in the channel dimension for observations only, not goal images.
        patches = rearrange(images, 'b (h p1) (w p2) (c hs) -> b (h w hs) (p1 p2 c)', p1 = patch_size, p2 = patch_size, hs=cfg.policy.obs_stacking) ## Stack the history in the channel dimension
    return patches


def calc_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B,T,C = x.shape
        # Compute attention scores
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Scaled dot-product attention
        wei = q @ k.transpose(-2,-1) * C**-0.5  # (B, T, T)
        # Apply mask if provided (for block attention)
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        with torch.profiler.record_function("Self-Attention"):
            out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd, mlp_ratio, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, mlp_ratio * n_embd),
            nn.ReLU(),
            nn.Linear(mlp_ratio * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, mlp_ratio, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, mlp_ratio, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class GRP(nn.Module):
    def __init__(self, cfg, mlp_ratio=4):
        super(GRP, self).__init__()
        self._cfg = cfg
        chars = cfg.dataset.chars_list
        cfg.vocab_size = len(chars)
        # 1) Patch embedding projection
        patch_dim = (cfg.patch_size ** 2) * 3  # For RGB images
        if cfg.policy.obs_stacking > 1:
            patch_dim = (cfg.patch_size ** 2) * 3  # Still 3 channels per patch
        self.patch_embedding = nn.Linear(patch_dim, cfg.n_embd)
        
        # 2) Token embeddings for language (if not using T5)
        if not cfg.dataset.encode_with_t5:
            self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        
        # 3) Special tokens: [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.n_embd))
        self.goal_img_token = nn.Parameter(torch.randn(1, 1, cfg.n_embd))
        
        # 4) Positional embeddings
        # Max sequence length: CLS + obs patches + goal patches + language tokens + pose
        n_patches = (cfg.image_shape[0] // cfg.patch_size) * (cfg.image_shape[1] // cfg.patch_size)
        if cfg.policy.obs_stacking > 1:
            n_patches *= cfg.policy.obs_stacking
        n_goal_patches = (cfg.image_shape[0] // cfg.patch_size) * (cfg.image_shape[1] // cfg.patch_size)
        max_seq_len = 1 + n_patches + n_goal_patches + cfg.max_block_size + 1  # +1 for pose if used
        self.position_embedding_table = nn.Embedding(max_seq_len, cfg.n_embd)
        
        # 5) Pose embedding (optional)
        if hasattr(cfg, 'use_pose') and cfg.use_pose:
            self.pose_embedding = nn.Linear(cfg.policy.pose_dim if hasattr(cfg.policy, 'pose_dim') else 7, cfg.n_embd)
        
        # 6) Transformer encoder blocks
        self.blocks = nn.Sequential(*[Block(cfg.n_embd, cfg.n_head, mlp_ratio, cfg.dropout) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # 7) Action prediction head (MLP)
        action_dim = cfg.policy.action_dim
        if hasattr(cfg.policy, 'action_stacking'):
            action_dim *= cfg.policy.action_stacking
        self.action_head = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, action_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, images, goals_txt, goal_imgs, targets=None, pose=None, mask_=False):
        n, c, h, w = images.shape
        obs_patches = get_patches_fast(images, self._cfg)
        patches_g = get_patches_fast(goal_imgs, self._cfg)
        if self._cfg.dataset.encode_with_t5:
            goals_e = goals_txt
            B, T, E = goals_txt.shape
        else:
            goals_e = self.token_embedding_table(goals_txt)
            B, E = goals_txt.shape
            T = self._cfg.max_block_size

        # 1) Map patches to embedding dimension
        tok_obs = self.patch_embedding(obs_patches)  # (B, n_patches, n_embd)
        tok_goal = self.patch_embedding(patches_g)   # (B, n_goal_patches, n_embd)
        
        # 2) Process language tokens
        if self._cfg.dataset.encode_with_t5:
            tok_lang = goals_e  # Already embedded from T5: (B, T, n_embd)
        else:
            tok_lang = goals_e  # Character embeddings: (B, T, n_embd)
        
        # 3) Process pose if available
        if pose is not None and hasattr(self, 'pose_embedding'):
            tok_pose = self.pose_embedding(pose)  # (B, pose_dim) -> (B, 1, n_embd)
            if len(tok_pose.shape) == 2:
                tok_pose = tok_pose.unsqueeze(1)
        else:
            tok_pose = None
        
        # 4) Add CLS and goal image tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, n_embd)
        goal_img_tokens = self.goal_img_token.expand(B, -1, -1)  # (B, 1, n_embd)
        
        # 5) Concatenate all tokens: [CLS, obs_patches, goal_img_token, goal_patches, lang_tokens, pose]
        token_list = [cls_tokens, tok_obs, goal_img_tokens, tok_goal, tok_lang]
        if tok_pose is not None:
            token_list.append(tok_pose)
        x = torch.cat(token_list, dim=1)  # (B, total_seq_len, n_embd)
        
        # 6) Add positional embeddings
        seq_len = x.shape[1]
        pos_emb = self.position_embedding_table(torch.arange(seq_len, device=x.device))  # (seq_len, n_embd)
        x = x + pos_emb.unsqueeze(0)  # (B, seq_len, n_embd)
        
        # 7) Create block attention mask
        # Allow: CLS attends to all, obs attends to obs+lang+goal, goal attends to goal+lang, lang attends within lang
        mask = torch.ones((seq_len, seq_len), device=x.device)
        if mask_ or self._cfg.use_block_mask:
            # CLS token can attend to everything (first row already all 1s)
            # Set up block diagonal structure for efficiency
            n_obs = tok_obs.shape[1]
            n_goal_token = 1
            n_goal_patches = tok_goal.shape[1]
            n_lang = tok_lang.shape[1]
            
            # Define block boundaries
            idx = 1  # After CLS
            obs_start, obs_end = idx, idx + n_obs
            idx = obs_end
            goal_token_idx = idx
            idx += n_goal_token
            goal_start, goal_end = idx, idx + n_goal_patches
            idx = goal_end
            lang_start, lang_end = idx, idx + n_lang
            
            # Observation patches attend to: themselves, language, goal token, goal patches
            mask[obs_start:obs_end, obs_start:obs_end] = 1  # obs to obs
            mask[obs_start:obs_end, goal_token_idx:goal_end] = 1  # obs to goal token + patches
            mask[obs_start:obs_end, lang_start:lang_end] = 1  # obs to lang
            
            # Goal patches attend to: themselves, language, goal token
            mask[goal_start:goal_end, goal_token_idx:goal_end] = 1  # goal to goal
            mask[goal_start:goal_end, lang_start:lang_end] = 1  # goal to lang
            
            # Language tokens attend to themselves
            mask[lang_start:lang_end, lang_start:lang_end] = 1
        
        mask = mask.unsqueeze(0)  # (1, seq_len, seq_len)
        
        # 8) Pass through transformer blocks
        x = self.blocks(x)  # (B, seq_len, n_embd)
        x = self.ln_f(x)    # (B, seq_len, n_embd)
        
        # 9) Extract CLS token for action prediction
        cls_output = x[:, 0, :]  # (B, n_embd)
        
        # 10) Predict actions
        out = self.action_head(cls_output)  # (B, action_dim)
        
        # 11) Compute loss
        loss = None
        if targets is not None:
            loss = F.mse_loss(out, targets)
        
        return (out, loss)
    
    def resize_image(self, image):
        """
        Docstring for resize_image
        
        :param self: Description
        :param image: Description
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        import cv2
        import numpy as _np
        img = _np.array(image, dtype=_np.float32)
        img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        return img

    def normalize_state(self, image):
        """
        Docstring for preprocess_state
        
        :param self: Description
        :param image: Description
        self._encode_state = lambda af:   ((af/(255.0)*2.0)-1.0) # encoder: take a float, output an integer
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        # img = _np.array(image, dtype=_np.float32)
        # img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        enc = ((image / 255.0) * 2.0) - 1.0
        # t = _torch.tensor(enc, dtype=_torch.float32, device=self._cfg.device)
        return enc
    
    def preprocess_state(self, image):
        img = self.resize_image(image)
        img = self.normalize_state(img)
        return img

    def preprocess_goal_image(self, image):
        return self.preprocess_state(image)

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            # Tokenize and encode the text goal using T5
            input_ids = tokenizer(goal, return_tensors="pt", padding="max_length", 
                                 max_length=self._cfg.max_block_size, truncation=True).input_ids.to(self._cfg.device)
            with _torch.no_grad():
                goal_embedding = text_model.encoder(input_ids).last_hidden_state  # (1, seq_len, n_embd)
            return goal_embedding
        else:
            pad = " " * self._cfg.max_block_size
            goal_ = goal[:self._cfg.max_block_size] + pad[len(goal):self._cfg.max_block_size]
            try:
                stoi = {c: i for i, c in enumerate(self._cfg.dataset.chars_list)}
                ids = [stoi.get(c, 0) for c in goal_]
            except Exception:
                ids = [0] * self._cfg.max_block_size
            return _torch.tensor(_np.expand_dims(_np.array(ids, dtype=_np.int64), axis=0), dtype=_torch.long, device=self._cfg.device)

    def process_text_embedding_for_buffer(self, goal, tokenizer=None, text_model=None):
        """
        Process text goal embedding for storing in the circular buffer.
        Returns a numpy array of shape (max_block_size, n_embd) without batch dimension.
        """
        import numpy as _np
        if tokenizer is None or text_model is None:
            raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
        
        goal_ = _np.zeros((self._cfg.max_block_size, self._cfg.n_embd), dtype=_np.float32)
        input_ids = tokenizer(goal, return_tensors="pt").input_ids
        goal_t = text_model.encoder(input_ids).last_hidden_state.detach().cpu().numpy()
        goal_[:len(goal_t[0]), :] = goal_t[0][:self._cfg.max_block_size]
        return goal_

    def decode_action(self, action_tensor):
        
        """
        Docstring for decode_action
        
        :param self: Description
        :param action_tensor: Description
        self._decode_action = lambda binN: (binN * action_std) + action_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        ## The action tensor is of shape (batch_size, action_dim * action_stacking) so we need to repeat the mean and std per action stacking
        action_mean = _torch.tensor(np.repeat(self._cfg.env.action_mean, self._cfg.policy.action_stacking), dtype=action_tensor.dtype, device=action_tensor.device)
        action_std = _torch.tensor(np.repeat(self._cfg.env.action_std, self._cfg.policy.action_stacking), dtype=action_tensor.dtype, device=action_tensor.device)
        return (action_tensor * action_std) + action_mean
    
    def encode_action(self, action_float):
        """
        Docstring for encode_action
        
        :param self: Description
        :param action_float: Description
        self._encode_action = lambda af:   (af - action_mean)/(action_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        action_mean = _torch.tensor(self._cfg.env.action_mean, dtype=action_float.dtype, device=action_float.device)
        action_std = _torch.tensor(self._cfg.env.action_std, dtype=action_float.dtype, device=action_float.device)
        return (action_float - action_mean) / action_std


@torch.no_grad()
def estimate_loss(model, dataset):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_pose, x_goal, x_goal_img, Y = dataset.get_batch_grp(split, model._cfg, model._cfg.batch_size)
            logits, loss = model(X, x_goal, x_goal_img, Y, pose=x_pose)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
