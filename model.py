import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Pilosa(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, use_gc=True):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, use_gc=use_gc)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            
            beta1, beta2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']
            use_gc = group['use_gc']

            for p in group['params']:
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): continue
                    
                    params_with_grad.append(p)
                    grad = p.grad
                    
                    if use_gc and grad.dim() > 1:
                        grad.sub_(grad.mean(dim=list(range(1, grad.dim())), keepdim=True))
                    
                    grads.append(grad)
                    
                    state = self.state[p]
                    if len(state) == 0:
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avgs.append(state['exp_avg'])

            if not params_with_grad: continue

            if wd > 0:
                torch._foreach_mul_(params_with_grad, 1 - lr * wd)

            update = torch._foreach_mul(exp_avgs, beta1)

            torch._foreach_add_(update, grads, alpha=1 - beta1)
            torch._foreach_add_(params_with_grad, [u.sign() for u in update], alpha=-lr)
            torch._foreach_mul_(exp_avgs, beta2)
            torch._foreach_add_(exp_avgs, grads, alpha=1 - beta2)

        return loss
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x_f32 = x.float()
        rrms = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f32 * rrms * self.weight.float()).to(dtype=x.dtype)

def parallel_rnn_scan(x, log_decay, state=None):
    """
    x: (B, H, T, K)
    log_decay: (B, H, T)
    state: (B, H, K)
    """
    B, H, T, K = x.shape
    
    decay_cumsum = torch.cumsum(log_decay, dim=-1) 
    
    indices = torch.arange(T, device=x.device)
    mask = indices[:, None] >= indices[None, :]
    
    L = decay_cumsum.unsqueeze(-1) - decay_cumsum.unsqueeze(-2)
    L = L.masked_fill(~mask, -float('inf'))
    W = torch.exp(L)
    W = torch.nan_to_num(W, nan=0.0)
    
    y = W @ x
    
    if state is not None:
        decay_state = torch.exp(decay_cumsum).unsqueeze(-1)
        y = y + (state.unsqueeze(2) * decay_state)
        
    return y, y[:, :, -1, :]

class XylemMixer(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(dim)
        self.in_proj = nn.Linear(dim, dim * 3 + n_heads, bias=False)
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=4, groups=dim, padding=3)
        self.decay_param = nn.Parameter(torch.randn(n_heads) * 0.5 - 3.0)
        self.gn = nn.GroupNorm(n_heads, dim, eps=1e-5)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, state=None):
        B, T, D = x.shape
        H, K = self.n_heads, self.head_dim
        
        projected = self.in_proj(x)
        r, v, k, w = torch.split(projected, [D, D, D, H], dim=-1)
        k_in = k.permute(0, 2, 1) # (B, D, T)
        
        if state is not None and 'conv' in state:
            prev_k = state['conv']
            k_padded = torch.cat([prev_k, k_in], dim=-1)
        else:
            k_padded = F.pad(k_in, (3, 0))
            
        k_out = self.local_conv(k_padded)
        next_conv_state = k_padded[:, :, -3:].detach()
        k_out = k_out[:, :, :T].permute(0, 2, 1)
        
        v = v * F.silu(k_out)
        
        base_decay = self.decay_param.view(1, 1, H).float()
        w_in = w.view(B, T, H).float()
        
        decay_gate = torch.sigmoid(base_decay + w_in)
        log_decay = -8.0 * (1.0 - decay_gate) - 0.1
        
        log_decay_in = log_decay.permute(0, 2, 1)
        v_in = v.view(B, T, H, K).permute(0, 2, 1, 3).float() * self.scale
        rnn_prev = state['rnn'] if (state is not None and 'rnn' in state) else None
        
        with torch.amp.autocast('cuda', enabled=False):
            y_out, next_rnn_state = parallel_rnn_scan(v_in, log_decay_in, rnn_prev)
        
        # Reshape (B, H, T, K) -> (B, T, H, K) -> (B, T, D)
        y_out = y_out.permute(0, 2, 1, 3).reshape(B, T, D)
        r = F.silu(r).float()
        out = r * y_out
        
        # GroupNorm (B, D, T) -> (B, T, D)
        out = self.gn(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = out.to(dtype=x.dtype)

        next_state = {'conv': next_conv_state, 'rnn': next_rnn_state}
        return self.out_proj(out), next_state

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None: hidden_dim = int(dim * 2)
        
        self.w_in = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x_in = self.w_in(x)
        gate, val = x_in.chunk(2, dim=-1)
        return self.w_out(val * F.silu(gate))

class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.mixer = XylemMixer(dim, n_heads)
        self.ln2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x, state=None):
        attn_out, next_state = self.mixer(self.ln1(x), state)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, next_state

class XuyenChi(nn.Module):
    def __init__(self, vocab_size, dim=512, n_layers=6, n_heads=8):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
        self.ln_out = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        self.apply(self._init_weights)
        self._apply_deep_scale()
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

    def _apply_deep_scale(self):
        scale = 1.0 / math.sqrt(2 * self.n_layers)
        for name, p in self.named_parameters():
            if "out_proj.weight" in name or "w_out.weight" in name:
                with torch.no_grad():
                    p.mul_(scale)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        current_idx = idx
        state = None
        
        logits, _, state = self(current_idx, states=None)
        idx_next = current_idx[:, -1:]
        
        for _ in range(max_new_tokens):
            logits, _, state = self(idx_next, states=state)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

    def forward(self, idx, targets=None, states=None):
        # idx: (B, T)
        x = self.embedding(idx)
        
        next_states = []
        for i, layer in enumerate(self.layers):
            s = states[i] if states is not None else None
            x, new_s = layer(x, state=s)
            next_states.append(new_s)
            
        x = self.ln_out(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
        return logits, loss, next_states
if __name__ == "__main__":
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    
    model = XuyenChi(vocab_size=1000, dim=256, n_layers=4, n_heads=4).to(device)
    optimizer = Pilosa(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    print(f"Model created. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    x = torch.randint(0, 1000, (2, 64)).to(device)
    y = torch.randint(0, 1000, (2, 64)).to(device)
    
    print("\n--- Training Step ---")
    model.train()
    optimizer.zero_grad()
    logits, loss, _ = model(x, targets=y)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.4f}")
    
    print("\n--- Inference Step (Generate) ---")
    model.eval()
    
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
    
    t0 = time.time()
    generated = model.generate(prompt, max_new_tokens=50)
    dt = time.time() - t0
    
    print(f"Generated sequence length: {generated.shape[1]}")
    print(f"Speed: {50/dt:.1f} tokens/sec")
    print(f"Output IDs: {generated[0].tolist()}")