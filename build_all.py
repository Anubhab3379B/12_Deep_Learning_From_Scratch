"""Build Script: All 4 Deep Learning notebooks for P12"""
import json, os

def mc(ct, src):
    c = {"cell_type": ct, "metadata": {}, "source": src.split("\n")}
    if ct == "code": c["execution_count"] = None; c["outputs"] = []
    c["source"] = [l + "\n" if i < len(c["source"])-1 else l for i, l in enumerate(c["source"])]
    return c

def save_nb(cells, path):
    nb = {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11.0"}},"cells":cells}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Created: {os.path.basename(path)}")

BASE = r"D:\Completed Projects\12_Deep_Learning_From_Scratch"

# ============================================================
# NOTEBOOK 1: AUTOENCODER
# ============================================================
cells_ae = [
    mc("markdown", "# Autoencoder from Scratch (PyTorch)\n## Learn compression, feature extraction, and dimensionality reduction\n\nAn autoencoder learns to compress input into a lower-dimensional latent space (encoder) then reconstruct it (decoder). Used for: denoising, anomaly detection, feature learning.\n\n---"),
    mc("code", "# CELL 1: Setup\nimport subprocess, sys\nfor p in ['torch','torchvision','numpy','matplotlib','seaborn','tqdm']:\n    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', p])\n\nimport torch, torch.nn as nn, torch.optim as optim\nfrom torch.utils.data import DataLoader\nimport torchvision, torchvision.transforms as transforms\nimport numpy as np, matplotlib.pyplot as plt, os\nfrom pathlib import Path\nfrom tqdm.auto import tqdm\n\n# Config\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nSEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)\nOUTPUT_DIR = Path('outputs'); OUTPUT_DIR.mkdir(exist_ok=True)\nBATCH_SIZE = 128; EPOCHS = 30; LATENT_DIM = 32; LR = 1e-3\nprint(f'Device: {device}')"),
    mc("code", """# CELL 2: Load MNIST Dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)} | Image shape: 28x28")"""),
    mc("code", """# CELL 3: Autoencoder Architecture
class Autoencoder(nn.Module):
    \"\"\"
    Convolutional Autoencoder for MNIST.
    Encoder: Conv2d layers compress 28x28 -> latent vector
    Decoder: ConvTranspose2d layers reconstruct latent -> 28x28
    \"\"\"
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder: 1x28x28 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # -> 32x14x14
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # -> 64x7x7
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Flatten(),                               # -> 64*7*7 = 3136
            nn.Linear(64 * 7 * 7, latent_dim),          # -> latent_dim
            nn.ReLU(True)
        )
        # Decoder: latent_dim -> 1x28x28
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7), nn.ReLU(True),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        z = self.encoder(x)   # Encode
        x_recon = self.decoder(z)  # Decode
        return x_recon, z

model = Autoencoder(LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")"""),
    mc("code", """# CELL 4: Training Loop
train_losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch, (images, _) in enumerate(train_loader):
        images = images.to(device)
        recon, z = model(images)
        loss = criterion(recon, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

plt.plot(train_losses, 'b-', linewidth=2)
plt.title('Autoencoder Training Loss'); plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.savefig(OUTPUT_DIR / 'ae_training.png', dpi=150); plt.show()"""),
    mc("code", """# CELL 5: Visualize Reconstructions
model.eval()
with torch.no_grad():
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)
    recon, latent = model(test_images)

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
    axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
    axes[0, i].axis('off'); axes[1, i].axis('off')
axes[0, 0].set_ylabel('Original'); axes[1, 0].set_ylabel('Recon')
plt.suptitle('Autoencoder: Original vs Reconstructed')
plt.savefig(OUTPUT_DIR / 'ae_reconstructions.png', dpi=150); plt.show()

# Save model
torch.save(model.state_dict(), OUTPUT_DIR / 'autoencoder.pt')
print(f"Model saved! Latent dim: {LATENT_DIM}")"""),
]
save_nb(cells_ae, os.path.join(BASE, "01_autoencoder.ipynb"))

# ============================================================
# NOTEBOOK 2: VAE
# ============================================================
cells_vae = [
    mc("markdown", "# Variational Autoencoder (VAE) from Scratch\n## Learn probabilistic modeling and generative AI\n\nVAE learns a **probability distribution** over the latent space, enabling:\n- Smooth interpolation between samples\n- Generating new realistic data\n- Disentangled feature learning\n\nKey innovation: Reparameterization trick for backprop through stochastic nodes.\n\n---"),
    mc("code", "# CELL 1: Setup\nimport torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F\nfrom torch.utils.data import DataLoader\nimport torchvision, torchvision.transforms as transforms\nimport numpy as np, matplotlib.pyplot as plt\nfrom pathlib import Path\nfrom tqdm.auto import tqdm\n\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nSEED = 42; torch.manual_seed(SEED)\nOUTPUT_DIR = Path('outputs'); OUTPUT_DIR.mkdir(exist_ok=True)\nBATCH_SIZE = 128; EPOCHS = 30; LATENT_DIM = 20; LR = 1e-3\n\ntransform = transforms.Compose([transforms.ToTensor()])\ntrain_data = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)\ntrain_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)\nprint(f'Device: {device} | Dataset: {len(train_data)}')"),
    mc("code", """# CELL 2: VAE Architecture
class VAE(nn.Module):
    \"\"\"
    Variational Autoencoder with reparameterization trick.
    
    Instead of encoding to a single point, we encode to
    mu (mean) and log_var (log variance) of a Gaussian distribution,
    then sample from it using: z = mu + std * epsilon
    \"\"\"
    def __init__(self, latent_dim=20):
        super().__init__()
        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64*7*7, latent_dim)      # Mean
        self.fc_logvar = nn.Linear(64*7*7, latent_dim)   # Log variance
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 64*7*7)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.enc_conv(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        \"\"\"Reparameterization trick: z = mu + std * epsilon\"\"\"
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from N(0,1)
        return mu + eps * std
    
    def decode(self, z):
        h = self.dec_fc(z).view(-1, 64, 7, 7)
        return self.dec_conv(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    \"\"\"VAE Loss = Reconstruction (BCE) + KL Divergence\"\"\"
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence: pushes latent distribution toward N(0,1)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

model = VAE(LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
print(f"VAE params: {sum(p.numel() for p in model.parameters()):,}")"""),
    mc("code", """# CELL 3: Train VAE
losses = []
for epoch in range(EPOCHS):
    model.train(); total_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        recon, mu, logvar = model(images)
        loss = vae_loss(recon, images, mu, logvar)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(train_data)
    losses.append(avg)
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg:.2f}")

plt.plot(losses, 'r-', linewidth=2)
plt.title('VAE Training Loss (ELBO)'); plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.savefig(OUTPUT_DIR / 'vae_training.png', dpi=150); plt.show()"""),
    mc("code", """# CELL 4: Generate New Images (Sampling from Latent Space)
model.eval()
with torch.no_grad():
    # Sample from standard normal
    z = torch.randn(64, LATENT_DIM).to(device)
    generated = model.decode(z).cpu()

fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.suptitle('VAE Generated Digits (Sampled from N(0,1))', fontsize=14)
plt.savefig(OUTPUT_DIR / 'vae_generated.png', dpi=150); plt.show()"""),
    mc("code", """# CELL 5: Latent Space Interpolation
model.eval()
with torch.no_grad():
    # Get two real images
    imgs, labels = next(iter(train_loader))
    mu1, _ = model.encode(imgs[0:1].to(device))
    mu2, _ = model.encode(imgs[1:2].to(device))
    
    # Interpolate in latent space
    n_steps = 10
    alphas = np.linspace(0, 1, n_steps)
    interp = torch.stack([mu1 * (1-a) + mu2 * a for a in alphas]).squeeze(1)
    decoded = model.decode(interp).cpu()

fig, axes = plt.subplots(1, n_steps, figsize=(15, 2))
for i, ax in enumerate(axes):
    ax.imshow(decoded[i].squeeze(), cmap='gray'); ax.axis('off')
plt.suptitle(f'Latent Space Interpolation: {labels[0].item()} -> {labels[1].item()}')
plt.savefig(OUTPUT_DIR / 'vae_interpolation.png', dpi=150); plt.show()

torch.save(model.state_dict(), OUTPUT_DIR / 'vae.pt')
print("VAE complete! Model saved.")"""),
]
save_nb(cells_vae, os.path.join(BASE, "02_vae.ipynb"))

# ============================================================
# NOTEBOOK 3: ATTENTION MECHANISM
# ============================================================
cells_attn = [
    mc("markdown", "# Attention Mechanism from Scratch\n## Self-Attention | Multi-Head Attention | Mini Transformer\n\nThe Transformer architecture (\"Attention Is All You Need\") revolutionized NLP and now powers GPT, BERT, and all modern LLMs.\n\n---"),
    mc("code", "# CELL 1: Setup\nimport torch, torch.nn as nn, torch.nn.functional as F\nimport numpy as np, matplotlib.pyplot as plt, math\nfrom pathlib import Path\n\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nSEED = 42; torch.manual_seed(SEED)\nOUTPUT_DIR = Path('outputs'); OUTPUT_DIR.mkdir(exist_ok=True)\nprint(f'Device: {device}')"),
    mc("code", """# CELL 2: Scaled Dot-Product Attention (Core Building Block)
def scaled_dot_product_attention(query, key, value, mask=None):
    \"\"\"
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Args:
        query: (batch, heads, seq_len, d_k)
        key:   (batch, heads, seq_len, d_k)
        value: (batch, heads, seq_len, d_v)
        mask:  optional mask for causal/padding attention
    Returns:
        output: (batch, heads, seq_len, d_v)
        attention_weights: (batch, heads, seq_len, seq_len)
    \"\"\"
    d_k = query.size(-1)
    # QK^T / sqrt(d_k) -- scaling prevents softmax saturation
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

# Demo
batch, seq_len, d_model = 2, 8, 64
q = k = v = torch.randn(batch, 1, seq_len, d_model)
out, attn = scaled_dot_product_attention(q, k, v)
print(f"Input shape: {q.shape}")
print(f"Output shape: {out.shape}")
print(f"Attention weights shape: {attn.shape}")"""),
    mc("code", """# CELL 3: Multi-Head Attention
class MultiHeadAttention(nn.Module):
    \"\"\"
    Multi-Head Attention allows the model to attend to information
    from different representation subspaces at different positions.
    
    Instead of one attention function with d_model dims, we project
    Q, K, V into h heads of d_k dims each, run attention in parallel,
    then concatenate and project back.
    \"\"\"
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)  # Query projection
        self.W_k = nn.Linear(d_model, d_model)  # Key projection
        self.W_v = nn.Linear(d_model, d_model)  # Value projection
        self.W_o = nn.Linear(d_model, d_model)  # Output projection
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape to (batch, heads, seq, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_out)
        return output, attn_weights

mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(2, 10, 64)  # (batch=2, seq=10, dim=64)
out, weights = mha(x, x, x)
print(f"MHA output: {out.shape} | Attention: {weights.shape}")"""),
    mc("code", """# CELL 4: Positional Encoding
class PositionalEncoding(nn.Module):
    \"\"\"Sinusoidal positional encoding (from 'Attention Is All You Need').\"\"\"
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Visualize
pe = PositionalEncoding(64)
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(pe.pe[0, :50, :].numpy(), aspect='auto', cmap='RdBu')
ax.set_xlabel('Dimension'); ax.set_ylabel('Position')
ax.set_title('Sinusoidal Positional Encoding'); plt.colorbar(im)
plt.savefig(OUTPUT_DIR / 'positional_encoding.png', dpi=150); plt.show()"""),
    mc("code", """# CELL 5: Mini Transformer Encoder Block
class TransformerBlock(nn.Module):
    \"\"\"Single transformer encoder block: MHA + FFN + LayerNorm + Residual.\"\"\"
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, attn_w = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        # Feed-forward with residual connection
        x = self.norm2(x + self.ffn(x))
        return x, attn_w

class MiniTransformer(nn.Module):
    \"\"\"Mini Transformer for sequence classification.\"\"\"
    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=256, n_layers=3, n_classes=2, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(n_layers)])
        self.classifier = nn.Linear(d_model, n_classes)
    
    def forward(self, x, mask=None):
        x = self.pos_enc(self.embedding(x))
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x, mask)
            attn_maps.append(attn)
        # Use [CLS]-like pooling (mean of sequence)
        pooled = x.mean(dim=1)
        return self.classifier(pooled), attn_maps

model = MiniTransformer(vocab_size=10000, n_classes=2).to(device)
dummy = torch.randint(0, 10000, (4, 32)).to(device)
logits, attn = model(dummy)
print(f"Logits: {logits.shape} | Attention maps: {len(attn)} layers")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
torch.save(model.state_dict(), OUTPUT_DIR / 'mini_transformer.pt')
print("Attention mechanism notebook complete!")"""),
]
save_nb(cells_attn, os.path.join(BASE, "03_attention_mechanism.ipynb"))

# ============================================================
# NOTEBOOK 4: GAN
# ============================================================
cells_gan = [
    mc("markdown", "# Generative Adversarial Network (GAN) from Scratch\n## DCGAN for Image Generation\n\nGANs train two networks in competition:\n- **Generator**: Creates fake images from random noise\n- **Discriminator**: Distinguishes real from fake\n\nThrough adversarial training, the generator learns to create increasingly realistic images.\n\n---"),
    mc("code", "# CELL 1: Setup\nimport torch, torch.nn as nn, torch.optim as optim\nfrom torch.utils.data import DataLoader\nimport torchvision, torchvision.transforms as transforms\nimport numpy as np, matplotlib.pyplot as plt\nfrom pathlib import Path\nfrom tqdm.auto import tqdm\n\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\ntorch.manual_seed(42)\nOUTPUT_DIR = Path('outputs'); OUTPUT_DIR.mkdir(exist_ok=True)\nBATCH_SIZE = 128; EPOCHS = 50; NZ = 100; NGF = 64; NDF = 64; LR = 0.0002\n\ntransform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),\n    transforms.Normalize([0.5], [0.5])])\ntrain_data = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)\nloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)\nprint(f'Device: {device} | Samples: {len(train_data)}')"),
    mc("code", """# CELL 2: Generator Network (DCGAN Architecture)
class Generator(nn.Module):
    \"\"\"
    DCGAN Generator: Maps random noise z -> image.
    Uses transposed convolutions to upsample:
    z (100) -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
    \"\"\"
    def __init__(self, nz=100, ngf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            # Input: z (nz x 1 x 1)
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),      # -> ngf*8 x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),      # -> ngf*4 x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),      # -> ngf*2 x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),         # -> ngf x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()                                    # -> nc x 64 x 64
        )
    
    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    \"\"\"
    DCGAN Discriminator: Maps image -> real/fake probability.
    Mirror of Generator: 64x64 -> 32x32 -> ... -> scalar
    \"\"\"
    def __init__(self, nc=1, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),                    # -> ndf x 32 x 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True), # -> ndf*2 x 16 x 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True), # -> ndf*4 x 8 x 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, True), # -> ndf*8 x 4 x 4
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()                                 # -> 1 x 1 x 1
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# Weight initialization (from DCGAN paper)
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

netG = Generator(NZ, NGF).to(device).apply(weights_init)
netD = Discriminator(1, NDF).to(device).apply(weights_init)
print(f"Generator params: {sum(p.numel() for p in netG.parameters()):,}")
print(f"Discriminator params: {sum(p.numel() for p in netD.parameters()):,}")"""),
    mc("code", """# CELL 3: GAN Training Loop
criterion = nn.BCELoss()
optimD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
optimG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
fixed_noise = torch.randn(64, NZ, 1, 1, device=device)

G_losses, D_losses = [], []
for epoch in range(EPOCHS):
    for real_imgs, _ in loader:
        real_imgs = real_imgs.to(device)
        b_size = real_imgs.size(0)
        real_label = torch.ones(b_size, device=device) * 0.9  # Label smoothing
        fake_label = torch.zeros(b_size, device=device)
        
        # === Train Discriminator ===
        netD.zero_grad()
        output_real = netD(real_imgs)
        lossD_real = criterion(output_real, real_label)
        
        noise = torch.randn(b_size, NZ, 1, 1, device=device)
        fake = netG(noise)
        output_fake = netD(fake.detach())
        lossD_fake = criterion(output_fake, fake_label)
        
        lossD = lossD_real + lossD_fake
        lossD.backward(); optimD.step()
        
        # === Train Generator ===
        netG.zero_grad()
        output = netD(fake)
        lossG = criterion(output, torch.ones(b_size, device=device))
        lossG.backward(); optimG.step()
    
    G_losses.append(lossG.item()); D_losses.append(lossD.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] D_loss: {lossD.item():.4f} G_loss: {lossG.item():.4f}")

plt.plot(G_losses, label='Generator'); plt.plot(D_losses, label='Discriminator')
plt.title('GAN Training Losses'); plt.legend()
plt.savefig(OUTPUT_DIR / 'gan_training.png', dpi=150); plt.show()"""),
    mc("code", """# CELL 4: Generate & Visualize Final Images
netG.eval()
with torch.no_grad():
    fake_images = netG(fixed_noise).cpu()

fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_images[i].squeeze() * 0.5 + 0.5, cmap='gray')
    ax.axis('off')
plt.suptitle('DCGAN Generated Digits (Epoch {})'.format(EPOCHS), fontsize=14)
plt.savefig(OUTPUT_DIR / 'gan_generated.png', dpi=150); plt.show()

# Save models
torch.save(netG.state_dict(), OUTPUT_DIR / 'generator.pt')
torch.save(netD.state_dict(), OUTPUT_DIR / 'discriminator.pt')
print("\\nGAN complete! All 4 deep learning models built.")
print("Models saved to outputs/")"""),
]
save_nb(cells_gan, os.path.join(BASE, "04_gan.ipynb"))

print("\nAll 4 P12 notebooks created!")
