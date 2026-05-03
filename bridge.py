import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np

class StrengthenedMoralProbe(nn.Module):
    def __init__(self, embed_dim=384, moral_dim=4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        # Residual block now matches dimensions (512 -> 512)
        self.res_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, moral_dim)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.proj(x)
        z = z + self.res_block(z)   # Now dimensions match (512 + 512)
        logits = self.head(z)
        return torch.softmax(logits, dim=-1)


class MGEPlusBridge:
    def __init__(self):
        self.device = torch.device("cpu")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.probe = StrengthenedMoralProbe().to(self.device)
        self.alpha_U = np.array([0.40, 0.30, 0.20, 0.10], dtype=np.float32)
        self.lambda_p = 0.15

    def get_embedding(self, text):
        with torch.no_grad():
            emb = self.embedder.encode(text, convert_to_tensor=True)
        return emb.to(self.device).clone().detach()

    def compute_passion(self, text):
        emb = self.get_embedding(text)
        p = torch.norm(emb).item() / 20.0
        return max(0.0, min(p, 1.0))

    def text_to_moral_vector(self, text):
        emb = self.get_embedding(text).unsqueeze(0)
        with torch.no_grad():
            s = self.probe(emb).detach().cpu().numpy()[0]
        return s

    def passion_adjusted_vector(self, text):
        s = self.text_to_moral_vector(text)
        p = self.compute_passion(text)
        s_prime = s * (1 + self.lambda_p * p)
        s_prime = s_prime / (np.sum(s_prime) + 1e-12)
        return s_prime

    def xi_m(self, s):
        s = np.asarray(s, dtype=np.float64)
        mat = np.diag(s) - np.outer(s, s) + 1e-8 * np.eye(len(s))
        sign, logdet = np.linalg.slogdet(mat)
        if sign <= 0:
            return 12.0
        return -logdet

    def compute_dual_coherence(self, founder_text, investor_text):
        s_f = self.passion_adjusted_vector(founder_text)
        s_i = self.passion_adjusted_vector(investor_text)
        s_shared = 0.5 * (s_f + s_i)
        s_shared = s_shared / (np.sum(s_shared) + 1e-12)
        xi = self.xi_m(s_shared)
        coherence = 1.0 / (1.0 + xi * 0.7)
        score = int(round(100 * coherence))
        return {"score": score, "s_f": s_f, "s_i": s_i, "s_shared": s_shared, "xi": xi}

    def compute_gap(self, founder_text, investor_text, s_f, s_i):
        diff = s_f - s_i
        idx = int(np.argmax(np.abs(diff)))
        founder_higher = diff[idx] > 0

        gap_map = {
            0: ("Trust", 
                "The founder shows stronger relational trust than the investor currently feels comfortable extending.",
                "The investor has higher trust expectations than the founder has yet demonstrated."),
            1: ("Justice / Impact", 
                "The founder emphasizes broader societal impact more than the investor currently prioritizes.",
                "The investor expects stronger ethical framing and fairness considerations."),
            2: ("Coherence / Vision", 
                "The founder’s strategic thinking appears more developed than the investor’s current lens.",
                "The investor’s strategic framework is sharper than the founder’s current articulation."),
            3: ("Execution", 
                "The founder is moving faster than the investor is comfortable with.",
                "The investor needs more concrete execution evidence than the founder has shown.")
        }
        name, founder_text, investor_text = gap_map[idx]
        return idx, (founder_text if founder_higher else investor_text)

    def compute_improved_score(self, s_f, s_i, gap_idx):
        s_f_adj = s_f.copy()
        s_i_adj = s_i.copy()
        
        gap_size = abs(s_f[gap_idx] - s_i[gap_idx])
        step = 0.32 + 0.18 * gap_size
        step = min(step, 0.55)
        
        s_f_adj[gap_idx] += step * (self.alpha_U[gap_idx] - s_f[gap_idx])
        s_i_adj[gap_idx] += step * (self.alpha_U[gap_idx] - s_i[gap_idx])
        
        s_f_adj = s_f_adj / np.sum(s_f_adj)
        s_i_adj = s_i_adj / np.sum(s_i_adj)
        
        s_shared_new = 0.5 * (s_f_adj + s_i_adj)
        s_shared_new = s_shared_new / (np.sum(s_shared_new) + 1e-12)
        
        xi_new = self.xi_m(s_shared_new)
        coherence_new = 1.0 / (1.0 + xi_new * 0.65)
        return int(round(100 * coherence_new))

    def run_dual_analysis(self, founder_text, investor_text):
        if not founder_text.strip() or not investor_text.strip():
            return {
                "score": 0,
                "gap": "Please provide both testimonies.",
                "founder_adjustment": "",
                "investor_adjustment": "",
                "improved_score": 0,
                "explanation": ""
            }

        result = self.compute_dual_coherence(founder_text, investor_text)
        gap_idx, gap_text = self.compute_gap(founder_text, investor_text, result["s_f"], result["s_i"])
        
        founder_adj = "Show one concrete milestone with timeline and measurable outcome."
        investor_adj = "Define the minimum proof of execution needed to feel confident moving forward."
        
        improved = self.compute_improved_score(result["s_f"], result["s_i"], gap_idx)
        dimension_name = ["Trust", "Justice/Impact", "Coherence/Vision", "Execution"][gap_idx]

        return {
            "score": result["score"],
            "gap": gap_text,
            "founder_adjustment": founder_adj,
            "investor_adjustment": investor_adj,
            "improved_score": improved,
            "explanation": f"After targeted alignment on the {dimension_name} dimension."
        }