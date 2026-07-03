"""
Robust Explainer: robust causal-inference explainer (Coca-enhanced)
====================================================================
Based on Coca (ICSE'24) dual-view causal inference, this module adds adversarially robust training:
min-max optimization keeps the generated explanation mask stable even under worst-case perturbations
delta on node features.

Core idea (compared with Coca):
  - Coca optimizes only the mask so that factual/counterfactual predictions satisfy the objective
    (min over mask).
  - Before each outer mask update, Robust first solves for a worst-case feature perturbation delta*
    in the inner loop (max over delta), making the explained sample fail as much as possible under
    that perturbation; then it makes the mask preserve the explanation on (x + delta*).
  -> The resulting explanations are more stable under small out-of-distribution shifts and are
     harder for the attacker to bypass.

Mathematical form:
    min_{m_v, m_e}  L_coca(x, m_v, m_e)
                  + lambda_adv * max_{||delta||_inf <= eps}  L_coca(x + delta, m_v, m_e)
                  + regularization terms

Compatibility notes:
  - Fully compatible with the CocaExplainer interface (same explain(data, label) call).
  - delta perturbs only node features x, not the graph structure edge_index.
  - The inner PGD loop shares the same mask for K ascent steps, while the outer loop uses Adam descent.
  - adv_warmup runs Coca-only training for the first few epochs to avoid cold-start instability.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data

from explain.coca_explainer import CocaExplainer


# ══════════════════════════════════════════════════════════════════════
# Core class: Robust Explainer
# ══════════════════════════════════════════════════════════════════════

class RobustExplainer(CocaExplainer):
    """
    Explainer with Coca + PGD adversarial robust training.

    Inherits from CocaExplainer and reuses infrastructure such as _masked_forward,
    _continuity_regularization, and _empty_result. Only the main explain() loop is
    overridden to add the inner PGD loop.

    Usage:
        explainer = RobustExplainer(model, device='cuda',
                                    eps=0.05, pgd_steps=5, adv_loss_weight=1.0)
        result = explainer.explain(data, predicted_label=1)
    """

    def __init__(
        self,
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        alpha=0.5,
        lr=0.01,
        epochs=300,
        sparsity_coeff_feat=0.01,
        sparsity_coeff_edge=0.005,
        top_k=5,
        # ── Adversarial training hyperparameters ──
        eps=0.05,                # PGD perturbation radius (L-infinity), relative to node-feature scale
        pgd_steps=5,             # Number of inner PGD ascent steps
        pgd_step_size=None,      # Step size; None -> automatically set to eps / pgd_steps * 2.5
        adv_loss_weight=1.0,     # Outer adversarial loss weight lambda_adv
        adv_warmup=50,           # Number of initial epochs without adversarial training (stabilizes the mask first)
    ):
        super().__init__(
            model=model,
            device=device,
            alpha=alpha,
            lr=lr,
            epochs=epochs,
            sparsity_coeff_feat=sparsity_coeff_feat,
            sparsity_coeff_edge=sparsity_coeff_edge,
            top_k=top_k,
        )
        self.eps = eps
        self.pgd_steps = pgd_steps
        self.pgd_step_size = (
            pgd_step_size if pgd_step_size is not None
            else (eps / max(pgd_steps, 1)) * 2.5
        )
        self.adv_loss_weight = adv_loss_weight
        self.adv_warmup = adv_warmup

    # ─────────────────────────────────────────────────────────
    # Main entry point (overridden)
    # ─────────────────────────────────────────────────────────

    def explain(self, data, predicted_label):
        """
        Generate a robust explanation for a single sample.

        The signature and return format are fully consistent with the parent class.
        The result additionally includes 'adv_loss_history' for diagnosing whether
        the adversarial loss converges.
        """
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            empty = self._empty_result(num_nodes, 0)
            empty['adv_loss_history'] = []
            return empty

        y_hat = predicted_label
        y_hat_s = 1 - y_hat

        # ── Initialize masks ───────────────────────────────────
        feat_mask = (3.0 * torch.ones(num_nodes, device=self.device)
                    + 0.1 * torch.randn(num_nodes, device=self.device))
        feat_mask.requires_grad = True

        edge_mask = (3.0 * torch.ones(num_edges, device=self.device)
                    + 0.1 * torch.randn(num_edges, device=self.device))
        edge_mask.requires_grad = True

        optimizer = torch.optim.Adam([feat_mask, edge_mask], lr=self.lr)

        # ── Optimization loop ─────────────────────────────────────
        loss_history = []
        adv_loss_history = []
        best_loss = float('inf')
        best_feat_mask = None
        best_edge_mask = None

        with torch.backends.cudnn.flags(enabled=False):
            for epoch in range(self.epochs):
                optimizer.zero_grad()

                sigmoid_feat = torch.sigmoid(feat_mask)
                sigmoid_edge = torch.sigmoid(edge_mask)

                # ── (a) Coca loss under the clean view ──
                clean_loss, _ = self._coca_objective(
                    x, edge_index, sigmoid_feat, sigmoid_edge,
                    y_hat, y_hat_s,
                )

                # ── (b) Inner PGD solves for the worst-case perturbation delta* ──
                if epoch >= self.adv_warmup and self.adv_loss_weight > 0:
                    delta_star = self._inner_pgd(
                        x, edge_index,
                        sigmoid_feat.detach(), sigmoid_edge.detach(),
                        y_hat, y_hat_s,
                    )
                    # Recompute adversarial loss on (x + delta*) while keeping the mask differentiable
                    adv_loss, _ = self._coca_objective(
                        x + delta_star, edge_index,
                        sigmoid_feat, sigmoid_edge,
                        y_hat, y_hat_s,
                    )
                else:
                    adv_loss = torch.tensor(0.0, device=self.device)

                # ── (c) Sparsity + continuity regularization ──
                sparsity_loss = (
                    self.sparsity_coeff_feat * sigmoid_feat.sum()
                    + self.sparsity_coeff_edge * sigmoid_edge.sum()
                )
                continuity_loss = self._continuity_regularization(
                    sigmoid_feat, edge_index
                )

                # ── (d) Total loss ──
                loss = (
                    clean_loss
                    + self.adv_loss_weight * adv_loss
                    + sparsity_loss
                    + 0.001 * continuity_loss
                )

                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                loss_history.append(loss_val)
                adv_loss_history.append(
                    adv_loss.item() if isinstance(adv_loss, torch.Tensor) else 0.0
                )

                if loss_val < best_loss:
                    best_loss = loss_val
                    best_feat_mask = feat_mask.detach().clone()
                    best_edge_mask = edge_mask.detach().clone()

        # ── Post-processing ───────────────────────────────────────
        node_importance = torch.sigmoid(best_feat_mask).cpu().numpy()
        edge_importance = torch.sigmoid(best_edge_mask).cpu().numpy()

        k = min(self.top_k, num_nodes)
        top_nodes = np.argsort(node_importance)[::-1][:k].tolist()

        k_edge = min(self.top_k, num_edges)
        top_edges = np.argsort(edge_importance)[::-1][:k_edge].tolist()

        return {
            'node_importance': node_importance,
            'edge_importance': edge_importance,
            'top_nodes': top_nodes,
            'top_edges': top_edges,
            'feat_mask_raw': best_feat_mask.cpu().numpy(),
            'edge_mask_raw': best_edge_mask.cpu().numpy(),
            'loss_history': loss_history,
            'adv_loss_history': adv_loss_history,
        }

    # ─────────────────────────────────────────────────────────
    # Coca loss (factual + counterfactual, shared by clean and adversarial branches)
    # ─────────────────────────────────────────────────────────

    def _coca_objective(self, x, edge_index, sigmoid_feat, sigmoid_edge,
                        y_hat, y_hat_s):
        """
        Compute the Coca contrastive loss L_f + L_c for a single view
        (clean or perturbed).

        Returns:
            total: scalar tensor, alpha * L_f + (1 - alpha) * L_c
            (probs_fact, probs_cf): optional diagnostics
        """
        probs_fact = self._masked_forward(
            x, edge_index, sigmoid_feat, sigmoid_edge, mode='factual'
        )
        S_f = probs_fact[0, y_hat]

        probs_cf = self._masked_forward(
            x, edge_index, sigmoid_feat, sigmoid_edge, mode='counterfactual'
        )
        S_c_neg = -probs_cf[0, y_hat]

        L_f = F.relu(0.5 - S_f + probs_fact[0, y_hat_s])
        L_c = F.relu(0.5 - S_c_neg - probs_cf[0, y_hat_s])

        total = self.alpha * L_f + (1 - self.alpha) * L_c
        return total, (probs_fact, probs_cf)

    # ─────────────────────────────────────────────────────────
    # Inner loop: PGD solves for the worst-case feature perturbation delta*
    # ─────────────────────────────────────────────────────────

    def _inner_pgd(self, x, edge_index, sigmoid_feat, sigmoid_edge,
                   y_hat, y_hat_s):
        """
        Given the current mask, find the worst-case delta within radius eps
        along the ascent direction of the Coca loss.

        Notes:
            - delta is added only to node features and does not modify edge_index.
            - The mask is treated as a constant in the inner loop (already detached).
            - The ascent objective is to increase the Coca loss after the mask sees
              (x + delta), meaning the explanation fails under this perturbation.
            - Uses sign-gradient updates plus L-infinity projection (standard PGD-L-infinity).

        Returns:
            delta: detached tensor with shape (num_nodes, feat_dim)
        """
        delta = torch.zeros_like(x, requires_grad=True)
        # Random initialization (uniformly sampled within the eps ball) helps escape saddle points
        with torch.no_grad():
            delta.add_(torch.empty_like(x).uniform_(-self.eps, self.eps))

        for _ in range(self.pgd_steps):
            delta.requires_grad_(True)
            adv_loss, _ = self._coca_objective(
                x + delta, edge_index,
                sigmoid_feat, sigmoid_edge,
                y_hat, y_hat_s,
            )
            grad = torch.autograd.grad(
                adv_loss, delta, retain_graph=False, create_graph=False
            )[0]

            with torch.no_grad():
                # Take one ascent step
                delta = delta + self.pgd_step_size * grad.sign()
                # Project onto the L-infinity eps ball
                delta = torch.clamp(delta, min=-self.eps, max=self.eps)

        return delta.detach()


# ══════════════════════════════════════════════════════════════════════
# Batch explainer: same interface as BatchCocaExplainer
# ══════════════════════════════════════════════════════════════════════

class BatchRobustExplainer:
    """
    Run RobustExplainer in batch mode and generate explanations for all samples
    in the dataset that are correctly detected as vulnerable.

    The interface is consistent with BatchCocaExplainer and can be used as a direct replacement.
    """

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu",
                 **explainer_kwargs):
        self.explainer = RobustExplainer(model, device, **explainer_kwargs)
        self.device = device

    def explain_dataset(self, dataset, model_wrapper, only_vulnerable=True):
        results = []

        for idx, data in enumerate(dataset):
            pred_label = model_wrapper.predict_label(data)
            true_label = data.y.item() if hasattr(data, 'y') else None

            if only_vulnerable and pred_label != 1:
                continue
            if true_label is not None and pred_label != true_label:
                continue

            try:
                result = self.explainer.explain(data, pred_label)
                result['sample_index'] = idx
                result['predicted_label'] = pred_label
                result['true_label'] = true_label
                results.append(result)

                if len(results) % 50 == 0:
                    print(f"[Robust_Exp] Explained {len(results)} samples...")

            except Exception as e:
                print(f"[Robust_Exp] Failed to explain sample {idx}: {e}")
                continue

        print(f"[Robust_Exp] Done. Explained {len(results)} samples in total.")
        return results


# ══════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════

def demo_explain_robust():
    from src.model.wrapper import ModelWrapper
    from src.utils.gen_embedding import read_json

    wrapper = ModelWrapper(
        'reveal',
        '{MODEL_SAVE_PATH}/trained_model/'
        'ori-ds/reveal/reveal-cwe119/mod_94.59_92.5_96.77_93.61.ckpt'
    )

    explainer = RobustExplainer(
        model=wrapper.model,
        device='cuda',
        alpha=0.5,
        lr=0.01,
        epochs=300,
        top_k=5,
        # ── Adversarial training hyperparameters ──
        eps=0.05,
        pgd_steps=5,
        adv_loss_weight=1.0,
        adv_warmup=50,
    )

    data = read_json(
        '{HOME_PATH}/VulDS/BigVul/ori-embedding/vul/'
        '1_CVE-2013-1788_poppler_CWE-119_'
        'bbc2d8918fe234b7ef2c480eb148943922cc0959_1.json'
    )
    pred_label = wrapper.predict_label(data)
    print(f"Original sample predicted label: {pred_label}")

    result = explainer.explain(data, pred_label)
    print("top-k key nodes: " + str(result['top_nodes']))
    print(f"Adversarial loss convergence: first={result['adv_loss_history'][0]:.4f}, "
          f"last={result['adv_loss_history'][-1]:.4f}")


if __name__ == '__main__':
    demo_explain_robust()
