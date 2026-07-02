"""
Coca Explainer: dual-view causal-reasoning explainer
================================================
Implemented based on Section 5.2 of the Coca (ICSE'24) paper.

Core idea:
  By jointly optimizing factual reasoning and counterfactual reasoning,
  generate an explanation subgraph that is both effective (covers true vulnerable statements) and concise (keeps the scope as small as possible).

  - Factual reasoning: the retained subgraph should preserve the original prediction → ensures effectiveness
  - Counterfactual reasoning: removing the subgraph should change the prediction → ensures conciseness

Adaptation notes:
  - Supports GNN models with the model(x, edge_index) signature
  - Node-feature masks are fully differentiable; edge masks use a differentiable approximation
  - Outputs node-level importance scores that can be directly mapped to source-code statements
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from copy import deepcopy
from src.model.wrapper import ModelWrapper
from src.utils.gen_embedding import read_json
from common.utils.gen_embedding import (
    src2embedding,
)


# ══════════════════════════════════════════════════════════════════════
# Core class: Coca Explainer
# ══════════════════════════════════════════════════════════════════════

class CocaExplainer:
    """
    Coca dual-view causal-reasoning explainer.

    For a sample detected as vulnerable, optimize learnable masks
    to find the nodes (statements) and edges (dependencies) most critical to the model prediction.

    Usage:
        explainer = CocaExplainer(model, device='cuda')
        result = explainer.explain(data, predicted_label=1)
        print(result['node_importance'])   # Importance score for each node
        print(result['top_nodes'])         # Top-k key node indices
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
    ):
        """
        Args:
            model:              GNN model instance (weights loaded), with forward signature model(x, edge_index)
            device:             compute device
            alpha:              trade-off coefficient between factual and counterfactual reasoning; larger values emphasize effectiveness
                                alpha=0.5 means balanced (paper default); alpha>0.5 emphasizes effectiveness, alpha<0.5 emphasizes conciseness
            lr:                 learning rate for mask optimization
            epochs:             number of optimization iterations
            sparsity_coeff_feat: sparsity regularization coefficient for the node-feature mask
            sparsity_coeff_edge: sparsity regularization coefficient for the edge mask
            top_k:              return the top-k most important nodes
        """
        self.model = model
        self.device = device
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.sparsity_coeff_feat = sparsity_coeff_feat
        self.sparsity_coeff_edge = sparsity_coeff_edge
        self.top_k = top_k

        # Freeze model parameters (optimize only masks; do not update the model)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    # ─────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────

    def explain(self, data, predicted_label):
        """
        Generate an explanation for a single sample.

        Args:
            data:             PyG Data object (x, edge_index, edge_attr, y)
            predicted_label:  model-predicted label for this sample (usually 1 = vulnerable)

        Returns:
            dict: {
                'node_importance':  np.ndarray (num_nodes,) importance score for each node (0~1),
                'edge_importance':  np.ndarray (num_edges,) importance score for each edge (0~1),
                'top_nodes':        list[int]  top-k key node indices,
                'top_edges':        list[int]  top-k key edge indices,
                'feat_mask_raw':    np.ndarray raw mask values before sigmoid,
                'edge_mask_raw':    np.ndarray raw mask values before sigmoid,
                'loss_history':     list[float] training loss curve,
            }
        """
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            return self._empty_result(num_nodes, 0)

        y_hat = predicted_label
        y_hat_s = 1 - y_hat  # Binary classification: the other label

        # Initialize to positive values; after sigmoid this is approximately 0.95, meaning "keep everything by default"
        # Add small noise to break symmetry
        feat_mask = (3.0 * torch.ones(num_nodes, device=self.device)
                    + 0.1 * torch.randn(num_nodes, device=self.device))
        feat_mask.requires_grad = True

        edge_mask = (3.0 * torch.ones(num_edges, device=self.device)
                    + 0.1 * torch.randn(num_edges, device=self.device))
        edge_mask.requires_grad = True

        optimizer = torch.optim.Adam([feat_mask, edge_mask], lr=self.lr)

        # ── Optimization loop ────────────────────────────────────
        loss_history = []
        best_loss = float('inf')
        best_feat_mask = None
        best_edge_mask = None

        with torch.backends.cudnn.flags(enabled=False):
            for epoch in range(self.epochs):
                optimizer.zero_grad()

                sigmoid_feat = torch.sigmoid(feat_mask)
                sigmoid_edge = torch.sigmoid(edge_mask)

                # ── Factual reasoning ──
                # The retained subgraph should preserve the original prediction
                probs_fact = self._masked_forward(
                    x, edge_index, sigmoid_feat, sigmoid_edge, mode='factual'
                )
                S_f = probs_fact[0, y_hat]

                # ── Counterfactual reasoning ──
                # Removing the subgraph should change the prediction
                probs_cf = self._masked_forward(
                    x, edge_index, sigmoid_feat, sigmoid_edge, mode='counterfactual'
                )
                S_c_neg = -probs_cf[0, y_hat]

                # ── Contrastive loss (Eq. 7) ──
                L_f = F.relu(0.5 - S_f + probs_fact[0, y_hat_s])
                L_c = F.relu(0.5 - S_c_neg - probs_cf[0, y_hat_s])

                # ── Sparsity regularization ──
                sparsity_loss = (
                    self.sparsity_coeff_feat * sigmoid_feat.sum()
                    + self.sparsity_coeff_edge * sigmoid_edge.sum()
                )

                # ── Continuity regularization (encourages adjacent nodes to have similar mask values) ──
                continuity_loss = self._continuity_regularization(
                    sigmoid_feat, edge_index
                )

                # ── Total loss (Eq. 8) ──
                loss = (
                    sparsity_loss
                    + self.alpha * L_f
                    + (1 - self.alpha) * L_c
                    + 0.001 * continuity_loss
                )

                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                loss_history.append(loss_val)

                if loss_val < best_loss:
                    best_loss = loss_val
                    best_feat_mask = feat_mask.detach().clone()
                    best_edge_mask = edge_mask.detach().clone()

        # ── Post-processing: generate final results ──────────────────────
        node_importance = torch.sigmoid(best_feat_mask).cpu().numpy()
        edge_importance = torch.sigmoid(best_edge_mask).cpu().numpy()

        # Top-k nodes
        k = min(self.top_k, num_nodes)
        top_nodes = np.argsort(node_importance)[::-1][:k].tolist()

        # Top-k edges
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
        }

    # ─────────────────────────────────────────────────────────
    # Masked forward pass
    # ─────────────────────────────────────────────────────────

    def _masked_forward(self, x, edge_index, sigmoid_feat, sigmoid_edge, mode):
        """
        Differentiable forward pass with masks.

        For the node mask: multiply it directly with node features (fully differentiable).
        For the edge mask: approximate it by scaling the source-node contribution on each edge (differentiable approximation).

        Args:
            x:            (num_nodes, feat_dim) original node features
            edge_index:   (2, num_edges) edge indices
            sigmoid_feat: (num_nodes,) node-mask values after sigmoid
            sigmoid_edge: (num_edges,) edge-mask values after sigmoid
            mode:         'factual' or 'counterfactual'

        Returns:
            probs: (1, num_classes) softmax probabilities
        """
        if mode == 'factual':
            # Factual reasoning: keep the parts selected by the mask
            node_weight = sigmoid_feat
            edge_weight = sigmoid_edge
        else:
            # Counterfactual reasoning: remove the parts selected by the mask
            node_weight = 1.0 - sigmoid_feat
            edge_weight = 1.0 - sigmoid_edge

        # ── Node-feature mask (fully differentiable) ──
        x_masked = x * node_weight.unsqueeze(-1)

        # ── Edge mask (differentiable approximation) ──
        # Strategy: aggregate edge weights onto nodes as an additional scaling factor
        # Rationale: if all incoming-edge weights of a node are low,
        #       the information received by that node is not important and should be further suppressed
        x_masked = self._apply_edge_mask_to_features(
            x_masked, edge_index, edge_weight
        )

        # ── Model forward pass (without no_grad, to keep it differentiable) ──
        logits = self.model(x_masked, edge_index)
        probs = torch.softmax(logits, dim=-1)

        return probs

    def _apply_edge_mask_to_features(self, x, edge_index, edge_weight):
        """
        Incorporate edge-mask information into node features.

        Because the model forward signature model(x, edge_index) does not support edge_weight,
        we approximate the effect of edge masks by modifying node features:

        For each node d, compute the weighted average w_d of its incoming-edge weights,
        then scale the node features to x[d] * w_d.

        This approximates the effect that "if edges pointing to d are removed, the information received by d is reduced".

        Args:
            x:           (num_nodes, feat_dim) features after applying the node mask
            edge_index:  (2, num_edges)
            edge_weight: (num_edges,) edge weights

        Returns:
            x_scaled: (num_nodes, feat_dim) scaled features
        """
        num_nodes = x.shape[0]
        src, dst = edge_index

        # Compute the sum of incoming-edge weights for each node
        weighted_degree = torch.zeros(num_nodes, device=x.device)
        weighted_degree.scatter_add_(0, dst, edge_weight)

        # Compute the number of incoming edges for each node
        degree = torch.zeros(num_nodes, device=x.device)
        degree.scatter_add_(0, dst, torch.ones_like(edge_weight))

        # Weighted average (avoid division by zero)
        degree = degree.clamp(min=1.0)
        node_scale = weighted_degree / degree  # (num_nodes,)

        # For nodes with no incoming edges (such as entry nodes), keep the original features
        no_incoming = (degree <= 1.0) & (weighted_degree == 0)
        node_scale[no_incoming] = 1.0

        # Scale node features
        x_scaled = x * node_scale.unsqueeze(-1)

        return x_scaled

    # ─────────────────────────────────────────────────────────
    # Regularization
    # ─────────────────────────────────────────────────────────

    def _continuity_regularization(self, sigmoid_feat, edge_index):
        """
        Continuity regularization: encourage adjacent nodes in the graph to have similar mask values.
        If node s is important, then node d that depends on s also tends to be important.
        This helps generate a coherent explanation subgraph rather than isolated nodes.
        """
        src, dst = edge_index
        diff = (sigmoid_feat[src] - sigmoid_feat[dst]) ** 2
        return diff.mean()

    # ─────────────────────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────────────────────

    def _empty_result(self, num_nodes, num_edges):
        """Default return value when the graph is empty or has no edges"""
        return {
            'node_importance': np.zeros(num_nodes),
            'edge_importance': np.zeros(num_edges),
            'top_nodes': [],
            'top_edges': [],
            'feat_mask_raw': np.zeros(num_nodes),
            'edge_mask_raw': np.zeros(num_edges),
            'loss_history': [],
        }


# ══════════════════════════════════════════════════════════════════════
# Batch explainer: generate explanations for all correctly detected samples in the dataset
# ══════════════════════════════════════════════════════════════════════

class BatchCocaExplainer:
    """
    Run Coca Explainer in batch mode and generate explanations for all samples correctly detected as vulnerable in the dataset.

    Usage:
        batch_explainer = BatchCocaExplainer(model, device='cuda')
        all_results = batch_explainer.explain_dataset(dataset, model_wrapper)
    """

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu", **explainer_kwargs):
        self.explainer = CocaExplainer(model, device, **explainer_kwargs)
        self.device = device

    def explain_dataset(self, dataset, model_wrapper, only_vulnerable=True):
        """
        Generate explanations for samples in a dataset in batch mode.

        Args:
            dataset:          list[Data] list of PyG Data objects
            model_wrapper:    ModelWrapper instance (used to get predicted labels)
            only_vulnerable:  whether to explain only samples predicted as vulnerable

        Returns:
            list[dict]: explanation result for each sample, including the original index
        """
        results = []

        for idx, data in enumerate(dataset):
            pred_label = model_wrapper.predict_label(data)
            true_label = data.y.item() if hasattr(data, 'y') else None

            # Explain only samples that the model predicts correctly and as vulnerable
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

                if (len(results)) % 50 == 0:
                    print(f"[Coca_Exp] Explained {len(results)} samples...")

            except Exception as e:
                print(f"[Coca_Exp] Failed to explain sample {idx}: {e}")
                continue

        print(f"[Coca_Exp] Done. Explained {len(results)} samples in total.")
        return results


# ══════════════════════════════════════════════════════════════════════
# Explanation-result analysis utilities
# ══════════════════════════════════════════════════════════════════════

class ExplanationAnalyzer:
    """Utility class for analyzing and visualizing explanation results."""

    @staticmethod
    def get_explanation_subgraph(data, result, threshold=0.5):
        """
        Extract an explanation subgraph from an explanation result.

        Args:
            data:      original PyG Data object
            result:    return value of CocaExplainer.explain()
            threshold: threshold for node/edge masks

        Returns:
            Data: PyG Data object for the explanation subgraph
        """
        node_mask = result['node_importance'] > threshold
        edge_mask = result['edge_importance'] > threshold

        # Retained nodes
        kept_nodes = np.where(node_mask)[0]
        if len(kept_nodes) == 0:
            # Threshold is too high; fall back to top-k
            kept_nodes = np.array(result['top_nodes'])

        # Create node mapping (old index → new index)
        node_mapping = {old: new for new, old in enumerate(kept_nodes)}

        # Filter edges: keep only edges whose endpoints are both in kept_nodes
        edge_index = data.edge_index.numpy()
        kept_edges = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src in node_mapping and dst in node_mapping:
                if edge_mask[i]:
                    kept_edges.append([node_mapping[src], node_mapping[dst]])

        # Build the subgraph
        x_sub = data.x[kept_nodes]
        if kept_edges:
            edge_index_sub = torch.tensor(kept_edges, dtype=torch.long).t()
        else:
            edge_index_sub = torch.zeros((2, 0), dtype=torch.long)

        sub_data = Data(x=x_sub, edge_index=edge_index_sub)
        sub_data.original_node_indices = kept_nodes.tolist()
        return sub_data

    @staticmethod
    def evaluate_explanation(result, ground_truth_nodes):
        """
        Evaluate explanation quality (VTP metrics).

        Args:
            result:              return value of CocaExplainer.explain()
            ground_truth_nodes:  list[int] ground-truth vulnerability-related node indices

        Returns:
            dict: {MSP, MSR, MIoU}
        """
        if not ground_truth_nodes:
            return {'MSP': 0.0, 'MSR': 0.0, 'MIoU': 0.0}

        explained = set(result['top_nodes'])
        ground_truth = set(ground_truth_nodes)

        intersection = explained & ground_truth
        union = explained | ground_truth

        sp = len(intersection) / len(explained) if explained else 0.0
        sr = len(intersection) / len(ground_truth) if ground_truth else 0.0
        iou = len(intersection) / len(union) if union else 0.0

        return {'MSP': sp, 'MSR': sr, 'MIoU': iou}

    @staticmethod
    def print_explanation(result, node_meta=None):
        """
        Print the explanation result.

        Args:
            result:    return value of explain()
            node_meta: optional node metadata list (including line numbers and code)
        """
        print("=" * 60)
        print("Coca Explanation Result")
        print("=" * 60)
        print(f"Top-{len(result['top_nodes'])} key nodes:")
        for rank, node_idx in enumerate(result['top_nodes']):
            score = result['node_importance'][node_idx]
            if node_meta and node_idx < len(node_meta):
                meta = node_meta[node_idx]
                line = meta.get('line_info', '?')
                code = meta.get('statement', '')
                print(f"  #{rank+1}  Node {node_idx} (score={score:.4f}) "
                      f"Line {line}: {code[:80]}")
            else:
                print(f"  #{rank+1}  Node {node_idx} (score={score:.4f})")
        print("-" * 60)


# ══════════════════════════════════════════════════════════════════════
# Usage example
# ══════════════════════════════════════════════════════════════════════

def demo_explain_coca():
    # 1. Load the model (your existing workflow)
    wrapper = ModelWrapper('reveal', '{HOME_PATH}/vul_explain/23_explain_eval_ISSTA/trained_model/ori-ds/reveal/reveal-cwe119/mod_94.59_92.5_96.77_93.61.ckpt')

    # 2. Create the explainer
    explainer = CocaExplainer(
        model=wrapper.model,        # Pass the model instance directly
        device='cuda',
        alpha=0.5,                  # Balance effectiveness and conciseness
        lr=0.01,
        epochs=300,
        top_k=5,
    )

    # 3. Generate an explanation for a single sample
    data = read_json('{HOME_PATH}/VulDS/BigVul/ori-embedding/vul/1_CVE-2013-1788_poppler_CWE-119_bbc2d8918fe234b7ef2c480eb148943922cc0959_1.json')               # A sample detected as vulnerable
    ori_code = """
static int default_filter_frame (AVFilterLink *inlink, AVFrame *wipe_side_data) {
    AVFilterContext *ctx = inlink->dst;
    FieldOrderContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    int h, plane, line_step, line_size, line;
    uint8_t *data;
    if (!wipe_side_data->interlaced_frame || wipe_side_data->top_field_first == s->dst_tff)
        return ff_filter_frame (outlink, wipe_side_data);
    av_dlog (ctx, "picture will move %s one line\n", s->dst_tff ? "up" : "down");
    h = wipe_side_data->height;
    for (plane = 0; plane < 4 & &wipe_side_data->data[plane]; plane++) {
        line_step = wipe_side_data->linesize[plane];
        line_size = s->line_size[plane];
        data = wipe_side_data->data[plane];
        if (s->dst_tff) {
            for (line = 0; line < h; line++) {
                if (1 + line < wipe_side_data->height) {
                    memcpy (data, data + line_step, line_size);
                }
                else {
                    memcpy (data, data - line_step - line_step, line_size);
                }
                data = data + line_step;
            }
        }
        else {
            data += (h - 1) * line_step;
            for (line = h - 1; line >= 0; line--) {
                if (line > 0) {
                    memcpy (data, data - line_step, line_size);
                }
                else {
                    memcpy (data, data + line_step + line_step, line_size);
                }
                data -= line_step;
            }
        }
    }
    wipe_side_data->top_field_first = s->dst_tff;
    return ff_filter_frame (outlink, wipe_side_data);
}
"""
    ori_data = src2embedding(src=ori_code,label=1)
    pred_label = wrapper.predict_label(data)  # Usually 1
    # print(f"Original subgraph: {data.x.shape[0]} nodes, "
    #       f"{data.edge_index.shape[1]} edges")
    print(f"Original sample predicted label: {pred_label}")

    # result = explainer.explain(data, pred_label)

    # # 4. Inspect results
    # print("Importance of each node from 0 to 1: " + str(result['node_importance']))
    # print("top-k key nodes: " + str(result['top_nodes']))

    # # 5. Extract the explanation subgraph
    # sub_data = ExplanationAnalyzer.get_explanation_subgraph(data, result)
    # print(f"Explanation subgraph: {sub_data.x.shape[0]} nodes, "
    #       f"{sub_data.edge_index.shape[1]} edges")

    # sub_pred_label = wrapper.predict_label(sub_data)
    # print(f"Explanation subgraph predicted label: {sub_pred_label}")


# test_file
# {HOME_PATH}/VulDS/BigVul/all-src/trans_vul/1_CVE-2013-4263_FFmpeg_CWE-119_e43a0a232dbf6d3c161823c2e07c52e76227a1bc_3_10.c
# {HOME_PATH}/VulDS/BigVul/all-src/vul/1_CVE-2012-2895_Chrome_CWE-119_baef1ffd73db183ca50c854e1779ed7f6e5100a8_5.c
if __name__ == '__main__':
    demo_explain_coca()
