"""
Robust Explainer: 鲁棒因果推理解释器（基于 Coca 增强）
================================================
在 Coca (ICSE'24) 双视角因果推理基础上，引入对抗鲁棒训练：
通过 min-max 优化，让生成的解释掩码对节点特征上的最坏扰动 δ 也保持稳定。

核心思想（与 Coca 对比）：
  - Coca 只优化 mask 让事实/反事实预测达标（min over mask）
  - Robust 在每个外层 mask 更新前，先内层求一个最坏特征扰动 δ*（max over δ）
    使被解释样本在该扰动下尽可能"翻车"，再让 mask 在 (x + δ*) 上仍能维持解释
  → 得到的解释对训练分布外的微小漂移更稳定，与攻击器结合时更难被绕过

数学形式:
    min_{m_v, m_e}  L_coca(x, m_v, m_e)
                  + λ_adv * max_{‖δ‖∞ ≤ ε}  L_coca(x + δ, m_v, m_e)
                  + 正则项

适配说明:
  - 与 CocaExplainer 接口完全兼容（同样的 explain(data, label) 调用）
  - δ 只扰动节点特征 x，不扰动图结构 edge_index
  - 内层 PGD 共享同一个 mask 进行 K 步上升，外层 Adam 下降
  - 提供 adv_warmup 让前若干 epoch 仅做 Coca 训练，避免冷启动震荡
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data

from explain.coca_explainer import CocaExplainer


# ══════════════════════════════════════════════════════════════════════
# 核心类：Robust Explainer
# ══════════════════════════════════════════════════════════════════════

class RobustExplainer(CocaExplainer):
    """
    Coca + PGD 对抗鲁棒训练的解释器。

    继承自 CocaExplainer，复用其 _masked_forward / _continuity_regularization
    / _empty_result 等基础设施，只重写 explain() 主循环以加入内层 PGD。

    用法:
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
        # ── 对抗训练相关超参 ──
        eps=0.05,                # PGD 扰动半径（L∞），相对节点特征量级
        pgd_steps=5,             # 内层 PGD 上升步数
        pgd_step_size=None,      # 单步步长，None → 自动设为 eps / pgd_steps * 2.5
        adv_loss_weight=1.0,     # 外层对抗损失权重 λ_adv
        adv_warmup=50,           # 前多少 epoch 不启用对抗训练（先稳定 mask）
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
    # 主入口（重写）
    # ─────────────────────────────────────────────────────────

    def explain(self, data, predicted_label):
        """
        生成单样本的鲁棒解释。

        与父类签名/返回完全一致；额外在 result 中带回 'adv_loss_history'，
        便于诊断对抗损失是否收敛。
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

        # ── 初始化掩码 ───────────────────────────────────
        feat_mask = (3.0 * torch.ones(num_nodes, device=self.device)
                    + 0.1 * torch.randn(num_nodes, device=self.device))
        feat_mask.requires_grad = True

        edge_mask = (3.0 * torch.ones(num_edges, device=self.device)
                    + 0.1 * torch.randn(num_edges, device=self.device))
        edge_mask.requires_grad = True

        optimizer = torch.optim.Adam([feat_mask, edge_mask], lr=self.lr)

        # ── 优化循环 ─────────────────────────────────────
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

                # ── (a) 干净视图下的 Coca 损失 ──
                clean_loss, _ = self._coca_objective(
                    x, edge_index, sigmoid_feat, sigmoid_edge,
                    y_hat, y_hat_s,
                )

                # ── (b) 内层 PGD 求最坏扰动 δ* ──
                if epoch >= self.adv_warmup and self.adv_loss_weight > 0:
                    delta_star = self._inner_pgd(
                        x, edge_index,
                        sigmoid_feat.detach(), sigmoid_edge.detach(),
                        y_hat, y_hat_s,
                    )
                    # 在 (x + δ*) 上重算对抗损失（mask 保持可微）
                    adv_loss, _ = self._coca_objective(
                        x + delta_star, edge_index,
                        sigmoid_feat, sigmoid_edge,
                        y_hat, y_hat_s,
                    )
                else:
                    adv_loss = torch.tensor(0.0, device=self.device)

                # ── (c) 稀疏 + 连续性正则 ──
                sparsity_loss = (
                    self.sparsity_coeff_feat * sigmoid_feat.sum()
                    + self.sparsity_coeff_edge * sigmoid_edge.sum()
                )
                continuity_loss = self._continuity_regularization(
                    sigmoid_feat, edge_index
                )

                # ── (d) 总损失 ──
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

        # ── 后处理 ───────────────────────────────────────
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
    # Coca 损失（事实 + 反事实，供干净/对抗两路共用）
    # ─────────────────────────────────────────────────────────

    def _coca_objective(self, x, edge_index, sigmoid_feat, sigmoid_edge,
                        y_hat, y_hat_s):
        """
        计算单一视图（干净或被扰动）下的 Coca 对比损失 L_f + L_c。

        返回:
            total: 标量 tensor，  α·L_f + (1−α)·L_c
            (probs_fact, probs_cf): 备用诊断
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
    # 内层：PGD 求最坏特征扰动 δ*
    # ─────────────────────────────────────────────────────────

    def _inner_pgd(self, x, edge_index, sigmoid_feat, sigmoid_edge,
                   y_hat, y_hat_s):
        """
        给定当前 mask，沿 Coca 损失上升方向找半径 ε 内最坏的 δ。

        说明:
            - δ 只加在节点特征上，不改 edge_index
            - mask 在内层视为常量（已 detach）
            - 上升目标：让 mask 看到 (x+δ) 后 Coca 损失变大，
                       即解释在该扰动下"失效"
            - 用 sign-gradient + L∞ 投影（标准 PGD-L∞）

        返回:
            delta: (num_nodes, feat_dim) 张量，已 detach
        """
        delta = torch.zeros_like(x, requires_grad=True)
        # 随机初始化（在 ε 球内均匀采样），有助于跳出鞍点
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
                # 上升一步
                delta = delta + self.pgd_step_size * grad.sign()
                # L∞ 投影到 ε 球
                delta = torch.clamp(delta, min=-self.eps, max=self.eps)

        return delta.detach()


# ══════════════════════════════════════════════════════════════════════
# 批量解释器：与 BatchCocaExplainer 同接口
# ══════════════════════════════════════════════════════════════════════

class BatchRobustExplainer:
    """
    批量运行 RobustExplainer，对数据集中所有被正确检测为 vulnerable 的样本生成解释。
    接口与 BatchCocaExplainer 一致，可直接替换。
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
                    print(f"[Robust_Exp] 已解释 {len(results)} 个样本...")

            except Exception as e:
                print(f"[Robust_Exp] 样本 {idx} 解释失败: {e}")
                continue

        print(f"[Robust_Exp] 完成，共解释 {len(results)} 个样本。")
        return results


# ══════════════════════════════════════════════════════════════════════
# 使用示例
# ══════════════════════════════════════════════════════════════════════

def demo_explain_robust():
    from src.model.wrapper import ModelWrapper
    from src.utils.gen_embedding import read_json

    wrapper = ModelWrapper(
        'reveal',
        '{HOME_PATH}/vul_explain/23_explain_eval_ISSTA/trained_model/'
        'ori-ds/reveal/reveal-cwe119/mod_94.59_92.5_96.77_93.61.ckpt'
    )

    explainer = RobustExplainer(
        model=wrapper.model,
        device='cuda',
        alpha=0.5,
        lr=0.01,
        epochs=300,
        top_k=5,
        # ── 对抗训练超参 ──
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
    print(f"原始样本预测标签: {pred_label}")

    result = explainer.explain(data, pred_label)
    print("top-k 关键节点: " + str(result['top_nodes']))
    print(f"对抗损失收敛: 首={result['adv_loss_history'][0]:.4f}, "
          f"末={result['adv_loss_history'][-1]:.4f}")


if __name__ == '__main__':
    demo_explain_robust()
