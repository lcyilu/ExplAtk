"""
Coca Explainer: 双视角因果推理解释器
================================================
基于 Coca (ICSE'24) 论文 Section 5.2 实现。

核心思想：
  通过同时优化事实推理（factual reasoning）和反事实推理（counterfactual reasoning），
  生成既有效（覆盖真正的漏洞语句）又简洁（范围尽可能小）的解释子图。

  - 事实推理：保留的子图应维持原始预测 → 保证有效性
  - 反事实推理：移除的子图应改变预测 → 保证简洁性

适配说明：
  - 适配 model(x, edge_index) 签名的 GNN 模型
  - 节点特征掩码完全可微，边掩码使用可微近似
  - 输出节点级重要性分数，可直接映射到源码语句
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
# 核心类：Coca Explainer
# ══════════════════════════════════════════════════════════════════════

class CocaExplainer:
    """
    Coca 双视角因果推理解释器。

    对一个被检测为 vulnerable 的样本，通过优化可学习掩码，
    找到对模型预测最关键的节点（语句）和边（依赖关系）。

    用法:
        explainer = CocaExplainer(model, device='cuda')
        result = explainer.explain(data, predicted_label=1)
        print(result['node_importance'])   # 每个节点的重要性分数
        print(result['top_nodes'])         # top-k 关键节点索引
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
            model:              GNN 模型实例（已加载权重），forward 签名为 model(x, edge_index)
            device:             计算设备
            alpha:              事实/反事实推理的权衡系数，值越大越侧重有效性
                                α=0.5 表示均衡（论文默认），α>0.5 侧重有效性，α<0.5 侧重简洁性
            lr:                 掩码优化的学习率
            epochs:             优化迭代轮数
            sparsity_coeff_feat: 节点特征掩码的稀疏正则系数
            sparsity_coeff_edge: 边掩码的稀疏正则系数
            top_k:              返回 top-k 个最重要的节点
        """
        self.model = model
        self.device = device
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.sparsity_coeff_feat = sparsity_coeff_feat
        self.sparsity_coeff_edge = sparsity_coeff_edge
        self.top_k = top_k

        # 冻结模型参数（只优化掩码，不更新模型）
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    # ─────────────────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────────────────

    def explain(self, data, predicted_label):
        """
        对单个样本生成解释。

        Args:
            data:             PyG Data 对象 (x, edge_index, edge_attr, y)
            predicted_label:  模型对该样本的预测标签（通常为 1=vulnerable）

        Returns:
            dict: {
                'node_importance':  np.ndarray (num_nodes,) 每个节点的重要性分数（0~1）,
                'edge_importance':  np.ndarray (num_edges,) 每条边的重要性分数（0~1）,
                'top_nodes':        list[int]  top-k 关键节点索引,
                'top_edges':        list[int]  top-k 关键边索引,
                'feat_mask_raw':    np.ndarray sigmoid 前的原始掩码值,
                'edge_mask_raw':    np.ndarray sigmoid 前的原始掩码值,
                'loss_history':     list[float] 训练损失曲线,
            }
        """
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            return self._empty_result(num_nodes, 0)

        y_hat = predicted_label
        y_hat_s = 1 - y_hat  # 二分类：另一个标签

        # 初始化为正值，sigmoid 后 ≈ 0.95，表示"默认全部保留"
        # 加小噪声打破对称性
        feat_mask = (3.0 * torch.ones(num_nodes, device=self.device)
                    + 0.1 * torch.randn(num_nodes, device=self.device))
        feat_mask.requires_grad = True

        edge_mask = (3.0 * torch.ones(num_edges, device=self.device)
                    + 0.1 * torch.randn(num_edges, device=self.device))
        edge_mask.requires_grad = True

        optimizer = torch.optim.Adam([feat_mask, edge_mask], lr=self.lr)

        # ── 优化循环 ────────────────────────────────────
        loss_history = []
        best_loss = float('inf')
        best_feat_mask = None
        best_edge_mask = None

        with torch.backends.cudnn.flags(enabled=False):
            for epoch in range(self.epochs):
                optimizer.zero_grad()

                sigmoid_feat = torch.sigmoid(feat_mask)
                sigmoid_edge = torch.sigmoid(edge_mask)

                # ── 事实推理 (Factual Reasoning) ──
                # 保留的子图应维持原始预测
                probs_fact = self._masked_forward(
                    x, edge_index, sigmoid_feat, sigmoid_edge, mode='factual'
                )
                S_f = probs_fact[0, y_hat]

                # ── 反事实推理 (Counterfactual Reasoning) ──
                # 移除子图后应改变预测
                probs_cf = self._masked_forward(
                    x, edge_index, sigmoid_feat, sigmoid_edge, mode='counterfactual'
                )
                S_c_neg = -probs_cf[0, y_hat]

                # ── 对比损失 (Eq. 7) ──
                L_f = F.relu(0.5 - S_f + probs_fact[0, y_hat_s])
                L_c = F.relu(0.5 - S_c_neg - probs_cf[0, y_hat_s])

                # ── 稀疏性正则 ──
                sparsity_loss = (
                    self.sparsity_coeff_feat * sigmoid_feat.sum()
                    + self.sparsity_coeff_edge * sigmoid_edge.sum()
                )

                # ── 连续性正则（鼓励相邻节点有相似的掩码值）──
                continuity_loss = self._continuity_regularization(
                    sigmoid_feat, edge_index
                )

                # ── 总损失 (Eq. 8) ──
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

        # ── 后处理：生成最终结果 ──────────────────────
        node_importance = torch.sigmoid(best_feat_mask).cpu().numpy()
        edge_importance = torch.sigmoid(best_edge_mask).cpu().numpy()

        # top-k 节点
        k = min(self.top_k, num_nodes)
        top_nodes = np.argsort(node_importance)[::-1][:k].tolist()

        # top-k 边
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
    # 掩码前向传播
    # ─────────────────────────────────────────────────────────

    def _masked_forward(self, x, edge_index, sigmoid_feat, sigmoid_edge, mode):
        """
        带掩码的可微前向传播。

        对于节点掩码：直接乘以节点特征（完全可微）。
        对于边掩码：通过缩放源节点在每条边上的贡献来近似实现（可微近似）。

        Args:
            x:            (num_nodes, feat_dim) 原始节点特征
            edge_index:   (2, num_edges) 边索引
            sigmoid_feat: (num_nodes,) sigmoid 后的节点掩码值
            sigmoid_edge: (num_edges,) sigmoid 后的边掩码值
            mode:         'factual' 或 'counterfactual'

        Returns:
            probs: (1, num_classes) softmax 概率
        """
        if mode == 'factual':
            # 事实推理：保留掩码选中的部分
            node_weight = sigmoid_feat
            edge_weight = sigmoid_edge
        else:
            # 反事实推理：移除掩码选中的部分
            node_weight = 1.0 - sigmoid_feat
            edge_weight = 1.0 - sigmoid_edge

        # ── 节点特征掩码（完全可微）──
        x_masked = x * node_weight.unsqueeze(-1)

        # ── 边掩码（可微近似）──
        # 策略：将边权重聚合到节点上，作为额外的缩放因子
        # 原理：如果一个节点的所有入边权重都很低，
        #       说明该节点接收的信息不重要，应进一步抑制
        x_masked = self._apply_edge_mask_to_features(
            x_masked, edge_index, edge_weight
        )

        # ── 模型前向传播（不加 no_grad，保持可微）──
        logits = self.model(x_masked, edge_index)
        probs = torch.softmax(logits, dim=-1)

        return probs

    def _apply_edge_mask_to_features(self, x, edge_index, edge_weight):
        """
        将边掩码的信息融入节点特征。

        由于模型 forward 签名为 model(x, edge_index) 不支持 edge_weight，
        我们通过修改节点特征来近似边掩码的效果：

        对每个节点 d，计算其入边权重的加权平均值 w_d，
        然后将节点特征缩放为 x[d] * w_d。

        这近似了"如果指向 d 的边被移除，d 接收到的信息就减少"的效果。

        Args:
            x:           (num_nodes, feat_dim) 已经过节点掩码的特征
            edge_index:  (2, num_edges)
            edge_weight: (num_edges,) 边权重

        Returns:
            x_scaled: (num_nodes, feat_dim) 缩放后的特征
        """
        num_nodes = x.shape[0]
        src, dst = edge_index

        # 计算每个节点的入边权重之和
        weighted_degree = torch.zeros(num_nodes, device=x.device)
        weighted_degree.scatter_add_(0, dst, edge_weight)

        # 计算每个节点的入边数量
        degree = torch.zeros(num_nodes, device=x.device)
        degree.scatter_add_(0, dst, torch.ones_like(edge_weight))

        # 加权平均（避免除零）
        degree = degree.clamp(min=1.0)
        node_scale = weighted_degree / degree  # (num_nodes,)

        # 对没有入边的节点（如入口节点），保持原始特征
        no_incoming = (degree <= 1.0) & (weighted_degree == 0)
        node_scale[no_incoming] = 1.0

        # 缩放节点特征
        x_scaled = x * node_scale.unsqueeze(-1)

        return x_scaled

    # ─────────────────────────────────────────────────────────
    # 正则化
    # ─────────────────────────────────────────────────────────

    def _continuity_regularization(self, sigmoid_feat, edge_index):
        """
        连续性正则：鼓励图中相邻节点的掩码值接近。
        如果节点 s 重要，那么与 s 有依赖关系的节点 d 也倾向于重要。
        这有助于生成连贯的解释子图，而不是孤立的节点。
        """
        src, dst = edge_index
        diff = (sigmoid_feat[src] - sigmoid_feat[dst]) ** 2
        return diff.mean()

    # ─────────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────────

    def _empty_result(self, num_nodes, num_edges):
        """当图为空或无边时的默认返回"""
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
# 批量解释器：对数据集中所有正确检测的样本批量生成解释
# ══════════════════════════════════════════════════════════════════════

class BatchCocaExplainer:
    """
    批量运行 Coca Explainer，对数据集中所有被正确检测为 vulnerable 的样本生成解释。

    用法:
        batch_explainer = BatchCocaExplainer(model, device='cuda')
        all_results = batch_explainer.explain_dataset(dataset, model_wrapper)
    """

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu", **explainer_kwargs):
        self.explainer = CocaExplainer(model, device, **explainer_kwargs)
        self.device = device

    def explain_dataset(self, dataset, model_wrapper, only_vulnerable=True):
        """
        对数据集中的样本批量生成解释。

        Args:
            dataset:          list[Data] PyG Data 对象列表
            model_wrapper:    ModelWrapper 实例（用于获取预测标签）
            only_vulnerable:  是否只解释被预测为 vulnerable 的样本

        Returns:
            list[dict]: 每个样本的解释结果，包含原始索引
        """
        results = []

        for idx, data in enumerate(dataset):
            pred_label = model_wrapper.predict_label(data)
            true_label = data.y.item() if hasattr(data, 'y') else None

            # 只解释模型预测正确且预测为 vulnerable 的样本
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
                    print(f"[Coca_Exp] 已解释 {len(results)} 个样本...")

            except Exception as e:
                print(f"[Coca_Exp] 样本 {idx} 解释失败: {e}")
                continue

        print(f"[Coca_Exp] 完成，共解释 {len(results)} 个样本。")
        return results


# ══════════════════════════════════════════════════════════════════════
# 解释结果分析工具
# ══════════════════════════════════════════════════════════════════════

class ExplanationAnalyzer:
    """分析和可视化解释结果的工具类。"""

    @staticmethod
    def get_explanation_subgraph(data, result, threshold=0.5):
        """
        从解释结果中提取解释子图。

        Args:
            data:      原始 PyG Data 对象
            result:    CocaExplainer.explain() 的返回值
            threshold: 节点/边掩码的阈值

        Returns:
            Data: 解释子图的 PyG Data 对象
        """
        node_mask = result['node_importance'] > threshold
        edge_mask = result['edge_importance'] > threshold

        # 保留的节点
        kept_nodes = np.where(node_mask)[0]
        if len(kept_nodes) == 0:
            # 阈值太高，回退到 top-k
            kept_nodes = np.array(result['top_nodes'])

        # 创建节点映射（旧索引 → 新索引）
        node_mapping = {old: new for new, old in enumerate(kept_nodes)}

        # 过滤边：只保留两端节点都在 kept_nodes 中的边
        edge_index = data.edge_index.numpy()
        kept_edges = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src in node_mapping and dst in node_mapping:
                if edge_mask[i]:
                    kept_edges.append([node_mapping[src], node_mapping[dst]])

        # 构建子图
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
        评估解释质量（VTP 指标）。

        Args:
            result:              CocaExplainer.explain() 的返回值
            ground_truth_nodes:  list[int] 真实漏洞相关节点索引

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
        打印解释结果。

        Args:
            result:    explain() 的返回值
            node_meta: 可选，节点元数据列表（含行号和代码）
        """
        print("=" * 60)
        print("Coca Explanation Result")
        print("=" * 60)
        print(f"Top-{len(result['top_nodes'])} 关键节点:")
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
# 使用示例
# ══════════════════════════════════════════════════════════════════════

def demo_explain_coca():
    # 1. 加载模型（你现有的流程）
    wrapper = ModelWrapper('reveal', '{HOME_PATH}/vul_explain/23_explain_eval_ISSTA/trained_model/ori-ds/reveal/reveal-cwe119/mod_94.59_92.5_96.77_93.61.ckpt')

    # 2. 创建解释器
    explainer = CocaExplainer(
        model=wrapper.model,        # 直接传入模型实例
        device='cuda',
        alpha=0.5,                  # 均衡有效性和简洁性
        lr=0.01,
        epochs=300,
        top_k=5,
    )

    # 3. 对单个样本生成解释
    data = read_json('{HOME_PATH}/VulDS/BigVul/ori-embedding/vul/1_CVE-2013-1788_poppler_CWE-119_bbc2d8918fe234b7ef2c480eb148943922cc0959_1.json')               # 一个被检测为 vulnerable 的样本
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
    pred_label = wrapper.predict_label(data)  # 通常为 1
    # print(f"原始子图: {data.x.shape[0]} 节点, "
    #       f"{data.edge_index.shape[1]} 边")
    print(f"原始样本预测标签: {pred_label}")

    # result = explainer.explain(data, pred_label)

    # # 4. 查看结果
    # print("每个节点 0~1 的重要性: " + str(result['node_importance']))
    # print("top-k 关键节点: " + str(result['top_nodes']))

    # # 5. 提取解释子图
    # sub_data = ExplanationAnalyzer.get_explanation_subgraph(data, result)
    # print(f"解释子图: {sub_data.x.shape[0]} 节点, "
    #       f"{sub_data.edge_index.shape[1]} 边")

    # sub_pred_label = wrapper.predict_label(sub_data)
    # print(f"解释子图预测标签: {sub_pred_label}")


# test_file
# {HOME_PATH}/VulDS/BigVul/all-src/trans_vul/1_CVE-2013-4263_FFmpeg_CWE-119_e43a0a232dbf6d3c161823c2e07c52e76227a1bc_3_10.c
# {HOME_PATH}/VulDS/BigVul/all-src/vul/1_CVE-2012-2895_Chrome_CWE-119_baef1ffd73db183ca50c854e1779ed7f6e5100a8_5.c
if __name__ == '__main__':
    demo_explain_coca()
