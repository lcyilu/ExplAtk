import torch
from src.config import DEVICE
from src.model.model import Devign_simplify, DevignModel, IVDetect, DeepWukong, IVDetect_simplify, RevealModel

_MODEL_FACTORIES = {
    "devign_simplify": lambda input_dim, output_dim: Devign_simplify(input_dim, output_dim),
    "ivdetect_simplify": lambda input_dim, output_dim: IVDetect_simplify(output_dim, input_dim),

    "devign": lambda input_dim, output_dim: DevignModel(input_dim, output_dim),
    "deepwukong": lambda input_dim, output_dim: DeepWukong(output_dim, input_dim),
    "ivdetect": lambda input_dim, output_dim: IVDetect(output_dim, input_dim),
    "reveal": lambda input_dim, output_dim: RevealModel(input_dim, output_dim),
}

def get_model(model_name, input_dim=100, output_dim=200):
    """
    Lazy model factory.
    只构造当前指定的模型，避免一次性实例化所有 victim model。
    """
    key = model_name.lower()

    factory = _MODEL_FACTORIES.get(key)
    if factory is None:
        supported = ", ".join(sorted(_MODEL_FACTORIES.keys()))
        raise ValueError(
            f"Model '{model_name}' is not recognized. "
            f"Supported models: {supported}"
        )

    return factory(input_dim, output_dim)

def load_model_from_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

class ModelWrapper:
    def __init__(self, model_name, checkpoint_path, input_dim=100, output_dim=200):
        self.model_name = model_name
        self.model = get_model(model_name, input_dim, output_dim)
        if self.model is None:
            raise ValueError(f"Model '{model_name}' is not recognized.")
        self.model = load_model_from_checkpoint(self.model, checkpoint_path)   
        self.model.to(torch.device(DEVICE))
        self.model.eval()
        self._query_count = 0

    def reset_query_count(self):
        self._query_count = 0

    def get_query_count(self):
        return self._query_count

    def _forward(self, data):
        with torch.no_grad():
            preds = self.model(data.x, data.edge_index)
        self._query_count += 1
        return preds
    
    def predict_prob(self, data, label):
        preds = self._forward(data)
        prob = preds[0][label].item()
        return prob
    
    def predict_label(self, data):
        preds = self._forward(data)
        predicted_label = torch.argmax(preds, dim=1)
        return predicted_label.item()

    def predict_label_and_true_conf(self, data, true_label):
        preds = self._forward(data)
        predicted_label = torch.argmax(preds, dim=1).item()
        true_conf = preds[0][true_label].item()
        return predicted_label, true_conf
    
    def predict_label_and_true_conf_margin(self, data, true_label):
        preds = self._forward(data)
        predicted_label = torch.argmax(preds, dim=1).item()
        true_conf = preds[0][true_label].item()
        other_conf = preds[0][1 - true_label].item()
        margin = true_conf - other_conf
        return predicted_label, true_conf, margin
    
    def attack_success(self, data, label):
        preds = self._forward(data)
        predicted_label = torch.argmax(preds, dim=1)
        return predicted_label.item() != label
    
    def compute_acceptance(self, true_label, current_data, proposed_data):
        # preds like: tensor([[0.0136, 0.9864]], device='cuda:0')
        current_preds = self._forward(current_data)
        proposed_preds = self._forward(proposed_data)
        
        current_prob = current_preds[0][true_label].item()
        proposed_prob = proposed_preds[0][true_label].item()
        
        alpha = min(1, current_prob / proposed_prob)

        return alpha
    
    def compute_importance(self, true_label, ori_pred, proposed_data):
        # preds like: tensor([[0.0136, 0.9864]], device='cuda:0')
        proposed_preds = self._forward(proposed_data)
        
        proposed_prob = proposed_preds[0][true_label].item()
        score = ori_pred - proposed_prob

        return score if score>=0 else 0