from transformers import StoppingCriteria, StoppingCriteriaList
import torch

# 1. 限制生成长度
class MaxLengthStopCriteria(StoppingCriteria):
    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[1] >= self.max_length  # 超过 max_length 停止

# 2. 避免重复（检查最后 max_repeats 个 token 是否重复）
class RepetitionStopCriteria(StoppingCriteria):
    def __init__(self, max_repeats=3):
        self.max_repeats = max_repeats

    def __call__(self, input_ids, scores, **kwargs):
        tokens = input_ids[0].tolist()
        if len(tokens) < self.max_repeats:
            return False
        last_tokens = tokens[-self.max_repeats:]
        return tokens[:-self.max_repeats].count(last_tokens) > 0  # 如果已经重复，停止

# 3. 置信度低（检测最高概率低于阈值）
class LowConfidenceStopCriteria(StoppingCriteria):
    def __init__(self, confidence_threshold=0.1):
        self.confidence_threshold = confidence_threshold

    def __call__(self, input_ids, scores, **kwargs):
        if scores is None or len(scores) == 0:
            return False
        last_probs = torch.softmax(scores[-1], dim=-1)  # 计算最后一个 token 的概率分布
        max_prob = torch.max(last_probs).item()  # 最高置信度
        return max_prob < self.confidence_threshold  # 置信度过低时停止


