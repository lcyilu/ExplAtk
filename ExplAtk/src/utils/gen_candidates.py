import re
import threading
from gensim.models import KeyedVectors, Word2Vec
import torch
import transformers
from src.config import LOCAL_CODEBERT_PATH, WORD2VEC_PATH, LOCAL_CODET5_PATH
from src.utils.parser import initialize_language_parser, is_not_keyword, src2tree
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer, T5ForConditionalGeneration


# ════════════════════════════════════════════════════════════════
# 线程安全的进程单例缓存
# ════════════════════════════════════════════════════════════════

# CodeBert 模型按 (path, device) 缓存,所有线程共享同一份 GPU 权重
_MLM_MODEL_CACHE = {}
_MLM_MODEL_LOCK = threading.Lock()

# CodeBertTokenizerAligned 用线程本地缓存:
# - 避免每次调用 gen_candis_* 都重新加载 tokenizer + 重建 tree-sitter parser
# - 用 thread-local 是为了规避 RobertaTokenizer / tree-sitter 在多线程下的潜在状态问题
#   每个线程一份,4 线程就 4 个 aligner(每个几十 MB CPU 内存,不上 GPU,可忽略)
_aligner_local = threading.local()

class CodeBertTokenizerAligned:
    def __init__(self, model_name=LOCAL_CODEBERT_PATH, lang='cpp'):
        print(f"正在加载 Tokenizer: {model_name} ...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # 加载 C++ 解析器
        self.parser = initialize_language_parser(lang)

    def get_all_leaf_nodes(self, node):
        """递归获取所有叶子节点"""
        # Blocklist for node types to skip entirely (comments, strings)
        if node.type in ['comment', 'string_literal', 'char_literal', 'preproc_include']:
            return []
        
        if len(node.children) == 0:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self.get_all_leaf_nodes(child))
        return leaves

    def tokenize_with_alignment(self, src):
        """
        核心函数：
        输入: 源代码字符串
        输出: 
          1. model_tokens: 用于输入 CodeBERT 的 token list (例如 ['<s>', 'int', 'Ġs', 'ush', 'u', ...])
          2. alignment_map: 一个列表，长度与 model_tokens 相同。
             map[i] = {
                'source_text': 'sushu',   # 这个 token 属于源码里的哪个词
                'node_type': 'identifier',# 源码里的语法类型
                'is_target': True/False   # 是否是潜在的攻击目标(变量名)
             }
        """
        # 1. 确保是 bytes，用于 tree-sitter 精确切片
        if isinstance(src, str):
            src = src.encode('utf-8')
        tree = self.parser.parse(src)
        root_node = tree.root_node
        
        # 2. 获取所有叶子节点 (Lexical Tokens)
        leaf_nodes = self.get_all_leaf_nodes(root_node)
        
        # 3. 初始化结果容器
        full_tokens = [self.tokenizer.cls_token] # [<s>]
        alignment_map = [None] # <s> 没有对应的源码
        
        last_end_byte = 0
        
        for node in leaf_nodes:
            # --- A. 获取节点文本 ---
            node_text = src[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
            
            # --- B. 处理节点前的空白 (关键!) ---
            # CodeBERT (RoBERTa) 依赖 'Ġ' (space) 来区分单词边界。
            # 我们检查当前节点和上一个节点之间是否有 gap
            has_space_prefix = False
            if node.start_byte > last_end_byte:
                gap = src[last_end_byte:node.start_byte].decode('utf-8', errors='ignore')
                if len(gap) > 0 and gap.isspace():
                    has_space_prefix = True
            
            # 构造分词输入：如果前面有空格，RoBERTa 需要在词前加空格
            # 注意：RoBERTa tokenizer 的行为是，如果输入字符串以空格开头，它会把第一个 token 标记为 'Ġ...'
            # 为了模拟句子的连续性，我们手动控制
            input_text = node_text
            if has_space_prefix:
                input_text = " " + node_text 
            
            # --- C. 调用 Tokenizer ---
            # add_special_tokens=False: 我们自己控制 <s> </s>
            sub_tokens = self.tokenizer.tokenize(input_text)
            
            # 修正：如果这是文件的第一个词，即使源码没空格，通常也不加 Ġ (视具体 tokenizer 实现而定，RoBERTa 比较 tricky)
            # 但最简单的方法是直接信赖 tokenizer 对 " text" 的处理。
            
            # --- D. 记录映射关系 ---
            # 判定这是否是一个攻击目标 (Identifier)
            # 简单逻辑：类型是 identifier 且符合变量命名规范
            is_target = (node.type == 'identifier' or node.type == 'field_identifier') and \
                        re.match(r'^[a-zA-Z_]\w*$', node_text) is not None
            
            # 将生成的每一个 sub-token 都指向当前的 source node
            for sub_token in sub_tokens:
                full_tokens.append(sub_token)
                alignment_map.append({
                    'source_text': node_text,
                    'node_type': node.type,
                    'is_target': is_target,
                    'start_byte': node.start_byte, # 方便后续替换源码
                    'end_byte': node.end_byte
                })
            
            last_end_byte = node.end_byte

        # 4. 结尾处理
        full_tokens.append(self.tokenizer.sep_token) # [</s>]
        alignment_map.append(None)
        
        return full_tokens, alignment_map
    
def generate_candidates_for_variable(model, tokenizer, tokens, align_map, target_var_name, top_k=30):
    """
    针对指定的变量名 (target_var_name)，生成替换候选词。
    策略：将该变量的所有 token 替换为 *单个* <mask>。
    """
    # 1. 找到所有属于 target_var_name 的索引区间
    # 结构: [ [2, 3, 4], [10, 11, 12] ]
    occurrences = [] 
    current_span = []
    
    for i, info in enumerate(align_map):
        if info and info['source_text'] == target_var_name and info['is_target']:
            current_span.append(i)
        else:
            if current_span:
                occurrences.append(current_span)
                current_span = []
    # 处理结尾
    if current_span: occurrences.append(current_span)
    
    if not occurrences:
        return []

    # 2. 构建 Masked Token IDs
    # 我们需要构建一个新的 token list，把 span 替换为单 mask
    masked_token_ids = []
    token_ids_raw = tokenizer.convert_tokens_to_ids(tokens) # 将 string token 转为 int ID
    mask_token_id = tokenizer.mask_token_id
    
    i = 0
    mask_indices_in_new_list = [] # 记录新列表中 mask 的位置，方便取预测结果
    
    while i < len(token_ids_raw):
        # 检查当前 i 是否在某个 span 的开头
        is_start_of_span = False
        span_len = 0
        
        for span in occurrences:
            if span[0] == i:
                is_start_of_span = True
                span_len = len(span)
                break
        
        if is_start_of_span:
            # 这是一个变量的开始，插入一个 mask
            masked_token_ids.append(mask_token_id)
            mask_indices_in_new_list.append(len(masked_token_ids) - 1)
            # 跳过这个变量原来的所有 token
            i += span_len
        else:
            # 普通 token，照搬
            masked_token_ids.append(token_ids_raw[i])
            i += 1
    model_max_len = 512

    target_mask_idx = mask_indices_in_new_list[0] # 取第一个 mask 位置

    print("Start Predicting Words!")
    # 3. 模型预测
    if len(masked_token_ids) > model_max_len:
        # 计算窗口的起止位置
        # 尽量让 mask 位于窗口中间
        half_window = model_max_len // 2
        start = max(0, target_mask_idx - half_window)
        end = min(len(masked_token_ids), start + model_max_len)
        
        # 修正 start：如果 end 到底了，start 要往前挪，保证凑够 512
        if end - start < model_max_len:
            start = max(0, end - model_max_len)
            
        # 截取窗口
        window_input_ids = masked_token_ids[start:end]
        
        # 修正 mask 在新窗口中的索引
        relative_mask_idx = target_mask_idx - start
        
        # 构造 Tensor
        # 1. 输入：把整个窗口喂进去，为了提供上下文
        input_tensor = torch.tensor([window_input_ids]).to(model.device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            # predictions shape: [Batch=1, Seq_Len=512, Vocab_Size=50265]
            predictions = outputs.logits 
            
        # 2. 提取：只看 Mask 那个位置的预测结果
        # 哪怕模型输出了 512 个位置的预测，我们只取 relative_mask_idx 这一行
        target_token_logits = predictions[0, relative_mask_idx] 
        
        # 3. 排序取 Top-K
        top_k_probs, top_k_ids = torch.topk(target_token_logits, top_k)
    else:
        input_tensor = torch.tensor([masked_token_ids]).to(model.device)
    
        with torch.no_grad():
            outputs = model(input_tensor)
            predictions = outputs.logits # [1, seq_len, vocab_size]
            
        # 4. 提取候选词
        # 我们通常取所有 mask 位置预测结果的综合（例如取第一个 mask 的预测，或者取所有 mask 预测的交集/乘积）
        # ALERT 简单做法：只看第一个 mask 的预测结果即可（因为上下文是双向的，模型知道所有 mask 是同一个变量）
        
        probs = predictions[0, target_mask_idx] # [vocab_size]
        
        top_k_probs, top_k_ids = torch.topk(probs, top_k)
    
    results = []
    for idx in top_k_ids:
        word = tokenizer.decode([idx]).strip()
        # 简单过滤：剔除原来的名字，剔除特殊字符
        if word != target_var_name and word.isidentifier() and is_not_keyword(word):
        # if word != target_var_name and word.isidentifier():
            results.append(word)
    return results

def generate_candidates_for_variable_codet5(model, tokenizer, tokens, align_map, target_var_name, top_k=30):
    """
    针对指定的变量名 (target_var_name)，使用 CodeT5 生成替换候选词。
    策略：
    1. 将该变量的所有 token 替换为 <extra_id_0>。
    2. 使用 model.generate 生成 top_k 个序列。
    3. 解析序列提取单词。
    """
    
    # ==========================
    # 1. 找到变量的所有位置 (逻辑同 CodeBERT)
    # ==========================
    occurrences = [] 
    current_span = []
    
    for i, info in enumerate(align_map):
        if info and info['source_text'] == target_var_name and info['is_target']:
            current_span.append(i)
        else:
            if current_span:
                occurrences.append(current_span)
                current_span = []
    if current_span: occurrences.append(current_span)
    
    if not occurrences:
        return []

    # ==========================
    # 2. 构建 Masked Input IDs
    # ==========================
    masked_token_ids = []
    token_ids_raw = tokenizer.convert_tokens_to_ids(tokens)
    
    # CodeT5 的哨兵 ID (Sentinel Token)
    # 对于 codet5-base, <extra_id_0> 的 ID 通常是 32099
    # 如果 tokenizer 加载了特殊 token，也可以用 tokenizer.convert_tokens_to_ids('<extra_id_0>')
    # 这里为了稳健，优先尝试获取，获取不到则用默认值
    sentinel_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    if sentinel_id == tokenizer.unk_token_id:
        sentinel_id = 32099 
    
    i = 0
    mask_indices_in_new_list = [] 
    
    while i < len(token_ids_raw):
        is_start_of_span = False
        span_len = 0
        
        for span in occurrences:
            if span[0] == i:
                is_start_of_span = True
                span_len = len(span)
                break
        
        if is_start_of_span:
            # 这里的区别：CodeT5 将整个 span 替换为 *一个* 哨兵
            masked_token_ids.append(sentinel_id)
            mask_indices_in_new_list.append(len(masked_token_ids) - 1)
            i += span_len
        else:
            masked_token_ids.append(token_ids_raw[i])
            i += 1
            
    # ==========================
    # 3. 窗口切分 (Windowing)
    # ==========================
    model_max_len = 512
    target_mask_idx = mask_indices_in_new_list[0] # 以第一个 mask 为中心

    final_input_ids = []

    if len(masked_token_ids) > model_max_len:
        half_window = model_max_len // 2
        start = max(0, target_mask_idx - half_window)
        end = min(len(masked_token_ids), start + model_max_len)
        
        if end - start < model_max_len:
            start = max(0, end - model_max_len)
            
        final_input_ids = masked_token_ids[start:end]
    else:
        final_input_ids = masked_token_ids

    # 转为 Tensor
    input_tensor = torch.tensor([final_input_ids]).to(model.device)

    # ==========================
    # 4. CodeT5 生成 (核心区别)
    # ==========================
    print(f"CodeT5 Predicting for: {target_var_name}...")
    
    # 使用 Beam Search 生成多个结果
    outputs = model.generate(
        input_tensor, 
        max_length=16,             # 变量名通常很短，不需要生成太长
        num_beams=top_k + 5,       # Beam 宽度略大于需要的 k，保证多样性
        num_return_sequences=top_k,# 返回 k 个序列
        early_stopping=True
    )
    
    # ==========================
    # 5. 解析与过滤
    # ==========================
    candidates = []
    seen_candidates = set() # 去重
    
    for output_ids in outputs:
        # 解码
        raw_text = tokenizer.decode(output_ids, skip_special_tokens=False)
        # print(f"Raw Generated Text: {raw_text}")
        
        # 解析：标准格式是 "<pad> <extra_id_0> prediction <extra_id_1> </s>"
        # 也有可能是 "<pad> <s> <extra_id_0> prediction <extra_id_1> </s>"
        text_content = raw_text.replace("<pad>", "").replace("</s>", "").replace("<s>", "").strip()
        
        predicted_word = ""
        
        if "<extra_id_0>" in text_content:
            parts = text_content.split("<extra_id_0>")
            if len(parts) > 1:
                content_after = parts[1]
                if "<extra_id_1>" in content_after:
                    predicted_word = content_after.split("<extra_id_1>")[0].strip()
                else:
                    predicted_word = content_after.strip()
            preds = re.findall(r'[a-zA-Z_]\w*', predicted_word)
            predicted_word = preds[0] if preds else predicted_word

        # 如果解析失败，可能是纯文本，直接用
        if not predicted_word:
             predicted_word = text_content.replace("<extra_id_0>", "").strip()

        # --- 过滤逻辑 ---
        if not predicted_word: continue
        
        # 1. 去除空格 (CodeT5 有时会生成带 'Ġ' 效果的空格，decode 后就是普通空格)
        predicted_word = predicted_word.strip()
        
        # 2. 排除原名, 非标识符，关键字, 去重
        if predicted_word != target_var_name and predicted_word.isidentifier() and is_not_keyword(predicted_word) and predicted_word not in seen_candidates:
            seen_candidates.add(predicted_word)
            candidates.append(predicted_word)
        
            
        if len(candidates) >= top_k:
            break
            
    return candidates

def init_mlm(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    线程安全 + 进程单例缓存版本。
    所有线程共享同一份 CodeBert,GPU 显存只占 1 份(原本 4 线程 = 4 份)。
    签名与原版一致,上游 {atk_name}.py 一行不动。
    """
    key = (LOCAL_CODEBERT_PATH, str(device))

    # 快路径:无锁读取
    cached = _MLM_MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    # 慢路径:双重检查锁,首次访问以外开销为零
    with _MLM_MODEL_LOCK:
        cached = _MLM_MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        print(f"[init_mlm] 首次加载 CodeBert: {LOCAL_CODEBERT_PATH} -> {device}")
        # print(f"[init_mlm] 首次加载 CodeBert: {LOCAL_CODET5_PATH} -> {device}")
        model = RobertaForMaskedLM.from_pretrained(LOCAL_CODEBERT_PATH)
        # model = T5ForConditionalGeneration.from_pretrained(LOCAL_CODET5_PATH)
        model.eval()
        model.to(device)
        _MLM_MODEL_CACHE[key] = model
        return model
    
def _get_aligner(model_name, lang='cpp'):
    """
    线程本地的 CodeBertTokenizerAligned 缓存。
    每个线程首次调用时构造 1 个 aligner 并复用,避免每次 gen_candis_* 都重新
    加载 RobertaTokenizer (vocab/merges 文件,几十 MB 的反复 IO)。
    """
    cache = getattr(_aligner_local, 'cache', None)
    if cache is None:
        cache = {}
        _aligner_local.cache = cache
    key = (model_name, lang)
    if key in cache:
        return cache[key]
    aligner = CodeBertTokenizerAligned(model_name=model_name, lang=lang)
    cache[key] = aligner
    return aligner

def gen_candis(code, mlm_model, target_var, *, _precomputed=None):
    """
    生成 CodeBERT 候选词。
    
    Parameters
    ----------
    _precomputed : tuple | None
        若提供 (tokens, align_map, tokenizer)，则跳过重复的 tokenize 步骤。
        可通过 precompute_tokenize() 获取。
    """
    if _precomputed is not None:
        tokens, align_map, tokenizer = _precomputed
    else:
        aligner = _get_aligner(LOCAL_CODEBERT_PATH)
        tokens, align_map = aligner.tokenize_with_alignment(code)
        tokenizer = aligner.tokenizer

    candidates = generate_candidates_for_variable(model=mlm_model, tokenizer=tokenizer, tokens=tokens, align_map=align_map, target_var_name=target_var)

    return candidates

def gen_candis_codet5(code, mlm_model, target_var, *, _precomputed=None):
    """
    生成 CodeT5 候选词。
    
    Parameters
    ----------
    _precomputed : tuple | None
        若提供 (tokens, align_map, tokenizer)，则跳过重复的 tokenize 步骤。
        可通过 precompute_tokenize_codet5() 获取。
    """
    if _precomputed is not None:
        tokens, align_map, tokenizer = _precomputed
    else:
        aligner = _get_aligner(LOCAL_CODET5_PATH)
        tokens, align_map = aligner.tokenize_with_alignment(code)
        tokenizer = aligner.tokenizer

    candidates = generate_candidates_for_variable_codet5(model=mlm_model, tokenizer=tokenizer, tokens=tokens, align_map=align_map, target_var_name=target_var)

    return candidates


def precompute_tokenize(code, lang='cpp'):
    """
    对代码做一次 CodeBERT tokenize，返回 (tokens, align_map, tokenizer) 三元组。
    后续对同一份代码的多个变量生成候选时，将此结果传入
    gen_candis(..., _precomputed=result) 即可避免重复 tokenize。
    """
    aligner = _get_aligner(LOCAL_CODEBERT_PATH, lang=lang)
    tokens, align_map = aligner.tokenize_with_alignment(code)
    return tokens, align_map, aligner.tokenizer


def precompute_tokenize_codet5(code, lang='cpp'):
    """
    对代码做一次 CodeT5 tokenize，返回 (tokens, align_map, tokenizer) 三元组。
    后续对同一份代码的多个变量生成候选时，将此结果传入
    gen_candis_codet5(..., _precomputed=result) 即可避免重复 tokenize。
    """
    aligner = _get_aligner(LOCAL_CODET5_PATH, lang=lang)
    tokens, align_map = aligner.tokenize_with_alignment(code)
    return tokens, align_map, aligner.tokenizer
    
import numpy as np

def most_dissimilar_w2v_fast(wv, target, top_k=10, exclude=None):
    """
    从 wv 中找与 target 最不相似的 top_k 个词。

    target 可以是:
      1. str: 词表中的 token
      2. np.ndarray/list: 已经算好的向量，比如 mean_vec

    Returns:
      List[(word, similarity)]
    """
    exclude = set(exclude or [])

    # case 1: target 是词表 token
    if isinstance(target, str):
        if target not in wv.key_to_index:
            return []

        target_vec = wv.get_vector(target, norm=True)
        exclude.add(target)

    # case 2: target 是向量，比如 mean_vec
    else:
        target_vec = np.asarray(target, dtype=np.float32).reshape(-1)

        if target_vec.shape[0] != wv.vector_size:
            raise ValueError(
                f"target vector dim mismatch: got {target_vec.shape[0]}, "
                f"expected {wv.vector_size}"
            )

        norm = np.linalg.norm(target_vec)

        if norm == 0 or not np.isfinite(norm):
            return []

        target_vec = target_vec / norm

    # 所有词向量，已归一化
    all_vecs = wv.get_normed_vectors()

    # cosine similarity
    sims = np.dot(all_vecs, target_vec)

    # 排除已有变量名 / 原词
    for word in exclude:
        idx = wv.key_to_index.get(word)
        if idx is not None:
            sims[idx] = np.inf

    k = min(top_k, len(sims))

    if k <= 0:
        return []

    # 取 similarity 最小的 k 个
    idx = np.argpartition(sims, k - 1)[:k]
    idx = idx[np.argsort(sims[idx])]

    return [
        (wv.index_to_key[i], float(sims[i]))
        for i in idx
    ]

def generate_candidates_w2v(
    wv,          # 已加载的 gensim Word2Vec 词表
    target_var: str,    # 目标变量名，如 "sushu_counter"
    top_k: int = 5,    # 返回候选词数量
) -> list:
    """
    基于 Word2Vec 余弦相似度生成候选标识符。

    策略：
      1. 直接查询 target_var 是否在 W2V 词表中
      2. 若不在（OOV），尝试拆分下划线子词后取均值向量查询
      3. 过滤：剔除原词、非标识符、C++ 关键字
    """

    # ── 情况 1：词直接在词表中 ───────────────────────────────────────────────
    if target_var in wv:
        similar = wv.most_similar(target_var, topn=top_k * 2)  # 多取一些，过滤后剩 top_k
        # similar = most_dissimilar_w2v_fast(wv,target_var,top_k * 2)

    # ── 情况 2：OOV，尝试子词均值（处理 snake_case 变量名）───────────────────
    else:
        parts = [p for p in re.split(r'[_\d]+', target_var) if p and p in wv]
        if not parts:
            # 完全 OOV，无法处理
            print(f"[W2V] '{target_var}' 及其子词均不在词表中，返回空列表")
            return []

        import numpy as np
        mean_vec = np.mean([wv[p] for p in parts], axis=0)
        similar  = wv.most_similar([mean_vec], topn=top_k * 2)
        # similar = most_dissimilar_w2v_fast(wv,mean_vec,top_k * 2)
        print(f"[W2V] '{target_var}' OOV，用子词 {parts} 的均值向量查询")

    # ── 过滤 ─────────────────────────────────────────────────────────────────
    candidates = []
    for word, score in similar:
        if (
            word != target_var          # 不是原词本身
            and word.isidentifier()     # 合法标识符
            and is_not_keyword(word)    # 不是关键字
        ):
            candidates.append(word)
        if len(candidates) >= top_k:
            break

    return candidates


# ── 替换原来的 gen_candis ─────────────────────────────────────────────────────
def gen_candis_w2v(
    target_var: str,
    wv,
    top_k: int = 30,
) -> list:
    """
    直接替换原来的 gen_candis()。
    不再需要 code、mlm_model、aligner，接口更简洁。
    """
    return generate_candidates_w2v(wv, target_var, top_k)
    

if __name__ == "__main__":
    aligner = CodeBertTokenizerAligned()
    
    code = "int sushu = 0; if (sushu > 10) { return; }"
    code = b"""
static void  rv34_pred_mv (RV34DecContext *r, int block_type, int subblock_no, int dmv_no) {
    MpegEncContext *s = &r->s;
    int mv_pos = s->mb_x * 2 + s->mb_y * 2 * s->b8_stride;
    int A [2] = {0}, B [2], C [2];
    int i, j;
    int mx, my;
    int avail_index = avail_indexes[subblock_no];
    int c_off = part_sizes_w[block_type];
    mv_pos += (subblock_no & 1) + (subblock_no >> 1) * s->b8_stride;
    if (subblock_no == 3)
        c_off = -1;
    if (r->avail_cache[avail_index - 1]) {
        A[0] = s->current_picture_ptr->f.motion_val[0][mv_pos - 1][0];
        A[1] = s->current_picture_ptr->f.motion_val[0][mv_pos - 1][1];
    }
    if (r->avail_cache[avail_index - 4]) {
        B[0] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride][0];
        B[1] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride][1];
    }
    else {
        B[0] = A[0];
        B[1] = A[1];
    }
    if (!r->avail_cache[avail_index - 4 + c_off]) {
        if (r->avail_cache[avail_index - 4] & &(r->avail_cache[avail_index - 1] || r->rv30)) {
            C[0] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride - 1][0];
            C[1] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride - 1][1];
        }
        else {
            C[0] = A[0];
            C[1] = A[1];
        }
    }
    else {
        C[0] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride + c_off][0];
        C[1] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride + c_off][1];
    }
    mx = mid_pred (A[0], B[0], C[0]);
    my = mid_pred (A[1], B[1], C[1]);
    mx += r->dmv[dmv_no][0];
    my += r->dmv[dmv_no][1];
    {
        j = 0;
        while (j < part_sizes_h[block_type]) {
            for (i = 0; i < part_sizes_w[block_type]; i++) {
                s->current_picture_ptr->f.motion_val[0][mv_pos + i + j * s->b8_stride][0] = mx;
                s->current_picture_ptr->f.motion_val[0][mv_pos + i + j * s->b8_stride][1] = my;
            }
            j++;
        }
    }
}
"""

    code = b'''
int FUN1(char* VAR1) {
    char VAR2[32];
    char VAR3[] = "";
    int VAR4 = 0;
    int VAR5;


    VAR5 = strlen(VAR1);
    FUN2("", VAR5);


    if (VAR5 > 0) {
        VAR4 = 1;
    }


    strcpy(VAR2, VAR1);


    FUN2("", VAR2);


    if (strlen(VAR2) > 0) {
        FUN2("");
        return VAR4;
    }

    return -1;
}                                                                                                             
'''

    # print(f"\n原始代码: {code}\n")
    
    tokens, align_map = aligner.tokenize_with_alignment(code)
    
    # # 模拟攻击：找到所有 'sushu' 的 token index
    # attack_indices = []
    
    # for i, token in enumerate(tokens):
    #     info = align_map[i]
    #     source_text = info['source_text'] if info else "N/A"
    #     node_type = info['node_type'] if info else "N/A"
    #     is_target = info['is_target'] if info else False
        
    #     print(f"{i:<6} | {token:<12} | {source_text:<12} | {node_type:<15} | {is_target}")
        
    #     if source_text == 'sushu':
    #         attack_indices.append(i)

    # print(f"\n[攻击目标定位] 变量 'sushu' 对应的 Token Indices: {attack_indices}")
    # print("这意味着如果你要 Mask 'sushu'，你需要把这些位置的 ID 都替换成 <mask>。")

    # 加载模型
    model = init_mlm()
    target_var = 'VAR1'
    candidates = generate_candidates_for_variable(model, aligner.tokenizer, tokens, align_map, target_var)

    print(f"变量 '{target_var}' 的 CodeBERT 推荐替换词: {candidates}")