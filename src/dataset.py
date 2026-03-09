# src/dataset.py
import torch
from torch.utils.data import Dataset
from tree_sitter import Language, Parser
import tree_sitter_solidity as tssoldity

# Configuration Tree-Sitter
SOL_LANGUAGE = Language(tssoldity.language())
parser = Parser(SOL_LANGUAGE)

def get_variable_nodes(node, variables):
    # Find all identifier nodes (potential variables)
    if node.type == 'identifier' and node.child_count == 0:
        variables.append(node)
    for child in node.children:
        get_variable_nodes(child, variables)
    return variables

def extract_data_flow_edges(node, edges):
    # Assignment: x = y
    if node.type == 'assignment_expression':
        left_node = node.child_by_field_name('left')
        right_node = node.child_by_field_name('right')
        add_edges(left_node, right_node, edges)
    # Declaration: uint x = a
    elif node.type == 'variable_declaration_statement':
        declarations = []
        get_variable_nodes(node, declarations)
        if len(declarations) >= 2:
            target = declarations[0]
            sources = declarations[1:]
            for s in sources:
                edges.append((s.id, target.id))
    for child in node.children:
        extract_data_flow_edges(child, edges)

def add_edges(lhs, rhs, edges):
    if lhs and rhs:
        lhs_vars = get_variable_nodes(lhs, [])
        rhs_vars = get_variable_nodes(rhs, [])
        for source in rhs_vars:
            for target in lhs_vars:
                edges.append((source.id, target.id))

def build_graph_mask(seq_len, code_indices, var_indices, flow_edges, alignment_edges):
    # Initialiser avec une valeur très négative (interdit)
    mask = torch.full((seq_len, seq_len), -1e9)

    # --- RÈGLE 1 : CLS, SEP et Code ---
    # In GraphCodeBERT, CLS, SEP and Code tokens can all see each other.
    # We define special indices (CLS is at 0, first SEP is after code, last SEP is at the end)
    special_indices = [0, code_indices[-1] + 1, seq_len - 1]
    allowed_text = special_indices + code_indices
    
    for i in allowed_text:
        for j in allowed_text:
            mask[i, j] = 0

    # --- RÈGLE 2 : Data Flow (E) ---
    # Variable nodes can see themselves and their sources.
    for i in var_indices:
        mask[i, i] = 0
        
    for src_idx, tgt_idx in flow_edges:
        # src_idx and tgt_idx are absolute indices in the sequence
        if tgt_idx < seq_len and src_idx < seq_len:
            mask[tgt_idx, src_idx] = 0 

    # --- RÈGLE 3 : Alignement (E') ---
    # Bidirectional link between graph node and code token
    for var_idx, code_idx in alignment_edges:
        if var_idx < seq_len and code_idx < seq_len:
            mask[var_idx, code_idx] = 0
            mask[code_idx, var_idx] = 0

    return mask

class SolidityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_seq_len=512, max_var_len=128):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_var_len = max_var_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = row['code']
        label = row['label_encoded']

        # 1. Parse AST and extract variables/flow
        tree = parser.parse(bytes(code, "utf8"))
        root = tree.root_node
        all_vars = get_variable_nodes(root, [])
        flow_edges_ast = []
        extract_data_flow_edges(root, flow_edges_ast)

        # 2. Tokenize code
        # We need to track token spans to align with AST nodes
        encoding = self.tokenizer(
            code,
            max_length=self.max_seq_len - self.max_var_len - 3, # CLS, 2 SEPs
            truncation=True,
            return_offsets_mapping=True,
            padding=False
        )
        
        code_ids = encoding['input_ids'] # Already has [CLS] and [SEP]
        offsets = encoding['offset_mapping']
        
        # Identify code tokens (excluding CLS and SEP)
        # Assuming CLS is at index 0 and SEP is at the end
        code_indices = list(range(1, len(code_ids) - 1))
        
        # 3. Handle Variable Nodes (V)
        # Limit the number of variable nodes to keep sequence length under control
        vars_to_use = all_vars[:self.max_var_len]
        var_names = [v.text.decode('utf8', errors='ignore') for v in vars_to_use]
        var_ids = self.tokenizer.convert_tokens_to_ids(var_names)
        
        # Total sequence: [CLS] + code + [SEP] + vars + [SEP]
        input_ids = code_ids + var_ids + [self.tokenizer.sep_token_id]
        
        # Sequence indices for variables
        var_start_idx = len(code_ids)
        var_indices = list(range(var_start_idx, var_start_idx + len(var_ids)))
        
        # 4. Map AST nodes to Sequence Indices
        node_id_to_seq_idx = {}
        alignment_edges = []
        
        # For each variable node in V, we find its corresponding token in C
        for i, var_node in enumerate(vars_to_use):
            var_seq_idx = var_indices[i]
            node_id_to_seq_idx[var_node.id] = var_seq_idx
            
            # Find which code token corresponds to this AST node
            # We look for the first token that overlaps with the node's range
            start_byte = var_node.start_byte
            for token_idx in code_indices:
                t_start, t_end = offsets[token_idx]
                if t_start <= start_byte < t_end:
                    alignment_edges.append((var_seq_idx, token_idx))
                    break

        # 5. Build Flow Edges for the sequence
        flow_edges = []
        for src_id, tgt_id in flow_edges_ast:
            if src_id in node_id_to_seq_idx and tgt_id in node_id_to_seq_idx:
                flow_edges.append((node_id_to_seq_idx[src_id], node_id_to_seq_idx[tgt_id]))

        # 6. Build Mask
        seq_len = len(input_ids)
        mask = build_graph_mask(seq_len, code_indices, var_indices, flow_edges, alignment_edges)

        # Padding
        padding_len = self.max_seq_len - seq_len
        if padding_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_len
            # Extend mask: padded tokens are fully masked
            new_mask = torch.full((self.max_seq_len, self.max_seq_len), -1e9)
            new_mask[:seq_len, :seq_len] = mask
            mask = new_mask
        else:
            input_ids = input_ids[:self.max_seq_len]
            mask = mask[:self.max_seq_len, :self.max_seq_len]

        # input_tensor = torch.tensor(input_ids, dtype=torch.long)
        
        # # Le vocabulaire de GraphCodeBERT a une taille maximale (souvent 50265).
        # # On force tous les IDs à rester entre 0 et la taille du vocab (ex: 50264)
        # # S'il y a un ID aberrant (ex: -1 ou 999999), on le remplace par l'ID du token [UNK] (qui est 3).
        # vocab_size = self.tokenizer.vocab_size
        # unk_id = self.tokenizer.unk_token_id
        
        # # Remplacer les valeurs hors limites par UNK
        # input_tensor = torch.where((input_tensor < 0) | (input_tensor >= vocab_size), 
        #                            torch.tensor(unk_id), 
        #                            input_tensor)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': mask.unsqueeze(0), # (1, seq_len, seq_len)
            'labels': torch.tensor(label, dtype=torch.long)
        }

