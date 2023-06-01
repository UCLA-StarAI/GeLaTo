import itertools
import torch
import torch.nn as nn


# check whether x ends with y
def end_with(x, y):
    if len(y) > len(x):
        return False
    return x[-len(y):] == y


def remove_from_cnf(cnf, keyword, fix_order=False):
    if fix_order:
        if len(cnf) == 0:
            return cnf
        if keyword in cnf[0]:
            return cnf[1:]
        return cnf
    else:
        for i, clause in enumerate(cnf):
            if keyword in clause:
                return cnf[:i] + cnf[i+1:]
        return cnf


def update_cnf(prefix, cnf, sep_tokens_set=None, fix_order=False):
    keywords = set([keyword for clause in cnf for keyword in clause])
    for i in range(0, len(prefix)-1):
        for j in range(i+1, len(prefix)):
            if (prefix[i: j] in keywords) and (prefix[j] in sep_tokens_set):
                cnf = remove_from_cnf(cnf, prefix[i:j], fix_order=fix_order)
                break
    return cnf


# checks whether the prefix ends with some partial keywords
# and returns all possible next_seqs and next_cnfs
def case_analysis(prefix, keywords, cnf, hmm_seq_len_left, fix_order=False):
    end_with_keyword = False
    end_with_partial_keyword = False

    next_seqs, next_cnfs = [], []
    for keyword in keywords:
        for j in range(1, len(keyword)):
            # append next_seqs that finish partial keywords
            if end_with(prefix, keyword[:j]) and len(keyword[j:]) <= hmm_seq_len_left:
                next_seqs.append(keyword[j:])
                next_cnfs.append(remove_from_cnf(cnf, keyword, fix_order=fix_order))
                end_with_with_parital_keyword = True
        if end_with(prefix, keyword):
            end_with_keyword = True

    if end_with_keyword or (not end_with_partial_keyword):
        for i in range(0, len(prefix)):
            if prefix[i:] in keywords:
                cnf = remove_from_cnf(cnf, prefix[i:], fix_order=fix_order)
        # keywords_cnf = set([keyword for clause in cnf for keyword in clause])
        for keyword in keywords:
            if len(keyword) <= hmm_seq_len_left:
                next_seqs.append(keyword)
                next_cnfs.append(remove_from_cnf(cnf, keyword, fix_order=fix_order))

    return next_seqs, next_cnfs, end_with_keyword, end_with_partial_keyword


class HMM(nn.Module):
    def __init__(self, weights_file, sep_tokens=[]):
        super().__init__()

        assert(weights_file[-2:] == 'th')
        
        d = torch.load(weights_file)
        alpha, beta, gamma = d['alpha'], d['beta'], d['gamma']

        alpha = torch.log_softmax(alpha, dim=1)
        beta = torch.log_softmax(beta, dim=1)
        gamma = torch.log_softmax(gamma, dim=0)

        self.alpha = nn.Parameter(alpha, requires_grad=False)
        self.beta = nn.Parameter(beta, requires_grad=False)
        self.gamma = nn.Parameter(gamma, requires_grad=False)

        self.cache = {}

        self.cache['sep_tokens'] = set(sep_tokens)


    def forward(self, x):
        device = x.device
        alpha, beta, gamma = self.alpha, self.beta, self.gamma

        batch_size, seq_len = x.shape
        hidden_states, vocab_size = beta.shape

        y = torch.zeros(batch_size, hidden_states).to(device)
        for t in range(seq_len - 1, -1, -1):
            if t != seq_len - 1:
                y = torch.logsumexp(alpha.unsqueeze(0) + y.unsqueeze(1), dim=2)
            inputs = beta[torch.arange(hidden_states).unsqueeze(0).to(device),
                x[:, t].unsqueeze(-1)] # batch_size * hidden_states
            y = y + inputs
        y = torch.logsumexp(gamma.unsqueeze(0) + y, dim=1)

        return y


    def initialize_cache(self, hmm_seq_len, cnf0,
            prompt_tokens=(), batch_size=256, fix_order=False):

        self.cache = {'sep_tokens': self.cache['sep_tokens']}

        torch.cuda.empty_cache()

        device = self.alpha.device
        inf, eos_token_id = 1e10, 50256
        hidden_states, vocab_size = self.beta.shape
        alpha, beta, gamma = self.alpha, self.beta, self.gamma

        beta = beta.clone()
        beta[:, 796:797] = -inf

        # compute start_tokens # beginning-of-keyword tokens
        sep_tokens = self.cache['sep_tokens']
        start_tokens = set([keyword[0] for clause in cnf0 for keyword in clause])
        self.cache['start_tokens'] = start_tokens

        sep_non_start_tokens = list(sep_tokens.difference(start_tokens))
        beta_sep_non_start_mars = torch.logsumexp(beta[:, sep_non_start_tokens], dim=1) # hidden_states

        non_sep_tokens = [token for token in range(0, vocab_size) if token not in sep_tokens]
        beta_non_sep_mars = torch.logsumexp(beta[:, non_sep_tokens], dim=1) # hidden_states

        non_start_tokens = [token for token in range(0, vocab_size) if token not in start_tokens]
        beta_non_start_mars = torch.logsumexp(beta[:, non_start_tokens], dim=1) # hidden_states

        # initialize cache A
        A_cache = {(): self.gamma.clone()}
        self.cache['A'] = A_cache
        for i in range(1, len(prompt_tokens)):
            self.update_A([prompt_tokens[:i]])

        # initialize cache C
        C_cache = {}
        C = torch.eye(hidden_states, device=device)

        alpha_exp, beta_exp = torch.exp(alpha), torch.exp(beta)
        keywords = list(set([keyword for clause in cnf0 for keyword in clause]))
        max_keyword_len = max([len(keyword) for keyword in keywords])

        C = C.unsqueeze(0) # 1 * hidden_states * hidden_states
        for suffix_len in range(1, max_keyword_len+1):
            input_probs = [beta_exp[:, x[-suffix_len]] if len(x) >= suffix_len
                else torch.ones(hidden_states, device=device) for x in keywords]
            input_probs = torch.stack(input_probs, dim=0) # len(keywords) * hidden_states
            C = input_probs.unsqueeze(-1) * torch.matmul(alpha_exp.unsqueeze(0), C)
            for i, keyword in enumerate(keywords):
                if len(keyword) >= suffix_len:
                    C_cache[keyword[-suffix_len:]] = torch.log(C[i])

        # initialize cache B and B_sep
        B_cache, B_sep_cache = {}, {}
        for subset_size in range(0, len(cnf0)+1):
            for subset in itertools.combinations(cnf0, subset_size):
                B_cache[tuple(subset)] = -inf * torch.ones(hmm_seq_len+1, hidden_states, device=device)
                B_sep_cache[tuple(subset)] = -inf * torch.ones(hmm_seq_len+1, hidden_states, device=device)
        B_cache[()][hmm_seq_len, :] = 0.0
        B_sep_cache[()][hmm_seq_len, :] = 0.0

        all_subsets = [subset for subset in B_cache]
        subset_batch_size = max(batch_size // len(keywords), 1)
        for t in range(hmm_seq_len-1, -1, -1):
            for subset_batch_idx in range(0, len(all_subsets), subset_batch_size):
                subset_batch_size_ = min(subset_batch_size, len(all_subsets) - subset_batch_idx)
                subset_batch = all_subsets[subset_batch_idx: subset_batch_idx + subset_batch_size_]

                # case 1: first token is sep and start
                C, B = [], []
                for subset in subset_batch:
                    C_subset, B_subset = [], []
                    for keyword in keywords:
                        if t + len(keyword) <= hmm_seq_len:
                            next_subset = remove_from_cnf(subset, keyword, fix_order=fix_order)
                            C_subset.append(C_cache[keyword])
                            B_subset.append(B_sep_cache[next_subset][t+len(keyword), :])
                        else:   # probability 0
                            C_subset.append(torch.eye(hidden_states, device=device))
                            B_subset.append(-inf * torch.ones(hidden_states, device=device))
                    C_subset = torch.stack(C_subset, dim=0) # len(keywords) * hidden_states * hidden_states
                    B_subset = torch.stack(B_subset, dim=0) # len(keywords) * hidden_states
                    C.append(C_subset)
                    B.append(B_subset)

                C = torch.stack(C, dim=0) # subset_num * len(keywords) * hidden_states * hidden_states
                B = torch.stack(B, dim=0) # subset_num * len(keywords) * hidden_states
                C += B.unsqueeze(2)
                CB = torch.logsumexp(C, dim=3) # subset_num * len(keywords) * hidden_states
                CB = torch.logsumexp(CB, dim=1) # subset_num * hidden_states

                B = torch.stack([B_cache[subset][t+1, :] for subset in subset_batch], dim=0) # subset_num * hidden_states
                B = torch.logsumexp(alpha.unsqueeze(0) + B.unsqueeze(1), dim=2) # subset_num * hidden_states

                B1 = beta_sep_non_start_mars.unsqueeze(0) + B # subset_num * hidden_states
                B2 = beta_non_sep_mars.unsqueeze(0) + B # subset_num * hidden_states
                B_sep = torch.logaddexp(CB, B1)
                B = torch.logaddexp(B_sep, B2)

                for i, subset in enumerate(subset_batch):
                    B_cache[subset][t, :] = B[i]
                    B_sep_cache[subset][t, :] = B_sep[i]

        self.cache['A'], self.cache['C'] = A_cache, C_cache
        self.cache['B'], self.cache['B_sep'] = B_cache, B_sep_cache


    def update_A(self, prefixes):
        A_cache = self.cache['A']
        A = torch.stack([A_cache[prefix[:-1]] for prefix in prefixes], dim=0) # len(prefixes) * hidden_states
        log_probs = torch.stack([self.beta[:, prefix[-1]] for prefix in prefixes], dim=0) # len(prefixes) * hidden_states
        alpha_t = torch.transpose(self.alpha, 0, 1).unsqueeze(0) # 1 * hidden_states * hidden_states
        A = torch.logsumexp(alpha_t + (A + log_probs).unsqueeze(1), dim=2) # len(prefixes) * hidden_states

        for i, prefix in enumerate(prefixes):
            A_cache[prefix] = A[i]


    # compute logits for next_token:
    # return Pr(prefix, next_token, cnf), Pr(prefix, next_token)
    # here we can assume all prefixes are of the same length
    def compute_logits(self, prefixes, cnf0,
            seq_len, prompt_len, seq2seq, early_stop=False, fix_order=False):
        inf = 1e10

        device = self.alpha.device
        neginf_cuda = -inf * torch.ones(1, device=device)
        eos_token_id = 50256
        hidden_states, vocab_size = self.beta.shape
        alpha, beta, gamma = self.alpha, self.beta, self.gamma

        # beta = beta.clone()
        # beta[:, 796:797] = neginf_cuda

        keywords0 = set([keyword for clause in cnf0 for keyword in clause])

        sep_tokens, start_tokens = self.cache['sep_tokens'], self.cache['start_tokens']

        sep_non_start_tokens_set = sep_tokens.difference(start_tokens)
        sep_non_start_mask = torch.tensor([(0.0 if token in sep_non_start_tokens_set else -inf)
            for token in range(0, vocab_size)], device=device).unsqueeze(0)
        beta_sep_non_start = beta + sep_non_start_mask

        non_start_mask = torch.tensor([(-inf if token in start_tokens else 0.0)
            for token in range(0, vocab_size)], device=device).unsqueeze(0)
        beta_non_start = beta + non_start_mask
        aib = torch.zeros(hidden_states, vocab_size, device=device)

        A_cache, C_cache = self.cache['A'], self.cache['C']
        B_cache, B_sep_cache = self.cache['B'], self.cache['B_sep']

        if seq2seq == 1:
            prefixes = [prefix[prompt_len:] for prefix in prefixes]

        prefix_len = len(prefixes[0])
        if prefix_len > 0:
            self.update_A(prefixes)

        logits_alpha, logits_unconditioned = [], []
        for prefix in prefixes:
            hmm_seq_len_left = seq_len - prefix_len
            if seq2seq == 2:
                prefix_non_prompt = prefix[prompt_len:]
            else:
                prefix_non_prompt = prefix

            # remove clauses that are already satisfied by prefix
            cnf = update_cnf(prefix_non_prompt, cnf0,
                sep_tokens_set=sep_tokens, fix_order=fix_order)

            if early_stop and len(cnf) == 0:
                logits_alpha.append(torch.zeros(vocab_size, device=device))
                logits_unconditioned.append(torch.zeros(vocab_size, device=device))
                continue

            # case analysis
            next_seqs, next_cnfs, end_with_keyword, end_with_partial_keyword = case_analysis(
                prefix_non_prompt, keywords0, cnf, hmm_seq_len_left, fix_order=fix_order)

            logits = -inf * torch.ones(vocab_size, device=device)

            if len(next_seqs) > 0:
                A = A_cache[prefix] # hidden_states
                C, B = [], []
                for next_seq, next_cnf in zip(next_seqs, next_cnfs):
                    C.append(C_cache[next_seq])
                    B.append(B_sep_cache[next_cnf][prefix_len+len(next_seq), :])
                C = torch.stack(C, dim=0) # len(next_seqs) * hidden_states * hidden_states
                B = torch.stack(B, dim=0) # len(next_seqs) * hidden_states

                log_probs = torch.logsumexp(A.unsqueeze(0) + torch.logsumexp(C + B.unsqueeze(1), dim=2), dim=1) # len(next_seqs)

                for i, seq in enumerate(next_seqs):
                    logits[seq[0]:seq[0]+1] = torch.logaddexp(logits[seq[0]:seq[0]+1], log_probs[i:i+1])

            if end_with_keyword or (not end_with_partial_keyword):
                inputs = beta_sep_non_start if end_with_keyword else beta_non_start # hidden_states * vocab_size
                if end_with_keyword:
                    for i in range(0, len(prefix_non_prompt)):
                        if prefix_non_prompt[i:] in keywords0:
                            cnf = remove_from_cnf(cnf, prefix_non_prompt[i:], fix_order=fix_order)

                a = A_cache[prefix] # hidden_states
                b = B_cache[cnf][prefix_len+1, :] # hidden_states
                aib = a.unsqueeze(-1) + inputs + torch.logsumexp(alpha + b.unsqueeze(0), dim=1).unsqueeze(-1)
                aib = torch.logsumexp(aib, dim=0)
                logits = torch.logaddexp(logits, aib)

            if len(cnf) > 0:
                logits[eos_token_id:eos_token_id+1] += -inf * torch.ones(1, device=device)

            logits[796:797] = neginf_cuda
            logits_alpha.append(logits)

            logits_ = torch.logsumexp(A_cache[prefix].unsqueeze(-1) + beta, dim=0)
            logits_unconditioned.append(logits_)

        logits_alpha = torch.stack(logits_alpha, dim=0)
        logits_unconditioned = torch.stack(logits_unconditioned, dim=0)

        return logits_alpha, logits_unconditioned
