import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from reproduce.eval_ppl import evaluate_perplexity, iter_windows
from reproduce.prune_wanda import prune_weight
from reproduce.recover_masked import mask_prompt_labels, restore_masks


@pytest.mark.parametrize("length,stride", [(2, 1), (9, 3), (10, 3), (17, 4), (4, 1)])
def test_ppl_scores_every_causal_target_once(length, stride):
    windows = list(iter_windows(length, max_len=6, stride=stride))
    assert [i for _, end, start in windows for i in range(start, end)] == list(range(1, length))
    class Model:
        def __call__(self, ids, labels, use_cache):
            assert (labels[:, 1:] != -100).any()
            return SimpleNamespace(loss=torch.tensor(math.log(2)))
    ppl, count = evaluate_perplexity(Model(), torch.arange(length)[None, :], 6, stride, "cpu")
    assert count == length - 1
    assert ppl == pytest.approx(2)


@pytest.mark.parametrize("sparsity,zeros", [(0, 0), (0.3, 3), (1, 10)])
def test_wanda_ties_have_exact_cardinality(sparsity, zeros):
    weights = torch.ones(2, 10)
    result = prune_weight(weights, torch.ones(10), sparsity)
    assert ((result == 0).sum(dim=1) == zeros).all()
    assert (weights == 1).all()


def test_semi_structured_ties_and_invalid_width():
    result = prune_weight(torch.ones(2, 8), torch.ones(8), 0.5, semi24=True)
    assert ((result.reshape(2, 2, 4) == 0).sum(-1) == 2).all()
    with pytest.raises(ValueError, match="divisible"):
        prune_weight(torch.ones(2, 6), torch.ones(6), 0.5, semi24=True)


def test_masked_recovery_rejects_empty_supervision():
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="no answer tokens"):
        mask_prompt_labels(ids, 3)
    assert mask_prompt_labels(ids, 2).tolist() == [[-100, -100, 3]]


def test_restore_masks_enforces_sparsity_and_layer_identity():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    mask = torch.tensor([[True, False, False], [False, True, False]])
    restore_masks(model, {"0": mask})
    assert (model[0].weight[mask] == 0).all()
    with pytest.raises(ValueError, match="missing masked layers"):
        restore_masks(model, {"missing": mask})


def test_sparsegpt_preserves_real_qwen_forward_at_zero_sparsity():
    transformers = pytest.importorskip("transformers")
    from third_party_adapters.sparsegpt_qwen import qwen_sequential
    torch.manual_seed(0)
    config = transformers.Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                     num_hidden_layers=2, num_attention_heads=2,
                                     num_key_value_heads=1, max_position_embeddings=32)
    config._attn_implementation = "eager"
    model = transformers.Qwen2ForCausalLM(config).eval()
    model.seqlen = 8
    tokens = torch.randint(0, 32, (1, 8))
    with torch.no_grad():
        before = model(tokens).logits.clone()
    batches = []
    class NoPrune:
        def __init__(self, module):
            self.module = module
        def add_batch(self, inputs, outputs):
            batches.append(inputs.shape)
        def fasterprune(self, *args, **kwargs):
            pass
        def free(self):
            pass
    options = SimpleNamespace(nsamples=1, sparsity=0, prunen=0, prunem=0, percdamp=.01, blocksize=16)
    qwen_sequential(model, [(tokens, None)], torch.device("cpu"), options, NoPrune)
    with torch.no_grad():
        after = model(tokens).logits
    assert batches
    assert torch.equal(before, after)
    assert model.config.use_cache is True
