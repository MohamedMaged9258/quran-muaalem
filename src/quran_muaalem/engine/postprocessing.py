import torch
from ray import serve

from ..modeling.multi_level_tokenizer import MultiLevelTokenizer
from ..modeling.vocab import PAD_TOKEN_IDX
from ..decode import ctc_decode


# def ctc_decode(
#     batch_ids: torch.LongTensor,
#     batch_probs: torch.FloatTensor,
#     blank_id: int = PAD_TOKEN_IDX,
#     collapse_consecutive: bool = True,
# ):
#     outs = []
#     for seq_idx in range(batch_ids.shape[0]):
#         seq = batch_ids[seq_idx]
#         probs_seq = batch_probs[seq_idx]
#
#         if collapse_consecutive:
#             tokens = []
#             probs = []
#             start = 0
#             end = 0
#
#             if len(seq) == 1 and seq[0] != blank_id:
#                 tokens.append(seq[0])
#                 probs.append(probs_seq[0])
#
#             for idx in range(len(seq) - 1):
#                 curr = seq[idx]
#                 next_tok = seq[idx + 1]
#
#                 if idx == len(seq) - 2 and curr != blank_id:
#                     if curr == next_tok:
#                         end = idx + 2
#                         tokens.append(curr)
#                         probs.append(probs_seq[start:end].sum() / (end - start))
#                     elif curr != next_tok:
#                         end = idx + 1
#                         tokens.append(curr)
#                         probs.append(probs_seq[start:end].sum() / (end - start))
#                         tokens.append(next_tok)
#                         probs.append(probs_seq[idx + 1])
#                 elif curr != next_tok and curr != blank_id:
#                     end = idx + 1
#                     tokens.append(curr)
#                     probs.append(probs_seq[start:end].sum() / (end - start))
#                     start = end
#                 elif curr == blank_id:
#                     start = idx + 1
#
#             outs.append({"ids": tokens, "probs": probs})
#         else:
#             mask = seq != blank_id
#             outs.append({"ids": seq[mask].tolist(), "probs": probs_seq[mask].tolist()})
#
#     return outs


@serve.deployment(
    name="postprocessing",
    ray_actor_options={"num_cpus": 2},
)
class PostProcessingDeployment:
    def __init__(self, model_name_or_path: str = "obadx/muaalem-model-v3_2"):
        self.multi_level_tokenizer = MultiLevelTokenizer(model_name_or_path)

    def __call__(self, level_to_logits: dict[str, torch.Tensor]) -> dict:
        level_to_probs = {}
        for level, logits in level_to_logits.items():
            probs = torch.nn.functional.softmax(logits, dim=-1)
            level_to_probs[level] = probs

        phonemes_probs = level_to_probs["phonemes"]
        batch_probs, batch_ids = phonemes_probs.topk(1, dim=-1)
        return batch_ids.tolist()  # TODO:

        # TODO:
        # phonemes_decoded = ctc_decode(
        #     batch_ids.squeeze(-1), batch_probs.squeeze(-1), collapse_consecutive=True
        # )
        #
        # phonemes_text = ""
        # for idx in phonemes_decoded[0]["ids"]:
        #     phonemes_text += self.multi_level_tokenizer.id_to_vocab["phonemes"][
        #         int(idx)
        #     ]
        #
        # result = {"phonemes": phonemes_text}
        #
        # for level in level_to_probs:
        #     if level == "phonemes":
        #         continue
        #
        #     probs = level_to_probs[level]
        #     batch_probs, batch_ids = probs.topk(1, dim=-1)
        #     decoded = ctc_decode(
        #         batch_ids.squeeze(-1),
        #         batch_probs.squeeze(-1),
        #         collapse_consecutive=True,
        #     )
        #
        #     level_key = level.replace("-", "_")
        #     result[level_key] = decoded[0]["ids"]
        #
        # return result
