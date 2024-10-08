from typing import List, Tuple, Dict

import pandas as pd

import lotus
from lotus.templates import task_instructions

def format_doc(tree_level: int, doc: str, ctr: int) -> str:
    """Formats a document based on its level in the aggregation tree."""
    if tree_level == 0:
        return f"\n\tDocument {ctr}: {doc}"
    return f"\n\tSource {ctr}: {doc}"

def generate_prompt(template: str, docs_str: str, user_instruction: str) -> str:
    """Generates the prompt by replacing placeholders in the template."""
    return template.replace("{{docs_str}}", docs_str).replace("{{user_instruction}}", user_instruction)

def batch_documents(
    docs: List[str],
    partition_ids: List[int],
    model: lotus.models.LM,
    template: str,
    user_instruction: str,
    tree_level: int,
) -> Tuple[List[List[Dict[str, str]]], List[int]]:
    """Batches documents based on token limits and partitioning rules."""
    batch = []
    new_partition_ids = []
    context_str = ""
    context_tokens = 0
    doc_ctr = 1
    cur_partition_id = partition_ids[0]

    for idx, doc in enumerate(docs):
        partition_id = partition_ids[idx]
        formatted_doc = format_doc(tree_level, doc, doc_ctr)
        new_tokens = model.count_tokens(formatted_doc)

        if (new_tokens + context_tokens + model.count_tokens(template) > model.max_ctx_len - model.max_tokens) or (
            partition_id != cur_partition_id
        ):
            prompt = generate_prompt(template, context_str, user_instruction)
            batch.append([{"role": "user", "content": prompt}])
            new_partition_ids.append(cur_partition_id)
            cur_partition_id = partition_id
            doc_ctr = 1

            formatted_doc = format_doc(tree_level, docs[idx], doc_ctr)
            context_str = formatted_doc
            context_tokens = new_tokens
        else:
            context_str += formatted_doc
            context_tokens += new_tokens
            doc_ctr += 1

    if context_str:
        prompt = generate_prompt(template, context_str, user_instruction)
        batch.append([{"role": "user", "content": prompt}])
        new_partition_ids.append(cur_partition_id)

    return batch, new_partition_ids

class BaseAggregator:
    def __init__(self, model: lotus.models.LM):
        self.model = model

    def get_template(self, tree_level: int) -> str:
        """Returns the appropriate instruction template for the aggregation."""
        if tree_level == 0:
            return (
                "Your job is to provide an answer to the user's instruction given the context below from multiple documents.\n"
                "Remember that your job is to answer the user's instruction by combining all relevant information from all provided documents, into a single coherent answer.\n"
                "Do NOT copy the format of the sources! Instead output your answer in a coherent, well-structured manner that best answers the user instruction.\n"
                "You have limited space to provide your answer, so be concise and to the point.\n\n---\n\n"
                "Follow the following format.\n\nContext: relevant facts from multiple documents\n\n"
                "Instruction: {{user_instruction}}\n\nAnswer:\n"
            )
        else:
            return (
                "Your job is to provide an answer to the user's instruction given the context below from multiple sources.\n"
                "Note that each source may be formatted differently and contain information about several different documents.\n"
                "Remember that your job is to answer the user's instruction by combining all relevant information from all provided sources, into a single coherent answer.\n"
                "The sources may provide opposing viewpoints or complementary information.\n"
                "Be sure to include information from ALL relevant sources in your answer.\n"
                "Do NOT copy the format of the sources, instead output your answer in a coherent, well-structured manner that best answers the user instruction.\n"
                "You have limited space to provide your answer, so be concise and to the point.\n\n---\n\n"
                "Follow the following format.\n\nContext: relevant facts from multiple sources\n\n"
                "Instruction: {{user_instruction}}\n\nAnswer:\n"
            )

    def call_model(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        """This function will be overridden in sync/async versions to make model calls."""
        raise NotImplementedError("This method should be overridden in derived classes.")

class SyncAggregator(BaseAggregator):
    def aggregate(self, docs: List[str], user_instruction: str, partition_ids: List[int]) -> str:
        """Base aggregation function that handles batching and document aggregation."""
        tree_level = 0
        summaries = []
        new_partition_ids = partition_ids
        template = self.get_template(tree_level)
        
        while len(docs) > 1 or not summaries:
            batch, new_partition_ids = batch_documents(docs, partition_ids, self.model, template, user_instruction, tree_level)
            summaries = self.call_model(batch)
            docs = summaries
            partition_ids = new_partition_ids
            tree_level += 1
            template = self.get_template(tree_level)

        return summaries[0]

    def call_model(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        """Calls the model synchronously and returns the results."""
        return self.model(batch)

class AsyncAggregator(BaseAggregator):
    async def aggregate(self, docs: List[str], user_instruction: str, partition_ids: List[int]) -> str:
        """Base aggregation function that handles batching and document aggregation."""
        tree_level = 0
        summaries = []
        new_partition_ids = partition_ids
        template = self.get_template(tree_level)
        
        while len(docs) > 1 or not summaries:
            batch, new_partition_ids = batch_documents(docs, partition_ids, self.model, template, user_instruction, tree_level)
            summaries = await self.call_model(batch)
            docs = summaries
            partition_ids = new_partition_ids
            tree_level += 1
            template = self.get_template(tree_level)

        return summaries[0]

    async def call_model(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        """Calls the model asynchronously and returns the results."""
        return await self.model.generate_async(batch)

@pd.api.extensions.register_dataframe_accessor("sem_agg")
class SemAggDataframe:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self, user_instruction: str, all_cols: bool = False, suffix: str = "_output") -> pd.DataFrame:
        aggregator = SyncAggregator(lotus.settings.lm)
        col_li = list(self._obj.columns) if all_cols else lotus.nl_expression.parse_cols(user_instruction)

        if "_lotus_partition_id" in self._obj.columns:
            partition_ids = self._obj["_lotus_partition_id"].tolist()
        else:
            partition_ids = [0] * len(self._obj)

        df_txt = task_instructions.df2text(self._obj, col_li)
        answer = aggregator.aggregate(df_txt, user_instruction, partition_ids)

        return pd.DataFrame([answer], columns=[suffix])

@pd.api.extensions.register_dataframe_accessor("sem_agg_async")
class SemAggAsyncDataframe:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    async def __call__(self, user_instruction: str, all_cols: bool = False, suffix: str = "_output") -> pd.DataFrame:
        aggregator = AsyncAggregator(lotus.settings.lm)
        col_li = list(self._obj.columns) if all_cols else lotus.nl_expression.parse_cols(user_instruction)

        if "_lotus_partition_id" in self._obj.columns:
            partition_ids = self._obj["_lotus_partition_id"].tolist()
        else:
            partition_ids = [0] * len(self._obj)

        df_txt = task_instructions.df2text(self._obj, col_li)
        answer = await aggregator.aggregate(df_txt, user_instruction, partition_ids)

        return pd.DataFrame([answer], columns=[suffix])
