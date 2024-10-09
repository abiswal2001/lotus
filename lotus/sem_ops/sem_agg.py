from typing import List, Tuple, Dict

import pandas as pd

import lotus
from lotus.templates import task_instructions

def leaf_doc_formatter(doc, ctr):
        return f"\n\tDocument {ctr}: {doc}"

def node_doc_formatter(doc, ctr):
    return f"\n\tSource {ctr}: {doc}"

def doc_formatter(tree_level, doc, ctr):
    return leaf_doc_formatter(doc, ctr) if tree_level == 0 else node_doc_formatter(doc, ctr)

class BaseAggregator:
    def __init__(self, model: lotus.models.LM):
        self.model = model

    def get_template(self, tree_level: int, user_instruction: str) -> str:
        """Returns the appropriate instruction template for the aggregation."""
        if tree_level == 0:
            return (
                "Your job is to provide an answer to the user's instruction given the context below from multiple documents.\n"
                "Remember that your job is to answer the user's instruction by combining all relevant information from all provided documents, into a single coherent answer.\n"
                "Do NOT copy the format of the sources! Instead output your answer in a coherent, well-structured manner that best answers the user instruction.\n"
                "You have limited space to provide your answer, so be concise and to the point.\n\n---\n\n"
                "Follow the following format.\n\nContext: relevant facts from multiple documents\n\n"
                "Instruction: the instruction provided by the user\n\nAnswer: Write your answer\n\n---\n\n"
                "Context: {{docs_str}}\n\n"
                f"Instruction:  {user_instruction}\n\nAnswer:\n"
            )
        else:
            return (
                "Your job is to provide an answer to the user's instruction given the context below from multiple sources.\n"
                "Note that each source may be formatted differently and contain information about several different documents.\n"
                "Remember that your job is to answer the user's instruction by combining all relevant information from all provided sources, into a single coherent answer.\n"
                "The sources may provide opposing viewpoints or complementary information.\n"
                "Be sure to include information from ALL relevant sources in your answer.\n"
                "Do NOT copy the format of the sources, instead output your answer in a coherent, well-structured manner that best answers the user instruction.\n"
                "You have limited space to provide your answer, so be concise and to the point.\n"
                "You may need to draw connections between sources to provide a complete answer.\n\n---\n\n"
                "Follow the following format.\n\nContext: relevant facts from multiple sources\n\n"
                "Instruction: the instruction provided by the user\n\nAnswer: Write your answer\n\n---\n\n"
                "Context: {{docs_str}}\n\n"
                f"Instruction:  {user_instruction}\n\nAnswer:\n"
            )

    def call_model(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        """This function will be overridden in sync/async versions to make model calls."""
        raise NotImplementedError("This method should be overridden in derived classes.")

class SyncAggregator(BaseAggregator):
    def aggregate(self, docs: List[str], user_instruction: str, partition_ids: List[int]) -> str:
        """Base aggregation function that handles batching and document aggregation."""
        tree_level = 0
        summaries = []
        new_partition_ids = []
        while len(docs) != 1 or summaries == []:
            cur_partition_id = partition_ids[0]
            do_fold = len(partition_ids) == len(set(partition_ids))
            context_str = ""
            # prompt = ""
            batch = []
            template = self.get_template(tree_level, user_instruction)
            template_tokens = self.model.count_tokens(template)
            context_tokens = 0
            doc_ctr = 1  # num docs in current prompt
            for idx in range(len(docs)):
                partition_id = partition_ids[idx]
                formatted_doc = doc_formatter(tree_level, docs[idx], doc_ctr)
                new_tokens = self.model.count_tokens(formatted_doc)

                if (new_tokens + context_tokens + template_tokens > self.model.max_ctx_len - self.model.max_tokens) or (
                    partition_id != cur_partition_id and not do_fold
                ):
                    # close the current prompt

                    prompt = template.replace("{{docs_str}}", context_str)
                    lotus.logger.debug(f"Prompt added to batch: {prompt}")
                    batch.append([{"role": "user", "content": prompt}])
                    new_partition_ids.append(cur_partition_id)
                    cur_partition_id = partition_id
                    doc_ctr = 1

                    # add new context to next prompt
                    formatted_doc = doc_formatter(tree_level, docs[idx], doc_ctr)
                    context_str = formatted_doc
                    context_tokens = new_tokens
                    doc_ctr += 1
                else:
                    context_str = context_str + formatted_doc
                    context_tokens += new_tokens
                    doc_ctr += 1

            if doc_ctr > 1 or len(docs) == 1:
                prompt = template.replace("{{docs_str}}", context_str)
                lotus.logger.debug(f"Prompt added to batch: {prompt}")
                batch.append([{"role": "user", "content": prompt}])
                new_partition_ids.append(cur_partition_id)
            summaries = self.call_model(batch)
            partition_ids = new_partition_ids
            new_partition_ids = []

            docs = summaries
            lotus.logger.debug(f"Model outputs from tree level {tree_level}: {summaries}")
            tree_level += 1

        return summaries[0]

    def call_model(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        """Calls the model synchronously and returns the results."""
        return self.model(batch)

class AsyncAggregator(BaseAggregator):
    async def aggregate(self, docs: List[str], user_instruction: str, partition_ids: List[int]) -> str:
        """Base aggregation function that handles batching and document aggregation."""
        tree_level = 0
        summaries = []
        new_partition_ids = []
        while len(docs) != 1 or summaries == []:
            cur_partition_id = partition_ids[0]
            do_fold = len(partition_ids) == len(set(partition_ids))
            context_str = ""
            # prompt = ""
            batch = []
            template = self.get_template(tree_level, user_instruction)
            template_tokens = self.model.count_tokens(template)
            context_tokens = 0
            doc_ctr = 1  # num docs in current prompt
            for idx in range(len(docs)):
                partition_id = partition_ids[idx]
                formatted_doc = doc_formatter(tree_level, docs[idx], doc_ctr)
                new_tokens = self.model.count_tokens(formatted_doc)

                if (new_tokens + context_tokens + template_tokens > self.model.max_ctx_len - self.model.max_tokens) or (
                    partition_id != cur_partition_id and not do_fold
                ):
                    # close the current prompt

                    prompt = template.replace("{{docs_str}}", context_str)
                    lotus.logger.debug(f"Prompt added to batch: {prompt}")
                    batch.append([{"role": "user", "content": prompt}])
                    new_partition_ids.append(cur_partition_id)
                    cur_partition_id = partition_id
                    doc_ctr = 1

                    # add new context to next prompt
                    formatted_doc = doc_formatter(tree_level, docs[idx], doc_ctr)
                    context_str = formatted_doc
                    context_tokens = new_tokens
                    doc_ctr += 1
                else:
                    context_str = context_str + formatted_doc
                    context_tokens += new_tokens
                    doc_ctr += 1

            if doc_ctr > 1 or len(docs) == 1:
                prompt = template.replace("{{docs_str}}", context_str)
                lotus.logger.debug(f"Prompt added to batch: {prompt}")
                batch.append([{"role": "user", "content": prompt}])
                new_partition_ids.append(cur_partition_id)
            summaries = await self.call_model(batch)
            partition_ids = new_partition_ids
            new_partition_ids = []

            docs = summaries
            lotus.logger.debug(f"Model outputs from tree level {tree_level}: {summaries}")
            tree_level += 1

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

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"column {column} not found in DataFrame. Given usr instruction: {user_instruction}")

        if "_lotus_partition_id" in self._obj.columns:
            partition_ids = self._obj["_lotus_partition_id"].tolist()
        else:
            partition_ids = [0] * len(self._obj)

        df_txt = task_instructions.df2text(self._obj, col_li)
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)
        answer = aggregator.aggregate(df_txt, formatted_usr_instr, partition_ids)

        return pd.DataFrame([answer], columns=[suffix])

@pd.api.extensions.register_dataframe_accessor("sem_agg_async")
class SemAggAsyncDataframe:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    async def __call__(self, user_instruction: str, all_cols: bool = False, suffix: str = "_output") -> pd.DataFrame:
        aggregator = AsyncAggregator(lotus.settings.lm)
        col_li = list(self._obj.columns) if all_cols else lotus.nl_expression.parse_cols(user_instruction)

        # check that column exists
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"column {column} not found in DataFrame. Given usr instruction: {user_instruction}")

        if "_lotus_partition_id" in self._obj.columns:
            partition_ids = self._obj["_lotus_partition_id"].tolist()
        else:
            partition_ids = [0] * len(self._obj)

        df_txt = task_instructions.df2text(self._obj, col_li)
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)
        answer = await aggregator.aggregate(df_txt, formatted_usr_instr, partition_ids)

        return pd.DataFrame([answer], columns=[suffix])
