from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

import lotus
from lotus.templates import task_instructions

from .postprocessors import filter_postprocess


def sem_filter(
    docs: List[str],
    model: lotus.models.LM,
    user_instruction: str,
    default: bool = True,
    examples_df_txt: Optional[str] = None,
    examples_answers: Optional[List[bool]] = None,
    cot_reasoning: Optional[List[str]] = None,
    strategy: Optional[str] = None,
    logprobs: bool = False,
) -> Tuple:
    """
    Filters a list of documents based on a given user instruction using a language model.

    Args:
        docs (List[str]): The list of documents to filter.
        model (lotus.models.LM): The language model used for filtering.
        user_instruction (str): The user instruction for filtering.
        default (Optional[bool]): The default value for filtering in case of parsing errors. Defaults to True.
        examples_df_txt (Optional[str]: The text for examples. Defaults to None.
        examples_answers (Optional[List[bool]]): The answers for examples. Defaults to None.
        cot_reasoning (Optional[List[str]]): The reasoning for CoT. Defaults to None.
        logprobs (Optional[bool]): Whether to return log probabilities. Defaults to False.

    Returns:
        Tuple: A tuple containing the True/False outputs, raw outputs, explanations, and raw log probabilities (if logprobs=True).
    """
    inputs = []
    for doc in docs:
        prompt = lotus.templates.task_instructions.filter_formatter(
            doc, user_instruction, examples_df_txt, examples_answers, cot_reasoning, strategy
        )
        lotus.logger.debug(f"input to model: {prompt}")
        inputs.append(prompt)
    res = model(inputs, logprobs=logprobs)
    if logprobs:
        raw_outputs, raw_logprobs = res
    else:
        raw_outputs = res

    outputs, explanations = filter_postprocess(
        raw_outputs, default=default, cot_reasoning=strategy in ["cot", "zs-cot"]
    )
    lotus.logger.debug(f"outputs: {outputs}")
    lotus.logger.debug(f"raw_outputs: {raw_outputs}")
    lotus.logger.debug(f"explanations: {explanations}")

    if logprobs:
        return outputs, raw_outputs, explanations, raw_logprobs
    return outputs, raw_outputs, explanations

async def sem_filter_async(
    docs: List[str],
    model: lotus.models.LM,
    user_instruction: str,
    default: bool = True,
    examples_df_txt: Optional[str] = None,
    examples_answers: Optional[List[bool]] = None,
    cot_reasoning: Optional[List[str]] = None,
    strategy: Optional[str] = None,
    logprobs: bool = False,
) -> Tuple:
    """
    Asynchronously filters a list of documents based on a given user instruction using a language model.

    Args:
        docs (List[str]): The list of documents to filter.
        model (lotus.models.LM): The language model used for filtering.
        user_instruction (str): The user instruction for filtering.
        default (Optional[bool]): The default value for filtering in case of parsing errors. Defaults to True.
        examples_df_txt (Optional[str]): The text for examples. Defaults to None.
        examples_answers (Optional[List[bool]]): The answers for examples. Defaults to None.
        cot_reasoning (Optional[List[str]]): The reasoning for CoT. Defaults to None.
        logprobs (Optional[bool]): Whether to return log probabilities. Defaults to False.

    Returns:
        Tuple: A tuple containing the True/False outputs, raw outputs, explanations, and raw log probabilities (if logprobs=True).
    """
    inputs = []
    for doc in docs:
        prompt = lotus.templates.task_instructions.filter_formatter(
            doc, user_instruction, examples_df_txt, examples_answers, cot_reasoning, strategy
        )
        lotus.logger.debug(f"input to model: {prompt}")
        inputs.append(prompt)

    res = await model.generate_async(inputs, logprobs=logprobs)  # Async model call
    if logprobs:
        raw_outputs, raw_logprobs = res
    else:
        raw_outputs = res

    outputs, explanations = filter_postprocess(
        raw_outputs, default=default, cot_reasoning=strategy in ["cot", "zs-cot"]
    )
    lotus.logger.debug(f"outputs: {outputs}")
    lotus.logger.debug(f"raw_outputs: {raw_outputs}")
    lotus.logger.debug(f"explanations: {explanations}")

    if logprobs:
        return outputs, raw_outputs, explanations, raw_logprobs
    return outputs, raw_outputs, explanations

class BaseSemFilterDataframe:
    """Base class for semantic filter operations."""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """Validate that the object is a DataFrame."""
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def _prepare_data(
        self,
        user_instruction: str,
        examples: Optional[pd.DataFrame] = None,
        strategy: Optional[str] = None,
    ):
        """Prepares the data required for filtering."""
        stats = {}
        lotus.logger.debug(user_instruction)
        col_li = lotus.nl_expression.parse_cols(user_instruction)
        lotus.logger.debug(col_li)

        # Check that columns exist
        for column in col_li:
            if column not in self._obj.columns:
                raise ValueError(f"Column {column} not found in DataFrame")

        df_txt = task_instructions.df2text(self._obj, col_li)
        lotus.logger.debug(df_txt)
        formatted_usr_instr = lotus.nl_expression.nle2str(user_instruction, col_li)

        examples_df_txt = None
        examples_answers = None
        cot_reasoning = None
        if examples is not None:
            assert "Answer" in examples.columns, "Answer must be a column in examples dataframe"
            examples_df_txt = task_instructions.df2text(examples, col_li)
            examples_answers = examples["Answer"].tolist()

            if strategy == "cot":
                cot_reasoning = examples["Reasoning"].tolist()

        return df_txt, formatted_usr_instr, examples_df_txt, examples_answers, cot_reasoning, stats
    
    def _process_outputs(
        self,
        outputs: List[bool],
        raw_outputs: List[str],
        explanations: List[str],
        return_raw_outputs: bool,
        return_explanations: bool,
        stats: Dict[str, Any],
        suffix: str,
        return_stats: bool,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Processes the outputs from the filter function and returns a filtered dataframe."""
        
        # Find indices where output is True
        ids = [i for i, x in enumerate(outputs) if x]
        idx_ids = [self._obj.index[i] for i, x in enumerate(outputs) if x]
        lotus.logger.debug(f"ids: {ids}")
        lotus.logger.debug(f"idx_ids: {idx_ids}")

        filtered_explanations = [explanations[i] for i in ids]
        filtered_raw_outputs = [raw_outputs[i] for i in ids]

        new_df = self._obj.iloc[ids]
        new_df.attrs["index_dirs"] = self._obj.attrs.get("index_dirs", None)

        # Return rows where output is True
        if return_explanations and return_raw_outputs:
            new_df["explanation" + suffix] = filtered_explanations
            new_df["raw_output" + suffix] = filtered_raw_outputs
        elif return_explanations:
            new_df["explanation" + suffix] = filtered_explanations
        elif return_raw_outputs:
            new_df["raw_output" + suffix] = filtered_raw_outputs

        if return_stats:
            return new_df, stats

        return new_df

@pd.api.extensions.register_dataframe_accessor("sem_filter")
class SemFilterDataframe(BaseSemFilterDataframe):
    """DataFrame accessor for synchronous semantic filter."""
    def __call__(
        self,
        user_instruction: str,
        return_raw_outputs: bool = False,
        return_explanations: bool = False,
        default: bool = True,
        suffix: str = "_filter",
        examples: Optional[pd.DataFrame] = None,
        strategy: Optional[str] = None,
        return_stats: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Applies semantic filter over a dataframe (sync)."""
        
        # Prepare the data
        df_txt, formatted_usr_instr, examples_df_txt, examples_answers, cot_reasoning, stats = self._prepare_data(
            user_instruction, examples, strategy
        )

        # Call the sync sem_filter function
        outputs, raw_outputs, explanations = sem_filter(
            df_txt,
            lotus.settings.lm,
            formatted_usr_instr,
            default=default,
            examples_df_txt=examples_df_txt,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
        )

        return self._process_outputs(outputs, raw_outputs, explanations, return_raw_outputs, return_explanations, stats, suffix, return_stats)

@pd.api.extensions.register_dataframe_accessor("sem_filter_async")
class SemFilterAsyncDataframe(BaseSemFilterDataframe):
    """DataFrame accessor for asynchronous semantic filter."""

    async def __call__(
        self,
        user_instruction: str,
        return_raw_outputs: bool = False,
        return_explanations: bool = False,
        default: bool = True,
        suffix: str = "_filter",
        examples: Optional[pd.DataFrame] = None,
        strategy: Optional[str] = None,
        return_stats: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Applies semantic filter over a dataframe (async)."""

        df_txt, formatted_usr_instr, examples_df_txt, examples_answers, cot_reasoning, stats = self._prepare_data(
            user_instruction, examples, strategy
        )

        outputs, raw_outputs, explanations = await sem_filter_async(
            df_txt,
            lotus.settings.lm,
            formatted_usr_instr,
            default=default,
            examples_df_txt=examples_df_txt,
            examples_answers=examples_answers,
            cot_reasoning=cot_reasoning,
            strategy=strategy,
        )

        return self._process_outputs(outputs, raw_outputs, explanations, return_raw_outputs, return_explanations, stats, suffix, return_stats)