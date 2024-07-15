import pandas as pd
import regex
from tqdm import tqdm
from chat import openai, time

def print_(text):
    debug = False
    if debug:
        print(text)
    return


def distinct(string, n=1):
    n_grams = [string[i:i+n] for i in range(len(string)-n+1)]
    distinct_n_grams = set(n_grams)
    return len(distinct_n_grams) / len(n_grams)


def auto_retry(func, *args):
    max_retry_times = 3
    e_info = None
    for _ in range(max_retry_times):
        try:
            outputs = func(*args)
            return outputs
        except Exception as e:
            print(e)
            e_info = str(e)
    raise Exception(e_info)


class PipeJudge:
    def __init__(self):
        from references_pipe import ReferencesPipe
        from article_pipe_judge import (
            ArticleInferencePipe,
            ArticleDeduplicatePipe,
            wrap_pipe_for_parallel,
            repetition_score
        )
        self.references_pipe = ReferencesPipe()
        self.article_inference_pipe_for_parallel = wrap_pipe_for_parallel(ArticleInferencePipe)()
        self.article_deduplicate_pipe = ArticleDeduplicatePipe()
        self.repetition_score = repetition_score
        self.result_keys = [
            "query", "references", "time_references",
            "article_draft", "time_article_draft",
            "article", "time_article",
            "exception", "dollars", "seconds"
        ]

    def __call__(self, query, references=None):
        start_num_tokens, start_time = openai.num_tokens, time.time()
        result = {key: None for key in self.result_keys}
        result["query"] = query
        print_(f"query:\n{query}")
        try:
            if references is None:
                # get references from baidu under the help of chatgpt
                references = auto_retry(self.references_pipe, query)
            print_(f"references:\n{references}")
            result["references"] = references
            result["time_references"] = time.time() - start_time

            # generate article draft with different temperature
            list_kwargs_for_parallel = [
                {
                    "query": query, "references": references, "temperature": temperature
                }
                for temperature in [0.3, 0.5, 1.0]
            ]
            list_articles_draft = self.article_inference_pipe_for_parallel(list_kwargs_for_parallel)

            # rerank the generated article drafts and select the best one
            for kwargs, article_draft in zip(list_kwargs_for_parallel, list_articles_draft):
                kwargs.update({
                    "references": "skip to show",
                    "article_draft": article_draft,
                    "distinct": distinct(article_draft, n=4) if article_draft is not None else -1
                })
                kwargs["priority"] = (
                    kwargs["article_draft"] is None,
                    self.repetition_score(kwargs["article_draft"]) if kwargs["article_draft"] is not None else (100, 100),
                    kwargs["temperature"],
                    -kwargs["distinct"]
                )  # lower is better
            list_kwargs_for_parallel = sorted(list_kwargs_for_parallel, key=lambda kwargs: kwargs["priority"])
            print_(f"list_kwargs_for_parallel:\n{list_kwargs_for_parallel}")
            best_result = list_kwargs_for_parallel[0]
            article_draft = best_result["article_draft"]
            print_(f"article_draft:\n{article_draft}")
            result["article_draft"] = article_draft
            result["time_article_draft"] = time.time() - start_time
            if article_draft is None:
                raise Exception("no valid article is generated")

            # deduplicate
            article = self.article_deduplicate_pipe(article_draft)
            print_(f"article:\n{article}")
            result["article"] = article
            result["time_article"] = time.time() - start_time

        except Exception as e:
            result["exception"] = str(e)

        # record time and dollars
        end_num_tokens, end_time = openai.num_tokens, time.time()
        num_tokens = end_num_tokens - start_num_tokens
        dollars = num_tokens / 1000 * 0.002
        seconds = end_time - start_time
        result.update({
            "dollars": dollars,
            "seconds": seconds
        })

        return result


class PipeSplits:
    def __init__(self):
        from references_pipe import ReferencesPipe
        from article_pipe_splits import (
            ArticleOutlinesInferencePipe,
            ArticleConditionalInferencePipe,
            ArticleDeduplicatePipe,
            wrap_pipe_for_parallel,
            repetition_score
        )
        self.references_pipe = ReferencesPipe()
        self.article_outlines_inference_pipe = ArticleOutlinesInferencePipe()
        self.article_conditional_inference_pipe_for_parallel = wrap_pipe_for_parallel(ArticleConditionalInferencePipe)()
        self.article_deduplicate_pipe = ArticleDeduplicatePipe()
        self.repetition_score = repetition_score
        self.result_keys = [
            "query", "references", "time_references",
            "indexed_causes", "indexed_methods", "time_outlines_inference",
            "indexed_outlines", "article_draft", "priority", "temperature", "time_article_draft",
            "article", "time_article_deduplicate",
            "exception", "dollars", "seconds"
        ]

    def __call__(self, query, outlines_type, references=None):
        start_num_tokens, start_time = openai.num_tokens, time.time()
        result = {key: None for key in self.result_keys}
        result["query"] = query
        print_(f"query:\n{query}")
        try:
            if references is None:
                # get references from baidu under the help of chatgpt
                references = auto_retry(self.references_pipe, query)
            print_(f"references:\n{references}")
            result["references"] = references
            result["time_references"] = time.time() - start_time

            # inference and rerank causes and methods
            indexed_causes, indexed_methods = auto_retry(self.article_outlines_inference_pipe, query, references)
            print_(f"indexed_causes, indexed_methods:\n{indexed_causes}\n{indexed_methods}")
            result["indexed_causes"], result["indexed_methods"] = indexed_causes, indexed_methods
            result["time_outlines_inference"] = time.time() - start_time

            # generate article draft given causes/methods with different temperature
            if outlines_type in ["病因类","病因分析"]:
                if len(indexed_causes) >= 5:
                    list_kwargs_for_parallel = [
                        {
                            "query": query, "references": references, "temperature": temperature,
                            "outlines_type": "病因分析", "indexed_outlines": indexed_causes
                        }
                        for temperature in [0.5, 0.8, 1.0]
                    ]
                else:
                    raise Exception(f"number of outlines is not enough: {len(indexed_causes)} < 5")
            elif outlines_type in ["方法类","治疗方法"]:
                if len(indexed_methods) >= 5:
                    list_kwargs_for_parallel = [
                        {
                            "query": query, "references": references, "temperature": temperature,
                            "outlines_type": "治疗方法", "indexed_outlines": indexed_methods
                        }
                        for temperature in [0.5, 0.8, 1.0]
                    ]
                else:
                    raise Exception(f"number of outlines is not enough: {len(indexed_methods)} < 5")
            else:
                raise NotImplementedError(f"outlines_type: {outlines_type}")
            list_articles_draft = self.article_conditional_inference_pipe_for_parallel(list_kwargs_for_parallel)

            # rerank the generated article drafts and select the best one
            for kwargs,article_draft in zip(list_kwargs_for_parallel,list_articles_draft):
                kwargs.update({
                    "references": "skip to show",
                    "article_draft": article_draft,
                    "distinct": distinct(article_draft, n=4) if article_draft is not None else -1
                })
                kwargs["priority"] = (
                    kwargs["article_draft"] is None,
                    "等" in kwargs["article_draft"].split("\n")[0][-3:] if kwargs["article_draft"] is not None else True,
                    len(regex.findall(r"[^\p{Script=Han}a-zA-Z\p{N}\p{P}\p{Z}\n]", kwargs["article_draft"]))>3 if kwargs["article_draft"] is not None else True,
                    self.repetition_score(kwargs["article_draft"]) if kwargs["article_draft"] is not None else (100,100),
                    max([len(_) for _ in kwargs["indexed_outlines"]]),
                    kwargs["temperature"],
                    -kwargs["distinct"]
                ) # lower is better
            list_kwargs_for_parallel = sorted(list_kwargs_for_parallel, key=lambda kwargs:kwargs["priority"])
            print_(f"list_kwargs_for_parallel:\n{list_kwargs_for_parallel}")
            assert len(list_kwargs_for_parallel) > 0
            best_result = list_kwargs_for_parallel[0]
            article_draft = best_result["article_draft"]
            indexed_outlines = best_result["indexed_outlines"]
            priority = best_result["priority"]
            print_(f"article_draft:\n{article_draft}")
            print_(f"indexed_outlines:\n{indexed_outlines}")
            print_(f"priority:\n{priority}")
            result["article_draft"] = article_draft
            result["indexed_outlines"] = indexed_outlines
            result["priority"] = priority
            result["temperature"] = best_result["temperature"]
            result["time_article_draft"] = time.time() - start_time
            if article_draft is None:
                raise Exception(f"all article draft fails")

            # deduplicate
            article = auto_retry(self.article_deduplicate_pipe, article_draft)
            print_(f"article:\n{article}")
            result["article"] = article
            result["time_article_deduplicate"] = time.time() - start_time

        except Exception as e:
            result["exception"] = str(e)

        # record time and dollars
        end_num_tokens, end_time = openai.num_tokens, time.time()
        num_tokens = end_num_tokens - start_num_tokens
        dollars = num_tokens / 1000 * 0.002
        seconds = end_time - start_time
        result.update({
            "dollars": dollars,
            "seconds": seconds
        })

        return result


class PipeEnsemble:
    def __init__(self):
        self.pipe_splits = PipeSplits()
        self.pipe_judge = PipeJudge()

    def __call__(self, query, outlines_type):
        if outlines_type=="判断类":
            return self.pipe_judge(query=query)
        else:
            return self.pipe_splits(query=query, outlines_type=outlines_type)


class PipeEnsembleForExcel:
    def __init__(self, save_interval=None):
        self.pipe = PipeEnsemble()
        self.save_interval = save_interval

    def __call__(self, input_excel_file, output_excel_file):
        import pandas as pd
        df = pd.read_excel(input_excel_file)
        queries, types = df["query"].tolist(), df["类型"].tolist()
        results = []
        for query, outlines_type in tqdm(zip(queries, types), total=len(queries)):
            result = self.pipe(query=query, outlines_type=outlines_type)
            results.append(result)
            if len(results) % self.save_interval == 0:
                output_excel_file_name = output_excel_file.replace(".xlsx", f"{len(results)}-of-{len(queries)}.xlsx")
                self.save_results(results, output_excel_file_name)
        self.save_results(results, output_excel_file)

    def save_results(self, results, output_excel_file):
        result_keys = [
            "query", "references", "article",
            "exception", "dollars", "seconds"
        ]
        list_results = {k:[result[k] for result in results] for k in result_keys}
        pd.DataFrame.from_dict(list_results).to_excel(output_excel_file)


if __name__ == "__main__":
    pipe = PipeEnsembleForExcel(save_interval=10)
    pipe(
        input_excel_file="250条chat生成.xlsx",
        output_excel_file="250条chat-生成结果.xlsx"
    )