import json
import requests
from tqdm import tqdm

from utils import print_
from chat import chat_with_auto_retry, num_tokens_of_string
from files import read_cached_answers, save_cached_answers, read_queries


def baidu(query):
    print_(f"search for query: {query}")
    answers = []
    url = "https://sp0.baidu.com/8aQDcjqpAAV3otqbppnN2DJv/api.php"
    params = {
        "resource_id": "5300", "format": "json", "ie": "utf-8", "oe": "utf-8",
        "tn": "tangram", "dsp": "iphone", "dspName": "iphone", "alr": "1", "from_mid": "1",
        "query": query, "new_need_di": "1", "pn": 0, "rn": 10, "cb": "jsonp_1625297415567_25430"
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.text
            json_data = json.loads(data[data.find("{"):data.rfind("}") + 1])
            with open("json_data.jsonl", "a+", encoding="utf-8") as f:
                f.write(json.dumps(json_data, ensure_ascii=False) + "\n")
            result_num = int(json_data["ResultNum"])
            if result_num > 0:
                assert result_num == 1
                answers = json_data["Result"][0]["DisplayData"]["resultData"]["tplData"]["result"]["list"]
            else:
                print_(f"no answers: ResultNum = 0")
        else:
            print_(f"no answers: response.status_code = {response.status_code}")
    except Exception as e:
        print_(f"exception in baidu api: {e}")
    title2abstract = {answer['title']:answer['abstract'] for answer in answers}
    return title2abstract


def paraphrase_for_searching(query, searched_queries=None, temperature=0.2):
    prompt = f"请将以下query换一种简短表述用于搜索：\n{query}"
    if searched_queries is not None:
        prompt = f"{prompt}\n我已经尝试过以下表述，但没有理想效果：\n{searched_queries}"
    return chat_with_auto_retry(prompt, temperature=temperature) or query


def ner_for_searching(query, temperature=0.2):
    prompt = f"请提取以下query中最主要一个的实体词，并分析该query最关注实体词的哪一个方面的属性，尽量都返回名词：\n" \
             f"“{query}”\n" \
             f"按如下格式返回：\n" \
             f"最主要一个的实体词：\n" \
             f"最关注的实体词的属性名词：\n"
    answer = chat_with_auto_retry(prompt, temperature=temperature)
    answer_lines = answer.strip().split("\n")
    entity = [line.replace("最主要一个的实体词：","") for line in answer_lines if "最主要一个的实体词：" in line]
    entity_property = [line.replace("最关注的实体词的属性名词：","") for line in answer_lines if "最关注的实体词的属性名词：" in line]
    queries = []
    for ent in entity:
        for pro in entity_property:
            if "、" in pro:
                first_pro = pro.split("、")[0]
                queries.append(f"{ent} {first_pro}")
            queries.append(f"{ent} {pro}")
        queries.append(ent)
    return queries


def baidu_in_loop(query, title2abstract=None, expected_num_answers=5, max_retry_times=5):
    if title2abstract is None:
        title2abstract = {}
    elif len(title2abstract) >= expected_num_answers:
        return title2abstract
    searched_queries = []
    ner_queries = None
    for retry_time in range(max_retry_times):
        print_(f"retry_time: {retry_time}, len(title2abstract):{len(title2abstract)}")
        if ner_queries is None:
            ner_queries = ner_for_searching(query)
            ner_queries = [_ for _ in ner_queries if _ not in searched_queries]
        if len(title2abstract) > 0 or retry_time > 0:
            # the original query is already searched
            if len(ner_queries) > 0 and retry_time >= max_retry_times//2:
                query_ = ner_queries.pop(0)
            else:
                query_ = paraphrase_for_searching(query, searched_queries)
        else:
            query_ = query
        title2abstract.update(baidu(query_))
        searched_queries.append(query_)
        if len(title2abstract) >= expected_num_answers:
            break
    print_(f"retry_time: {retry_time+1}, len(title2abstract):{len(title2abstract)}")
    return title2abstract


class ReferencesPipe:
    def __init__(self, max_num_answers=5, max_single_len=350, max_total_num_tokens=1600):
        self.query2answers = read_cached_answers()
        self.max_num_answers = max_num_answers
        self.max_single_len = max_single_len
        self.max_total_num_tokens = max_total_num_tokens

    def get_cached_answers(self, query):
        title2abstract = self.query2answers.get(query, None)
        return title2abstract

    def make_references(self, title2abstract):
        references = [f"{i+1}. {title}：{abstract}" for i,(title,abstract) in enumerate(title2abstract.items())]
        total_num_tokens = lambda references:num_tokens_of_string("\n".join(references))
        if total_num_tokens(references) > self.max_total_num_tokens:
            references = [reference[:self.max_single_len] for reference in references]
        while total_num_tokens(references) > self.max_total_num_tokens and len(references) > self.max_num_answers:
            references = references[:-1]
        if total_num_tokens(references) > self.max_total_num_tokens:
            squeeze_ratio = self.max_total_num_tokens / total_num_tokens(references)
            references = [reference[:int(len(reference) * squeeze_ratio)] for reference in references]
        references = "\n".join(references)
        return references

    def __call__(self, query):
        title2abstract = self.get_cached_answers(query)
        title2abstract = baidu_in_loop(query, title2abstract)
        self.query2answers[query] = title2abstract
        save_cached_answers(self.query2answers)
        return self.make_references(title2abstract)


if __name__ == "__main__":
    queries = read_queries(num_queries_per_sheet=150, num_queries_to_read_per_sheet=150)
    references_pipe = ReferencesPipe()
    for query in tqdm(queries):
        references = references_pipe(query)
        print(f"query:\n{query}\nreferences:\n{references}\n\n")