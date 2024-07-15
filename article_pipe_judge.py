import re
import time

from utils import print_
from chat import chat_with_auto_retry, num_tokens_of_string
from concurrent.futures import ThreadPoolExecutor

def find_longest_common(A, B):
    m, n = len(A), len(B)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    end = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end = i
            else:
                dp[i][j] = 0

    return A[end - max_length: end]


def common_ratio(A, B):
    return len(find_longest_common(A, B)) / len(A)


def bleu(reference, candidate):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    reference = [list(reference)]
    candidate = list(candidate)
    chencherry = SmoothingFunction()
    score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
    return score


def wrap_pipe_for_parallel(pipe_class):
    class parallel_pipe_class(pipe_class):
        def __init__(self, *args, **kwargs):
            self.pipe = pipe_class(*args, **kwargs)

        def __call__(self, list_kwargs_for_parallel):
            futures = []
            with ThreadPoolExecutor() as executor:
                for kwargs_for_parallel in list_kwargs_for_parallel:
                    future = executor.submit(self.pipe.__call__, **kwargs_for_parallel)
                    futures.append(future)
            results = [future.result() for future in futures]
            return results

    return parallel_pipe_class


def find_repeated_substrings(article, min_length=6, should_in_different_lines=False):
    repeated_substrings = []

    part2_heads = [re.match(r"\d+\.\s(.*)：.*", line).group(1)
                   for line in article.split("\n")
                   if re.match(r"\d+\.\s(.*)：.*", line)]

    for i in range(len(article) - min_length + 1):
        substring = article[i:i + min_length]
        if not re.match(r'[a-zA-Z0-9\u4e00-\u9fff]{6,}', substring):
            continue

        if article.count(substring) > 1:
            j = 0
            while i + min_length + j <= len(article) and \
                    re.match(r'[a-zA-Z0-9\u4e00-\u9fff]{6,}', article[i:i + min_length + j]) and \
                    article.count(article[i:i + min_length + j]) > 1:
                j += 1
            substring = article[i:i + min_length + j - 1]
            if not any([substring in _ for _ in repeated_substrings]) and \
                    not any([part2_head in substring or substring in part2_head for part2_head in part2_heads]):
                repeated_substrings.append(substring)

    if should_in_different_lines:
        article_lines = article.split("\n")
        repeated_substrings = [_ for _ in repeated_substrings if len([line for line in article_lines if _ in line]) > 1]

    repeated_substrings = sorted(repeated_substrings, key=lambda substring: len(substring), reverse=True)

    return repeated_substrings


def repetition_score(article):
    repeated_substrings = find_repeated_substrings(article)
    score1 = len(repeated_substrings)
    score2 = sum([len(_) for _ in repeated_substrings])
    return score1, score2


class ArticleInferencePipe:
    def __init__(self):
        self.cases = [
            """
            （问题）
            白开水放一夜第二天能喝吗

            （第一行：给出一般情况的明确判断，补充考虑特殊情况）
            白开水放置一夜后，其水质会受到一定的影响，一般不建议饮用，但如果白开水保存条件良好，也是可以考虑饮用的。

            （第二行：讲解特质或原理，给出一般情况的判断，补充特殊情况的判断）
            （特质或原理）白开水是指经过沸腾消毒的水，它是人们日常饮用的最常见的水。白开水放置一段时间后，会受到空气中的微生物、灰尘等污染物的影响，导致水质发生变化，主要体现在水中的亚硝酸盐、细菌、氧化物等物质的含量上。（一般情况）一般来说，白开水放置时间越长，水质变化越大，对人体健康的影响也越大，因此不建议饮用隔夜的白开水，尤其是在夏季高温、湿度大、细菌繁殖快的环境下。（特殊情况）但是，并不是所有的白开水都会在放置一夜后就变质。如果白开水是在干净、密封、阴凉的容器中保存，并且没有受到外界污染物的侵入，那么它仍然可以保持一定的饮用安全性。

            （第三行：补充提醒事项和人文关怀）
            综上所述，放了一夜的白开水第二天是否能喝，需要根据具体的保存条件来判断。为了保证饮用水的安全和健康，建议大家尽量喝新鲜烧开的白开水，并且注意保持容器和环境的清洁。
            """,
            """
            （问题）
            45x染色体异常能要孩子吗

            （第一行：给出一般情况的明确判断，补充考虑特殊情况）
            45x染色体异常会一定程度上影响生育能力，大部分的患者无法自然怀孕，但也有部分患者仍具有生殖潜力。

            （第二行：讲解特质或原理，给出一般情况的判断，补充特殊情况的判断）
            （特质或原理）染色体异常45X，也称为Turner综合征，是一种女性性染色体疾病，由缺失一个X染色体所导致。这种染色体异常会影响女性的生长发育和生殖功能，导致身材矮小、性腺发育不全、卵巢功能低下等表现。（一般情况）染色体异常45X的女性大多数是不育的，因为她们的卵巢不能正常产生卵子，或者卵子的质量和数量都很差，据统计约有90%的患者不能自然怀孕。（特殊情况）但是也有一部分患者是嵌合型的，她们的一部分细胞是正常的46,XX，另一部分细胞是异常的45,X。这种情况下，如果正常细胞占有较高的比例，并且分布在性腺组织中，那么她们可能还有一定的生殖潜力。

            （第三行：补充提醒事项和人文关怀）
            45x染色体异常对女性生育能力的影响程度因个体差异而异，建议患者寻找专业的医疗团队进行咨询和评估。此外，患者家属还需要给予充分的关爱和支持，帮助患者建立自信、乐观、积极的人生态度。
            """,
            """
            （问题）
            2度1型房室传导阻滞严重吗

            （第一行：给出一般情况的明确判断，补充考虑特殊情况）
            2度1型房室传导阻滞，通常被认为是一种较为良性的心律失常，不算特别严重，但如果同时还伴有其它不适症状，则可能需要引起重视。

            （第二行：讲解特质或原理，给出一般情况的判断，补充特殊情况的判断）
            （特质或原理）2度1型房室传导阻滞是一种心电图现象，表现为心电图上的PR间期逐渐延长，最终导致非传导性P波的形成。这种情况通常是由于房室结水平的可逆传导阻滞所致。受损的房室结细胞通常会逐渐疲劳，直至无法传导冲动。（一般情况）2度1型房室传导阻滞通常被认为是一种良性节律，可引起较小的血流动力学紊乱，进展为三度房室传导阻滞的风险较低。（特殊情况）然而，如果患者还出现了如晕厥、心悸等症状，可能需要进行进一步的评估和治疗。

            （第三行：补充提醒事项和人文关怀）
            2度1型房室传导阻滞通常不算严重，但也建议患者定期进行心电图检查，密切关注症状变化。如有任何不适，应立即就医。此外，保持积极乐观的态度，也可以帮助患者更好地管理这种状况。
            """
        ]
        strip = lambda article: "\n".join([line.strip() for line in article.strip().split("\n")])
        self.cases = [strip(case) for case in self.cases]
        self.article_decode_pipe = ArticleDecodePipe()

    def __call__(self, query, references, temperature=0.1):
        examples_for_prompt = "\n\n".join(self.cases)
        prompt = f"请参考以下资料：\n“{references}”\n" \
                 f"考虑问题：“{query}”\n" \
                 f"按如下示例生成结果：\n\n" \
                 f"{examples_for_prompt}\n\n" \
                 f"现在请完成以下文章剩余部分：\n" \
                 f"（问题）\n" \
                 f"{query}\n" \
                 f"\n" \
                 f"（第一行：给出一般情况的明确判断，补充考虑特殊情况）\n"
        chat_start_time = time.time()
        article = chat_with_auto_retry(prompt, temperature=temperature, max_interval_seconds=60)
        chat_end_time = time.time()

        print_(f"num_tokens: references/prompt/article/total: "
               f"{num_tokens_of_string(references)}/"
               f"{num_tokens_of_string(prompt)}/"
               f"{num_tokens_of_string(article)}/"
               f"{num_tokens_of_string(prompt + article)}; seconds: {chat_end_time-chat_start_time:.2f}")

        split_lines = [
            "（第一行：给出一般情况的明确判断，补充考虑特殊情况）",
            "（第二行：讲解特质或原理，给出一般情况的判断，补充特殊情况的判断）",
            "（第三行：补充提醒事项和人文关怀）"
        ]
        part_2_inner_prompts = ["（特质或原理）", "（一般情况）", "（特殊情况）"]

        try:
            article = self.article_decode_pipe(article, split_lines, part_2_inner_prompts)
        except Exception as e:
            print_(f"article generated with temperature ({temperature}) fails to decode, so returns None:\n{article}\n{e}")
            return None

        return article


class Part3CompletePipe:
    def __init__(self):
        pass        

    def __call__(self, article, part_3):
        prompt = f"""
        下面这篇文章的最后一段没有写完：
        “{article}”

        请将其最后一段补全后返回给我，其上要求为：补充对患者的提醒事项和人文关怀
        示例如下：

        示例1：
        如果一个40岁的女性出现了3天内例假结束的情况，建议她尽快寻求医疗帮助，进行检查和治疗，以确定病因并进行针对性的治疗。此外，她应该注意调整生活方式，保持良好的心态，避免过度劳累和情绪波动，保持充足的睡眠和合理的饮食，有助于维护身体健康。同时，家人和朋友也应该给予她充分的关爱和支持，帮助她度过这个特殊的时期。

        示例2：
        百合花过敏是一种常见的过敏反应，需要引起足够的重视。为了避免过敏反应的发生，建议敏感人群尽量避免接触百合花及其制品。同时，对于已经出现过敏症状的患者，也需要积极进行治疗和调理，避免症状加重。此外，家人和朋友也应该给予患者充分的关爱和支持，帮助他们度过难关。

        示例3：
        总之，安宫牛黄丸并不能治疗肺癌，患者在治疗过程中应该听从医生的建议，进行规范的治疗方案。此外，对于使用安宫牛黄丸等中成药物，患者也需要遵医嘱用药，注意药物的剂量和不良反应。同时，患者也需要得到家人和社会的关爱和支持，保持乐观的心态，积极面对治疗和康复。

        现在请补全上述文章最后一段，返回补全后的最后一段，返回结果以最后一段原本内容为开头（即，以“{part_3[:5]}”为开头）：
        """.strip()
        prompt = "\n".join([line.strip() for line in prompt.split("\n")])
        answer = chat_with_auto_retry(prompt).strip()
        answer_lines = [line.strip("“”") for line in answer.split("\n")]
        answer_line = [line for line in answer_lines if line.startswith(part_3[:len(part_3) // 2])]
        if len(answer_line) != 1:
            raise Exception(f"the answer for part_3 completion fails to decode: {answer}")
        print_(f"complete part_3:\n{part_3} -> \n{answer_line[0]}")
        return answer_line[0]


class ArticleDecodePipe:
    def __init__(self):
        self.part3_complete_pipe = Part3CompletePipe()

    def __call__(self, article, split_lines, part_2_inner_prompts):
        article = article.replace("(", "（").replace(")", "）")
        article_lines = article.split("\n")
        article_lines = [line for line in article_lines if len(line) > 0]
        article_split_line_positions = []
        for split_line in split_lines[1:]:
            already_matched = False
            for i, article_line in enumerate(article_lines):
                ratio = common_ratio(split_line, article_line)
                if ratio > 0.5:
                    assert not already_matched
                    article_split_line_positions.append(i)
                    already_matched = True
            if not already_matched:
                raise Exception(f"fail to match split_line: {split_line}:\n{article}")
        if not article_split_line_positions == sorted(article_split_line_positions):
            raise Exception(f"split_line match results in disorder: {article_split_line_positions}")
        aslp = article_split_line_positions
        part_1 = "\n".join([line.strip() for line in article_lines[:aslp[0]]])
        part_2 = "\n".join([line.strip() for line in article_lines[aslp[0] + 1:aslp[1]]])  # “1” is split_line
        part_3 = "\n".join([line.strip() for line in article_lines[aslp[1] + 1:]])  # “1” is split_line
        part_2 = self.remove_part_2_inner_prompts(part_2, part_2_inner_prompts)
        if any([_ in part_1 for _ in part_2_inner_prompts]):
            raise Exception(f"part_1 contains part_2_inner_prompt:\n{part_1}")
        if any([_ in part_3 for _ in part_2_inner_prompts]):
            raise Exception(f"part_3 contains part_2_inner_prompt:\n{part_3}")
        if not len(part_1) > 0:
            raise Exception("part_1 is empty")
        if not len(part_2) > 0:
            raise Exception("part_2 is empty")
        if not len(part_3) > 0:
            raise Exception("part_3 is empty")
        parts = [part_1, part_2, part_3]
        for i, part in enumerate(parts):
            if part.startswith("("):
                match = re.match(r"（.*）", part)
                if match:
                    print_(f"remove {match.group(0)} at the start of part_{i + 1}:\n{part}")
                    parts[i] = part[len(match.group(0)):]
                else:
                    raise Exception(f"line startswith “（” but not found “）”")
        if not part_3.endswith("。"):
            part_3 = self.part3_complete_pipe("\n\n".join(parts), part_3)
            parts[2] = part_3
        return "\n\n".join(parts)

    def remove_part_2_inner_prompts(self, part_2, part_2_inner_prompts):
        part_2_lines = []
        for line in part_2.split("\n"):
            for prompt in part_2_inner_prompts:
                if prompt not in line:
                    
                    print_(f"inner prompt {prompt} not in line {line}")
                else:
                    line = line.replace(prompt, "")
                    if common_ratio(prompt, line) > 0.5:
                        
                        print_(f"inner prompt {prompt} may not completely removed from line {line}")
            part_2_lines.append(line)
        return "\n".join(part_2_lines)


class ArticleDeduplicatePipe:
    def __init__(self):
        pass

    def __call__(self, article, max_retry_times=5):
        parts = article.split("\n\n")
        retry_times = 0
        repeated_substrings = find_repeated_substrings("\n\n".join(parts), min_length=8, should_in_different_lines=True)
        while len(repeated_substrings) > 0 and retry_times < max_retry_times:
            duplication = repeated_substrings[0]
            parts_with_duplication = [(i, part) for i, part in enumerate(parts) if duplication in part]
            assert len(parts_with_duplication) >= 2
            index_part_to_deduplicate, part_to_deduplicate = parts_with_duplication[-1]
            index_duplication = part_to_deduplicate.rfind(duplication)
            part_in_prompt = part_to_deduplicate[:index_duplication] + f"“{duplication}”" \
                             + part_to_deduplicate[index_duplication + len(duplication):]
            prompt = f"请将下文中的引号之间的部分换一个说法，其余内容保持不变：\n{part_in_prompt}"

            futures = []
            with ThreadPoolExecutor() as executor:
                for temperature in [0.3, 0.5, 1.0]:
                    future = executor.submit(chat_with_auto_retry, prompt=prompt, temperature=temperature)
                    futures.append(future)
            answers = [future.result().strip() for future in futures] + [part_to_deduplicate]
            answers = sorted(answers, key=lambda answer: (duplication in answer, 1 - bleu(part_to_deduplicate, answer)))

            for _ in "“”\"":
                if not _ in part_to_deduplicate:
                    answers = [__.replace(_, "") for __ in answers]
                else:
                    
                    print_(f"content contains {_}:\n{part_to_deduplicate}")

            for _ in ":：":
                if not _ in part_to_deduplicate:
                    answers = [answer for answer in answers if _ not in answer]

            parts[index_part_to_deduplicate] = answers[0]
            repeated_substrings = find_repeated_substrings("\n\n".join(parts), min_length=8,
                                                           should_in_different_lines=True)
            retry_times += 1
        return "\n\n".join(parts)