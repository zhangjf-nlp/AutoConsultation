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


def add_index_to_lines(lines):
    return [f"{i+1}. {line}" for i,line in enumerate(lines)]


def should_except_span(span, spans_for_exception):
    for span_for_exception in spans_for_exception:
        if common_ratio(span, span_for_exception) > 0.5 and \
                common_ratio(span_for_exception, span) > 0.5:
            return True
    return False


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
        substring = article[i:i+min_length]
        if not re.match(r'[a-zA-Z0-9\u4e00-\u9fff]{6,}', substring):
            continue

        if article.count(substring) > 1:
            j = 0
            while i+min_length+j <= len(article) and \
                    re.match(r'[a-zA-Z0-9\u4e00-\u9fff]{6,}', article[i:i+min_length+j]) and \
                    article.count(article[i:i+min_length+j]) > 1 :
                j += 1
            substring = article[i:i+min_length+j-1]
            if not any([substring in _ for _ in repeated_substrings]) and \
                    not any([part2_head in substring or substring in part2_head for part2_head in part2_heads]):
                repeated_substrings.append(substring)

    if should_in_different_lines:
        article_lines = article.split("\n")
        repeated_substrings = [_ for _ in repeated_substrings if len([line for line in article_lines if _ in line])>1]

    repeated_substrings = sorted(repeated_substrings, key=lambda substring:len(substring), reverse=True)

    return repeated_substrings


def repetition_score(article):
    repeated_substrings = find_repeated_substrings(article)
    score1 = len(repeated_substrings)
    score2 = sum([len(_) for _ in repeated_substrings])
    return score1, score2


class NonChineseCharRemovePipe:
    def __init__(self):
        pass

    def make_prompt(self, phase):
        splits = re.findall(r'[\u4e00-\u9fffA-Za-z0-9]+', phase)
        choices = "\n".join([f"{chr(ord('a')+i)}. {split}" for i,split in enumerate(splits)])
        answer_format = "或".join([f"{chr(ord('a')+i)}" for i,split in enumerate(splits)])
        prompt = f"请分析以下短语中哪一部分为最主要的成分：\n" \
                 f"短语：“{phase}”\n" \
                 f"主要成分选项：\n" \
                 f"{choices}\n" \
                 f"请回答：{answer_format}"
        return prompt

    def has_non_chinese_characters(self, text):
        pattern = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9]')
        return bool(pattern.search(text))

    def __call__(self, text):
        if self.has_non_chinese_characters(text):
            splits = re.findall(r'[\u4e00-\u9fff]+', text)
            if len(splits) == 1:
                return splits[0]
            choices = "\n".join([f"{chr(ord('a') + i)}. {split}" for i, split in enumerate(splits)])
            answer_format = "或".join([f"{chr(ord('a') + i)}" for i, split in enumerate(splits)])
            prompt = f"请分析以下短语中哪一部分为最主要的成分：\n" \
                     f"短语：“{text}”\n" \
                     f"主要成分选项：\n" \
                     f"{choices}\n" \
                     f"请回答：{answer_format}"
            answer = chat_with_auto_retry(prompt)
            chosen = re.findall(f"[a-{chr(ord('a')+len(splits)-1)}]", answer)
            if len(chosen) == 0:
                print_(f"fail to decode the answer in NonChineseCharRemovePipe, so returns None: {splits} -> \n{answer}")
                return None
            chosen_split = splits[ord(chosen[0])-ord('a')]
            print_(f"{text} -> {chosen_split}")
            text = chosen_split
        if text.endswith("等"):
            text = text[:-1]
        return text


class ArticleOutlinesRerankPipe:
    def __init__(self):
        pass

    def classify_methods(self, query, outlines):
        indexed_outlines = "\n".join(add_index_to_lines(outlines))
        prompt = f"以下是关于问题“{query}”的若干条关于治疗方法的分类讨论讲解：\n" \
                 f"“{indexed_outlines}”\n" \
                 f"请判断它们各自属于以下哪一类治疗方法：" \
                 f"a. 保守治疗：通过生活方式调整、营养补充、体能锻炼、心理咨询来改善病情\n" \
                 f"b. 轻度干预：除了生活方式调整外，还可能包括物理治疗，以及一些非处方药物的使用，如维生素和矿物补充剂、解热镇痛药、消化系统药物、皮肤用药、抗组胺药等\n" \
                 f"c. 中度干预：除了物理治疗和非处方药外，还可能包括通过就医咨询，开具并使用处方药物，如抗生素、心血管药物、抗抑郁药、抗糖尿病药、激素类药物等\n" \
                 f"d. 重度干预：可能包括强力药物的使用和一些小型手术，如关节注射、内窥镜手术等\n" \
                 f"e. 侵入性治疗：可能包括复杂的手术和其他高风险、高侵入性的医学干预，如心脏手术、器官移植等\n" \
                 f"按如下格式作答：\n" \
                 f"治疗方法分类：1. <a>或<b>或<c>或<d>或<e>; 2. <a>或<b>或<c>或<d>或<e>; ..."
        answer = chat_with_auto_retry(prompt)
        classes = list(re.findall(r'\d+\.\s*([a-zA-Z]+)', answer.replace(">", "").replace("<", "")))
        print_(f"method classification: {indexed_outlines} -> \n{answer}")
        if not len(classes) == len(outlines):
            raise Exception(f"number of classification mismatch with number of lines: "
                            f"{len(classes)} != {len(outlines)}: {answer}")
        return classes

    def classify_causes(self, query, outlines, left_retry_times=3):
        def infer_method(query, cause):
            prompt = f"以下是关于问题“{query}”的一条可能的病因：\n" \
                     f"“{cause}”\n" \
                     f"请用30字简要概述该病因的主要治疗方法"
            answer = chat_with_auto_retry(prompt)
            return answer
        causes_with_method = "\n".join([f"{i+1}. {cause}：{infer_method(query, cause)}"
                                        for i,cause in enumerate(outlines)])
        prompt = f"以下是关于问题“{query}”的若干条可能的病因与对应的主要治疗方法：\n" \
                 f"“{causes_with_method}”\n" \
                 f"请判断它们各自属于以下哪一类病因：" \
                 f"a. 很轻：不影响正常生活，可通过调整作息、锻炼身体、心理辅导来改善病情\n" \
                 f"b. 较轻：基本不影响正常生活，可通过调整饮食、补充营养、心理疏导来改善病情，暂不需要药物治疗\n" \
                 f"c. 中度：不影响基本生活功能，可能需要物理治疗、调整药物疗程，以及非处方药物的使用，如解热镇痛药、消化系统药物、皮肤用药、抗组胺药等\n" \
                 f"d. 较重：可能影响到一些基本生活功能，需要配合处方药物的使用，如抗生素、心血管药物、抗抑郁药、抗糖尿病药、激素类药物等\n" \
                 f"e. 重度：病情严重，影响基本生活功能，在强力药物治疗以外，还可能需要小型手术，如关节注射、内窥镜手术等\n" \
                 f"f. 紧急：病情发展迅速，可能迅速恶化，需要立即进行紧急治疗，或者需要高风险、高侵入性的医学干预，如心脏手术、器官移植等\n" \
                 f"按如下格式作答：\n" \
                 f"病因等级分类：1. <a>或<b>或<c>或<d>或<e>或<f>; 2. <a>或<b>或<c>或<d>或<e>或<f>; ..."
        answer = chat_with_auto_retry(prompt)
        classes = list(re.findall(r'\d+\.\s*([a-zA-Z]+)', answer.replace(">", "").replace("<", "")))
        while not len(classes) == len(outlines) and left_retry_times > 0:
            left_retry_times -= 1
            answer = chat_with_auto_retry(prompt, temperature=1-left_retry_times*0.3)
            classes = list(re.findall(r'\d+\.\s*([a-zA-Z]+)', answer.replace(">", "").replace("<", "")))
        print_(f"causes classification: {causes_with_method} -> \n{answer}")
        if not len(classes) == len(outlines):
            raise Exception(f"number of classification mismatch with number of lines: "
                            f"{len(classes)} != {len(outlines)}: {answer}")

        return classes

    def __call__(self, query, outlines, outlines_type):
        if outlines_type == "病因分析":
            classes = self.classify_causes(query, outlines)
        elif outlines_type == "治疗方法":
            classes = self.classify_methods(query, outlines)
        else:
            raise NotImplementedError(outlines_type)
        sorted_index = sorted([(c, i) for i, c in enumerate(classes)])

        reranked_outlines = [outlines[i] for c, i in sorted_index]
        print_(f"reranked_outlines:\n{reranked_outlines} ->")
        reranked_outlines = reranked_outlines[:5]
        print_(f"{reranked_outlines}")
        indexed_outlines = add_index_to_lines(reranked_outlines)
        return indexed_outlines


class ArticleOutlinesInferencePipe:
    def __init__(self):
        self.examples = [
            "问题：“大脑缺氧头疼怎么办”\n"
            "病情：大脑缺氧头疼\n"
            "病因分析（至少5点）：1. 睡眠不足；2. 颈椎病；3. 脑血管痉挛；4. 脑血栓；5. 贫血；6. 高血压\n"
            "治疗方法（至少5点）：1. 补充氧气；2. 调整睡眠；3. 调整饮食；4. 物理治疗；5. 药物治疗；6. 手术治疗",

            "问题：“大便长期干燥怎么办”\n"
            "病情：大便长期干燥\n"
            "病因分析（至少5点）：1. 饮食不当；2. 缺乏运动；3. 药物副作用；4. 精神压力大；5. 消化系统疾病；6. 甲亢\n"
            "治疗方法（至少5点）：1. 调整饮食；2. 调整睡眠；3. 增加运动；4. 调整用药；5. 心理调节；6. 物理治疗；7. 药物治疗",

            "问题：“吃海鲜脸过敏发红痒怎么办”\n"
            "病情：吃海鲜脸过敏发红痒\n"
            "病因分析（至少5点）：1. 蛋白质过敏；2. 食材不新鲜；3. 缺少组织胺酶；4. 面部肌肤敏感；5. 免疫系统异常；6. 肠道免疫菌群不足\n"
            "治疗方法（至少5点）：1. 避免抓挠；2. 避免热水烫洗；3. 调整饮食；4. 物理治疗；5. 外用止痒药膏；6. 使用抗过敏药物",

            "问题：“老人吃东西烧心怎么办”\n"
            "病情：吃东西烧心\n"
            "病因分析（至少5点）：1. 饮食刺激；2. 生活习惯不当；3. 反流性食管炎；4. 慢性胃炎；5. 胃溃疡；6. 十二指肠溃疡\n"
            "治疗方法（至少5点）：1. 调整饮食；2. 调整睡眠；3. 适当运动；4. 中药调理；5. 药物治疗；6. 手术治疗",

            "问题：“肠道溃疡怎么办”\n"
            "病情：肠道溃疡\n"
            "病因分析（至少5点）：1. 饮食刺激；2. 卫生习惯不佳；3. 药物副作用；4. 幽门螺杆菌感染；5. 缺血性肠炎；6. 肠道淋巴瘤\n"
            "治疗方法（至少5点）：1. 调整饮食；2. 改善卫生习惯；3. 调整用药；4. 药物治疗；5. 中药灌肠；6. 手术治疗",
        ]
        self.non_chinese_char_remove_pipe = wrap_pipe_for_parallel(NonChineseCharRemovePipe)()
        self.article_outlines_rerank_pipe = wrap_pipe_for_parallel(ArticleOutlinesRerankPipe)()

    def __call__(self, query, references, at_least=5):
        examples = "\n\n".join(self.examples)
        if at_least == 6:
            examples = examples.replace("至少5点", "至少6点")
        prompt = f"请根据以下来自互联网的信息：\n\n" \
                 f"“{references}”\n\n" \
                 f"结合关于问题“{query}”所述的病情，列出其对应的多种不同的具体病因，以及可行的多种不同的治疗方法，分别简述列举至少{at_least}点，每个分点字数严格限制十字以内、尽量不超过六个字，可参考如下示例：\n\n" \
                 f"{examples}\n\n" \
                 f"现在考虑问题：“{query}”\n" \
                 f"按如下格式返回你的结果：\n" \
                 f"病情：\n" \
                 f"病因分析（至少{at_least}点）：\n" \
                 f"治疗方法（至少{at_least}点）："
        answer = chat_with_auto_retry(prompt)
        print_(answer)
        answer_lines = answer.split("\n")
        causes_head, methods_head = f"病因分析（至少{at_least}点）：", f"治疗方法（至少{at_least}点）："
        causes_lines = [line for line in answer_lines if line.startswith(causes_head)]
        methods_lines = [line for line in answer_lines if line.startswith(methods_head)]
        if len(causes_lines) != 1:
            raise Exception(f"num of casues_lines != 1: {answer}")
        if len(methods_lines) != 1:
            raise Exception(f"num of methods_lines != 1: {answer}")
        causes_line, methods_line = causes_lines[0], methods_lines[0]
        if len(causes_line.split("；")) < 5:
            # maybe in different lines
            index_start, index_end = answer_lines.index(causes_line), answer_lines.index(methods_line)
            causes_line = self.try_to_recognize_multiple_lines(answer_lines[index_start:index_end])
        if len(methods_line.split("；")) < 5:
            # maybe in different lines
            index_start = answer_lines.index(methods_line)
            methods_line = self.try_to_recognize_multiple_lines(answer_lines[index_start:])
        causes_line = causes_line.replace(causes_head,"")
        methods_line = methods_line.replace(methods_head,"")
        try:
            matches = [re.match(r"\d+\.\s(.*)", _) for _ in causes_line.split("；")]
            causes = [match.group(1) for match in matches if match]
        except Exception as e:
            raise Exception(f"decoding causes fails: {causes_line}")
        try:
            matches = [re.match(r"\d+\.\s(.*)", _) for _ in methods_line.split("；")]
            methods = [match.group(1) for match in matches if match]
        except Exception as e:
            raise Exception(f"decoding methods fails: {methods_line}")
        list_kwargs_for_parallel = [{"text":cause} for cause in causes]
        causes = self.non_chinese_char_remove_pipe(list_kwargs_for_parallel)
        causes = [cause for cause in causes if cause is not None and not any([_ in cause for _ in ["其它","其他","等"]])]
        list_kwargs_for_parallel = [{"text":method} for method in methods]
        methods = self.non_chinese_char_remove_pipe(list_kwargs_for_parallel)
        methods = [method for method in methods if method is not None and not any([_ in method for _ in ["其它","其他","病因","等","保守"]])]
        causes, methods = list(set(causes)), list(set(methods))
        if len(causes) >= 5 and len(methods) >= 5:
            list_kwargs_for_parallel = [
                {"query": query, "outlines": causes, "outlines_type": "病因分析"},
                {"query": query, "outlines": methods, "outlines_type": "治疗方法"}
            ]
            indexed_causes, indexed_methods = self.article_outlines_rerank_pipe(list_kwargs_for_parallel)
        elif len(causes) >= 5 and len(methods) < 5:
            print_(f"num of methods < 5: {methods}")
            list_kwargs_for_parallel = [
                {"query": query, "outlines": causes, "outlines_type": "病因分析"}
            ]
            indexed_methods = []
            indexed_causes = self.article_outlines_rerank_pipe(list_kwargs_for_parallel)[0]
        elif len(causes) < 5 and len(methods) >= 5:
            print_(f"num of causes < 5: {causes}")
            list_kwargs_for_parallel = [
                {"query": query, "outlines": methods, "outlines_type": "治疗方法"}
            ]
            indexed_causes = []
            indexed_methods = self.article_outlines_rerank_pipe(list_kwargs_for_parallel)[0]
        else:
            if at_least == 5:
                print_(f"neither causes or methods reaches at least 5 items, retry with 6-item-prompt:\n{causes}\n{methods}")
                return self.__call__(query, references, at_least=6)
            else:
                raise Exception(f"neither causes or methods reaches at least 5 items, even with 6-item-prompt:\n{causes}\n{methods}")
        return indexed_causes, indexed_methods

    def try_to_recognize_multiple_lines(self, lines):
        head_line = lines[0]
        other_matched_lines = []
        for line in lines[1:]:
            if re.match(r"\d+\.\s.*", line):
                other_matched_lines.append(line.strip("。；"))
        return head_line + "；".join(other_matched_lines)


class ArticleConditionalInferencePipe:
    def __init__(self):
        self.cases_causes = [
            "问题：“大脑缺氧头疼怎么办”\n"
            "考虑病因分析：1. 休息不足；2. 缺乏运动；3. 颈椎病；4. 脑血管痉挛；5. 心脏病\n"
            "\n"
            "（第1行：给出简要的概括性的回答，列出给定的5项病因，简述需要怎么做，注意避免“等”字结尾）\n"
            "大脑缺氧头疼的常见病因包括休息不足、缺乏运动、颈椎病、脑血管痉挛和心脏病，患者需要根据自身情况对症治疗。\n"
            "\n"
            "（第2至6行：依次对给定的5项具体病因进行（病因分析），并给出（具体建议））\n"
            "1. 休息不足：（病因分析）对于长期劳作、睡眠不足或休息不充分的患者，大脑运转可能因休息不足而受到影响。（具体建议）建议保持良好的作息习惯，每天保证7-8小时的睡眠，并尽量避免熬夜和过度劳累。\n"
            "2. 缺乏运动：（病因分析）若是剧烈运动后呼吸急促、出现头疼，则一般考虑是平常缺乏运动所致。（具体建议）建议每天进行适量的运动，如散步、跑步、瑜伽等，以提高血液循环，增强心肺功能。\n"
            "3. 颈椎病：（病因分析）如果长时间低头、坐姿不正确、颈部肌肉过度劳损，则多考虑是由颈椎病引起。（具体建议）可以采取物理治疗，如针灸和按摩等，或在医生的指导下使用镇痛药进行缓解，如布洛芬和扑热息痛等。\n"
            "4. 脑血管痉挛：（病因分析）如果出现了头痛、恶心、呕吐、意识状态恶化等症状，则需要考虑是脑血管痉挛。（具体建议）患者可以遵照医嘱，使用尼莫地平等扩张脑血管的药物进行治疗，严重痉挛可进行血管介入手术治疗。\n"
            "5. 心脏病：（病因分析）患者若是出现心悸、胸痛、胸闷等症状，或是有心律失常等心脏疾病史，则多考虑是心脏病造成的。（具体建议）这类患者可在医生指导下使用硝酸酯、瑞舒伐他汀钙等进行药物治疗，必要时可以考虑心脏支架植入术、心脏搭桥手术或心脏起搏器植入术进行手术治疗。\n"
            "\n"
            "（最后一行：补充提醒事项和人文关怀）\n"
            "需要注意的是，不论是哪种情况导致的头疼，患者都需要注意保持良好的生活习惯。如果头疼症状持续不减或者加重，应立即就医。此外，保持积极乐观的心态，避免精神压力过大，也有助于缓解头疼。",
        ]
        self.split_lines_causes = [
            "（第1行：给出简要的概括性的回答，列出给定的5项病因，简述需要怎么做，注意避免“等”字结尾）",
            "（第2至6行：依次对给定的5项具体病因进行（病因分析），并给出（具体建议））",
            "（最后一行：补充提醒事项和人文关怀）",
        ]

        self.cases_methods = [
            "问题：“大脑缺氧头疼怎么办”\n"
            "考虑治疗方法：1. 优化睡眠；2. 适当运动；3. 物理治疗；4. 药物治疗；5. 手术治疗\n"
            "\n"
            "（第1行：给出简要的概括性的回答，列出给定的5种治疗方法，注意避免“等”字结尾）\n"
            "大脑缺氧头疼可能是由多种原因引起的，例如休息不足、缺乏运动、颈椎病和心脏病等，可以根据具体原因采用优化睡眠、适当运动、物理治疗、药物治疗和手术治疗来改善和缓解病情。\n"
            "\n"
            "（第2至6行：依次对给定的5种治疗方法阐述适用情况，并给出（具体建议））\n"
            "1. 优化睡眠：（具体建议）改善睡眠，包括保证足够的睡眠时间，打造舒适睡眠环境，以及采取良好的睡眠姿势，可以让大脑得到充足的休息，从而帮助改善缺氧头疼的症状。\n"
            "2. 适当运动：（具体建议）适度的运动可以提高身体的耐氧能力，帮助身体更好地处理缺氧的情况。不过，如果头疼的非常严重或者伴有其他身体不适，最好先咨询医生的意见再决定是否进行运动。\n"
            "3. 物理治疗：（具体建议）一些物理治疗方法，如热敷、冷敷、按摩等，可以帮助缓解头疼。同时，也可以咨询专业的理疗师，看是否有其他适合你的物理治疗方法。\n"
            "4. 药物治疗：（具体建议）在医生的指导下，一些药物可以帮助缓解头疼和缺氧的症状，例如氯芬酸钠缓释片、塞来昔布胶囊、尼莫地平片等。但是注意，药物的使用需要谨慎，某些药物使用不当可能会产生副作用。\n"
            "5. 手术治疗：（具体建议）严重心脏病造成的缺氧头疼，可能需要考虑手术治疗，例如心脏支架植入术、心脏搭桥手术或心脏起搏器植入术等，具体的手术方案应由医生根据患者的具体情况来制定。\n"
            "\n"
            "（最后一行：补充提醒事项和人文关怀）\n"
            "如果患者症状严重，建议立即就医检查具体原因并配合治疗。患者在治疗过程中需要注意休息、保持良好的心态，同时，家庭成员也应对患者给予足够的关爱与支持。",
        ]
        self.split_lines_methods = [
            "（第1行：给出简要的概括性的回答，列出给定的5种治疗方法，注意避免“等”字结尾）",
            "（第2至6行：依次对给定的5种治疗方法阐述适用情况，并给出（具体建议））",
            "（最后一行：补充提醒事项和人文关怀）",
        ]
        self.article_decode_pipe = ArticleDecodePipe()

    def get_duplication_count(self, query, outlines, article):
        spans_for_exception = [query] + outlines
        spans = re.findall(r'[a-zA-Z0-9\u4e00-\u9fff]{6,}', article)
        span_count = {}
        for span in spans:
            if span not in span_count:
                span_count[span] = 0
            span_count[span] += 1
        duplication_count = 0
        for span, count in span_count.items():
            if count > 1 and not should_except_span(span, spans_for_exception):
                duplication_count += count - 1
        return duplication_count

    def __call__(self, query, references, indexed_outlines, outlines_type, temperature=0.1, max_duplication_count=5):
        if outlines_type == "病因分析":
            part_2_inner_prompts = ["（病因分析）","（具体建议）"]
            cases, split_lines = self.cases_causes, self.split_lines_causes
        elif outlines_type == "治疗方法":
            part_2_inner_prompts = ["（具体建议）"]
            cases, split_lines = self.cases_methods, self.split_lines_methods
        else:
            raise NotImplementedError(outlines_type)

        examples_for_prompt = "\n\n".join(cases)
        split_lines_ = "\n".join(split_lines)
        prompt = f"请参考以下资料：\n“{references}”\n\n" \
                 f"考虑问题：“{query}”\n" \
                 f"按如下示例格式生成问诊回复：\n\n" \
                 f"“{examples_for_prompt}”\n\n" \
                 f"其中回复内容在格式上分为三部分，各部分的开头提示如下，其提示之后会紧接对应回复内容：\n" \
                 f"{split_lines_}\n" \
                 f"现在请针对以下问题和叙述角度，按照上述格式，依次生成以上三个部分的开头提示和对应回复内容：\n" \
                 f"问题：“{query}”\n" \
                 f"考虑{outlines_type}：{'；'.join(indexed_outlines)}\n" \
                 f"\n" \
                 f"{split_lines[0]}"
        chat_start_time = time.time()
        article = chat_with_auto_retry(prompt, temperature=temperature, max_interval_seconds=60)
        chat_end_time = time.time()

        print_(f"num_tokens: references/prompt/article/total: "
               f"{num_tokens_of_string(references)}/"
               f"{num_tokens_of_string(prompt)}/"
               f"{num_tokens_of_string(article)}/"
               f"{num_tokens_of_string(prompt+article)}; seconds: {chat_end_time-chat_start_time:.2f}")

        score = repetition_score(article)
        if score[0] > max_duplication_count:
            print_(f"article generated with temperature ({temperature}) has too many duplication "
                  f"({score[0]} > {max_duplication_count}), so returns None")
            return None

        try:
            article = self.article_decode_pipe(query, article, split_lines, part_2_inner_prompts)
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
        如果患者的打嗝症状持续较长时间或者伴有其他不适症状，应及时到医院就诊，寻求医生的帮助。同时，我们也应该给予患者足够的关爱和支持，帮助他们尽快缓解身体不适，恢复健康。
        
        示例2：
        需要提醒产妇，产后应进行定期检查和指导，保持轻松愉悦的情绪，搭配科学合理的营养饮食，定期运动等等，为乳腺健康护航。同时，家人和社会应该给予他们充分的理解和帮助，共同为新生命的健康成长尽心尽力。
        
        示例3：
        无论是何种原因引起的胃痛，患者在日常生活中应遵循健康的饮食和生活习惯，及时就医查明病因，并根据医生的指导和药物保持好的状态。同时，精神面貌也要乐观积极，避免心理压力过大带来的身体伤害。
        
        现在请补全上述文章最后一段，返回补全后的最后一段，返回结果以最后一段原本内容为开头（即，以“{part_3[:5]}”为开头）：
        """.strip()
        prompt = "\n".join([line.strip() for line in prompt.split("\n")])
        answer = chat_with_auto_retry(prompt).strip()
        answer_lines = [line.strip("“”") for line in answer.split("\n")]
        answer_line = [line for line in answer_lines if line.startswith(part_3[:len(part_3) // 2])]
        if len(answer_line) != 1:
            raise Exception(f"the answer for part_3 completion fails to decode: {answer}")
        return answer_line[0]


class ArticleDecodePipe:
    def __init__(self):
        self.part3_complete_pipe = Part3CompletePipe()

    def __call__(self, query, article, split_lines, part_2_inner_prompts):
        article = article.replace("(","（").replace(")","）").replace(":","：")
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

        if aslp[1] == len(article_lines) - 1 \
                and article_lines[-1].startswith(split_lines[2]) \
                and article_lines[-1].count(split_lines[2]) == 1:
            part3_line = article_lines[-1].replace(split_lines[2],"").strip()
            print_(f"repair part3 split_line and content:\n{article_lines[-1]} -> \n{split_lines[2]}\n{part3_line}")
            article_lines = article_lines[:-1] + [split_lines[2], part3_line]

        part_1 = "\n".join([line.strip() for line in article_lines[:aslp[0]]])
        part_2 = "\n".join([line.strip() for line in article_lines[aslp[0] + 1:aslp[1]]]) # “1” is split_line
        part_3 = "\n".join([line.strip() for line in article_lines[aslp[1] + 1:]]) # “1” is split_line
        part_2 = self.remove_part_2_inner_prompts(part_2, part_2_inner_prompts)
        if not len(part_1)>0:
            raise Exception("part_1 is empty")
        if not len(part_1.split("\n"))==1:
            raise Exception("part_1 contains more than 1 line")
        if not len(part_2.split("\n"))==5:
            raise Exception("part_2 doesn't contain 5 lines")
        if not len(part_3)>0:
            raise Exception("part_3 is empty")
        if not part_3.endswith("。"):
            part_3 = self.part3_complete_pipe(f"{part_1}\n\n{part_2}\n\n{part_3}", part_3)
        if not len(part_3.split("\n"))==1:
            raise Exception("part_3 contains more than 1 line")
        return f"{part_1}\n\n{part_2}\n\n{part_3}"

    def remove_part_2_inner_prompts(self, part_2, part_2_inner_prompts):
        part_2_lines = []
        for line in part_2.split("\n"):
            for prompt in part_2_inner_prompts:
                if prompt not in line:
                    print_(f"inner prompt {prompt} not in line {line}")
                    pass
                else:
                    line = line.replace(prompt, "")
                    if common_ratio(prompt, line)>0.5:
                        print_(f"inner prompt {prompt} may not completely removed from line {line}")
                        pass

            if re.match(r"(\d+\.\s.*：).*", line):
                pass
            elif re.match(r"(\d+\..*：).*", line):
                index_space = line.index(".") + 1
                line = line[:index_space] + " " + line[index_space:]
            else:
                raise Exception(f"part2_line doesn't match split format:\n{line}")

            part_2_lines.append(line)
        return "\n".join(part_2_lines)


class ArticleDuplicationRewritePipe:
    def __init__(self):
        pass

    def repetition_score_for_two_lines(self, line1, line2):
        return repetition_score(f"{line1}\n{line2}")

    def __call__(self, line1, line2, duplication):
        line1_match = re.match(r"(\d+\.\s.*：).*", line1)
        if line1_match is None:
            raise Exception(f"line1 doesn't match split format:\n{line1}")
        line1_head = line1_match.group(1)
        line2_match = re.match(r"(\d+\.\s.*：).*", line2)
        if line2_match is None:
            raise Exception(f"line2 doesn't match split format:\n{line2}")
        line2_head = line2_match.group(1)
        prompt = f"""
        以下两段话中重复使用“{duplication}”对不同情况下的病症或措施进行了描述：
        
        {line1}
        {line2}
        
        请根据不同的情况，将“{duplication}”更换为更加具体的描述，以对不同情况形成区分。
        """.strip()
        prompt = "\n".join([line.strip() for line in prompt.split("\n")])

        futures = []
        with ThreadPoolExecutor() as executor:
            for temperature in [0.5, 1.0]:
                future = executor.submit(chat_with_auto_retry, prompt=prompt, temperature=temperature)
                futures.append(future)
        answers = [future.result() for future in futures]

        answers_for_line1 = list(set([line for answer in answers
                                      for line in answer.split("\n")
                                      if line.startswith(line1_head)]))
        answers_for_line2 = list(set([line for answer in answers
                                      for line in answer.split("\n")
                                      if line.startswith(line2_head)]))

        best_tuple = (line1, line2)
        lowest_score = self.repetition_score_for_two_lines(line1, line2) + (2-bleu(line1, line1)-bleu(line2, line2),)
        for line1_ in answers_for_line1:
            for line2_ in answers_for_line2:
                score = self.repetition_score_for_two_lines(line1_, line2_) + (2-bleu(line1, line1_)-bleu(line2, line2_),)
                if score < lowest_score:
                    lowest_score = score
                    best_tuple = (line1_, line2_)
        input_score = self.repetition_score_for_two_lines(line1, line2)
        output_score = self.repetition_score_for_two_lines(best_tuple[0], best_tuple[1])
        if not output_score < input_score:
            print_(f"prompt:\n{prompt}")
            print_("answers:\n" + "\n".join(answers) + "\n")
        print_(f"({line1}\n{line2}) -> \n"
              f"({best_tuple[0]}\n{best_tuple[1]})")
        print_(f"input score: {input_score}")
        print_(f"output score: {output_score}\n")

        return best_tuple


class ArticleDeduplicatePipe:
    def __init__(self):
        self.article_duplication_rewrite_pipe = ArticleDuplicationRewritePipe()
        self.part_2_example_list_pipe = Part2ExampleListPipe()

    def deduplicate_part2(self, part2, max_retry_times=5):
        part2_lines = part2.split("\n")
        retry_times = 0
        repeated_substrings = find_repeated_substrings("\n".join(part2_lines), should_in_different_lines=True)
        while len(repeated_substrings) > 0 and retry_times < max_retry_times:
            duplication = repeated_substrings[0]
            lines_with_duplication = [(i,line) for i,line in enumerate(part2_lines) if duplication in line]
            assert len(lines_with_duplication) >= 2
            line1, line2 = lines_with_duplication[0][1], lines_with_duplication[1][1]
            line1, line2 = self.article_duplication_rewrite_pipe(line1, line2, duplication)
            part2_lines[lines_with_duplication[0][0]] = line1
            part2_lines[lines_with_duplication[1][0]] = line2
            repeated_substrings = find_repeated_substrings("\n".join(part2_lines), should_in_different_lines=True)
            retry_times += 1
        part2 = "\n".join(part2_lines)
        return part2

    def deduplicate_part3(self, article, part3):
        repeated_substrings = find_repeated_substrings(article, should_in_different_lines=True)
        repeated_substrings = [_ for _ in repeated_substrings if _ in part3]

        if len(repeated_substrings) > 0:
            duplication = repeated_substrings[0]

            prompt = f"""
            下文中的“{duplication}”在前文中已经被提及，请将其换个说法，其余内容保持不变：
            {part3}
            """.strip()

            futures = []
            with ThreadPoolExecutor() as executor:
                for temperature in [0.3, 0.5, 1.0]:
                    future = executor.submit(chat_with_auto_retry, prompt=prompt, temperature=temperature)
                    futures.append(future)
            answers = [future.result() for future in futures]

            answers = sorted(answers, key=lambda answer:(duplication in answer, 1-bleu(part3, answer)))
            part3 = answers[0]

        return part3

    def __call__(self, article):
        part1, part2, part3 = article.split("\n\n")
        part2 = self.deduplicate_part2(part2)
        part2 = self.part_2_example_list_pipe(part2)
        part3 = self.deduplicate_part3(article, part3)
        article = f"{part1}\n\n{part2}\n\n{part3}"
        repeated_substrings = find_repeated_substrings(article, should_in_different_lines=False)
        if len(repeated_substrings) > 0:
            raise Exception(f"deduplication fails:\n{repeated_substrings} ->\n{article}")
        return f"{part1}\n\n{part2}\n\n{part3}"


class Part2ExampleListPipe:
    def __init__(self):
        self.term2examples = {}
        content = open("terms.txt","r",encoding="utf-8").read().strip()
        lines = [line for line in content.split("\n") if len(line)>0]
        for line in lines:
            term = line[:line.index("：")]
            self.term2examples[term] = line

    def try_to_add_examples(self, statement, terms, examples, temperature=0.2):
        prompt = """
        请根据参考资料，改写话术，为其中的药物/食物类型补充具体示例，示例如下：

        示例1.
        话术：局部药物治疗包括使用抗生素、消炎药等药物来缓解包皮过长引起的炎症和不适，但需要在医生的指导下使用。
        参考资料：
        抗生素：青霉素、头孢菌素、红霉素等
        消炎药：布洛芬、阿司匹林、地塞米松等
        改写：局部药物治疗包括使用抗生素、消炎药等药物，例如青霉素、头孢菌素、布洛芬等，来缓解包皮过长引起的炎症和不适，但需要在医生的指导下使用。

        示例2.
        话术：对于胃酸过多、胃溃疡等引起的胃痛，可以遵医嘱服用药物来缓解疼痛，如抗酸药、抗生素等。
        参考资料：
        抗酸药：奥美拉唑、兰索拉唑、雷贝拉唑等
        抗生素：青霉素、头孢菌素、红霉素等
        改写：对于胃酸过多、胃溃疡等引起的胃痛，可以遵医嘱服用药物来缓解疼痛，例如奥美拉唑、兰索拉唑、头孢菌素、红霉素等。

        示例3.
        话术：摄入过多钙质会使粪便变硬，从而造成排便困难。建议适当减少摄入钙质的量，或者使用泻剂等药物进行治疗。
        参考资料：
        泻剂：开塞露、番泻叶、大黄、盐酸多塞酮等
        改写：摄入过多钙质会使粪便变硬，从而造成排便困难。建议适当减少摄入钙质的量，或者在医生的指导下，使用如开塞露、番泻叶、大黄等泻剂进行药物治疗。

        示例4.
        话术：长期便秘的患者可能会引起皮肤异常，出现鼻子上反复长痘痘的现象。建议保持良好的饮食习惯，多吃蔬菜水果，避免过度进食辛辣、甜腻的食物，保持肠道通畅。
        参考资料：
        蔬菜：西红柿、胡萝卜、菠菜、白菜、茄子等
        水果：苹果、香蕉、橙子、草莓、葡萄等
        辛辣食物：辣椒、花椒、咖喱、葱姜蒜等
        甜腻食物：糖果、巧克力、蛋糕、含糖饮料等
        改写：长期便秘的患者可能会引起皮肤异常，出现鼻子上反复长痘痘的现象。建议保持良好的饮食习惯，多吃蔬菜水果，如西红柿、苹果、香蕉等，避免过度进食辛辣、甜腻的食物，如辣椒、巧克力等，保持肠道通畅。
        
        示例5.
        话术：饮食中缺乏维生素A、B2、C等营养素，或者鼻孔周围皮肤缺乏保湿，建议多吃富含维生素的食物，如胡萝卜、绿叶蔬菜、水果等，同时注意局部皮肤的保护，如使用保湿霜等。
        参考资料：
        蔬菜：西红柿、胡萝卜、菠菜、白菜、茄子等
        水果：苹果、香蕉、橙子、草莓、葡萄等
        改写：饮食中缺乏维生素A、B2、C等营养素，或者鼻孔周围皮肤缺乏保湿，建议多吃富含维生素的蔬菜和水果，如胡萝卜、菠菜、苹果、香蕉等，同时注意局部皮肤的保护，如使用保湿霜等。
        
        示例6.
        话术：营养不良可能会导致乳汁分泌减少，从而出现奶水不够的情况，对于这种情况的产妇，建议多吃高蛋白、高维生素的食物，如水果、蔬菜、肉类、蛋类、奶类等，避免吃韭菜、麦芽之类具有回奶效果的食物。
        参考资料：
        蔬菜：西红柿、胡萝卜、菠菜、白菜、茄子等
        水果：苹果、香蕉、橙子、草莓、葡萄等
        蛋类：鸡蛋、鸭蛋、鹅蛋、鹌鹑蛋、鸽子蛋等
        肉类：牛肉、鸡肉、鱼肉、虾肉等
        奶类：牛奶、酸奶等
        改写：营养不良可能会导致乳汁分泌减少，从而出现奶水不够的情况，对于这种情况的产妇，建议多吃高蛋白、高维生素的食物，如苹果、西红柿、鸡肉、鸡蛋、牛奶等，避免吃韭菜、麦芽之类具有回奶效果的食物。
        
        示例7.
        话术：营养不良可能是子宫发育不良的原因之一，建议患者保持营养均衡，多食用富含蛋白质、维生素和矿物质的食物，如鱼、肉、蛋、奶、豆制品、蔬菜和水果等。
        参考资料：
        豆制品：豆腐、豆浆、豆皮
        蔬菜：西红柿、胡萝卜、菠菜、白菜、茄子等
        水果：苹果、香蕉、橙子、草莓、葡萄等
        改写：营养不良可能是子宫发育不良的原因之一，建议患者保持营养均衡，多食用富含蛋白质、维生素和矿物质的食物，如鱼、肉、蛋、奶、豆腐、西红柿和苹果等。
        
        现在请改写
        话术：{statement}
        参考资料：
        {term_lines}
        改写：
        """
        prompt = "\n".join([line.strip() for line in prompt.strip().split("\n")])
        term_lines = sorted([(statement.index(term),example) for term,example in zip(terms,examples)])
        term_lines = "\n".join([line for _,line in term_lines])
        prompt = prompt.format(statement=statement, term_lines=term_lines)
        while True:
            answer = chat_with_auto_retry(prompt, temperature=temperature)
            terms_in_answer = [term for term in self.term2examples if term in re.findall("(.*)。",answer)[-1]]
            success = all([term in term_lines for term in terms_in_answer]) and bleu(statement, answer) > 0.5
            if success or temperature > 1.0:
                break
            else:
                temperature += 0.4
        print_(f"add examples to part_2_line (temperature={temperature:.1f}):\n{statement}\n+\n{term_lines}\n=\n{answer}")
        if not bleu(statement, answer) > 0.5:
            print_(f"adding examples lead to large difference:\n{statement}\n+\n{term_lines}\n=\n{answer}")
        return answer

    def add_examples_to_line(self, line):
        head, statement = line[:line.index("：")+1], line[line.index("：")+1:]
        second_statement = re.findall("(.*)。",statement)[-1]
        terms, examples = [], []
        for term in self.term2examples:
            if term in second_statement:
                terms.append(term)
                examples.append(self.term2examples[term])
            elif "食物" in term and "食物" in second_statement and term.replace("食物", "") in second_statement:
                terms.append(term.replace("食物", ""))
                examples.append(self.term2examples[term])
        if len(terms) > 0:
            line = head + self.try_to_add_examples(statement, terms, examples)
        return line

    def __call__(self, part_2):
        part_2_lines = part_2.split("\n")

        futures = []
        with ThreadPoolExecutor() as executor:
            for line in part_2_lines:
                future = executor.submit(self.add_examples_to_line, line=line)
                futures.append(future)
        part_2_lines = [future.result() for future in futures]

        part_2 = "\n".join(part_2_lines)
        return part_2