from tqdm import tqdm

from chat import chat_with_auto_retry

class PriorOutlinePipe:
    def __init__(self):
        self.prompt = """
        请判断以下问题询问的意图属于哪种：
        a. 分析病因类：按多种可能病因分条解析
        b. 列举方法类：按多种治疗方法分条解析
        c. 判断类：按常见情况与特殊情况进行判断分析
        d. 都不属于
        问题：
        {query}
        请按如下格式返回你的结果：
        意图：<a>或<b>或<c>或<d>
        
        可参考如下示例：
        示例1.
        问题：
        顽固失眠怎么彻底治疗
        意图：<b>

        示例2.
        问题：
        羊肚的功效与营养价值
        意图：<a>或<b>
        
        示例3.
        问题：
        气胸手术是小手术吗
        意图：<c>
        
        示例4.
        问题：
        孕酮低于15不建议保胎吗
        意图：<c>
        
        示例5.
        问题：
        喝牛奶会长痘吗
        意图：<c>
        
        示例6.
        问题：
        血压昼夜节律消失是怎么回事，怎么办
        意图：<a>
        
        示例7.
        问题：
        颞下颌关节紊乱能自愈吗
        意图：<c>
        
        示例8.
        问题：
        吃梨会胖吗
        意图：<c>
        
        示例9.
        问题：
        坐灸仪的熏屁股能天天灸吗
        意图：<c>
        
        示例10.
        问题：
        血小板低于50能活多久
        意图：<c>
        
        此外还可以参考如下检索到的相关信息：
        {references}
        现在请考虑给定的问题作出回答：
        问题：
        {query}
        意图：
        """.strip()
        self.prompt = "\n".join([_.strip() for _ in self.prompt.split("\n")])

    def __call__(self, query, references, **kwargs):
        prompt = self.prompt.format(query=query, references=references)
        return chat_with_auto_retry(prompt, **kwargs)


def test():
    from references_pipe import ReferencesPipe
    reference_pipe = ReferencesPipe()
    pipe = PriorOutlinePipe()
    cases = """怀孕能化妆吗	判断类
老年人白肺病能撑多久	判断类
咖啡喝了会胖吗	判断类
黑豆浆可以促进卵泡发育吗	判断类
一天一个石榴上火吗	判断类
中暑会恶心想吐吗	判断类
没有输卵管可以做试管吗	判断类
酒后第二天能吃头孢吗	判断类
吗丁啉和莫沙必利哪个好	判断类
老人血压低就快去世了吗	判断类
炸蝎子一次能吃几只	判断类
前列腺炎坚决不能同房	判断类
五种人千万不能吃玛卡	判断类
山药发芽了还能不能吃	判断类
男人吃螃蟹壮阳吗	判断类
自主神经功能紊乱	判断类
肺大泡能活几年	判断类
司美格鲁肽能减肥吗	判断类
三阴早期无转移化疗能活多久	判断类
女人小腿肿是大病前兆吗	判断类
免疫球蛋白不能随便打	判断类
肾功能不全三大表现	判断类
胃食管反流是大病吗	判断类
一天三支烟有必要戒吗	判断类
20来岁飞蚊症老了会瞎吗	判断类
蛇盘疮盘一圈会死人吗	判断类
颞下颌关节紊乱能自愈吗	判断类
孕酮低于15不建议保胎吗	判断类
肺结核一辈子不能熬夜	判断类
螺旋藻片的正确吃法与功效	病因/方法类
牙齿出血是什么病征兆	病因/方法类
天天拉肚子大便不成形怎么回事	病因/方法类
葵花胃康灵的功效与作用	病因/方法类
dna是什么	病因/方法类
脑壳神经痛一阵一阵怎么办	病因/方法类
治疗黄褐斑的最好方法	病因/方法类
酚氨咖敏片为什么被禁用了	病因/方法类
女性尿道口小便刺痛什么原因	病因/方法类
喝酒前喝什么不容易醉	病因/方法类
泮托拉唑的严重副作用	病因/方法类
尿赤是什么意思	病因/方法类
肺里痰多是什么原因	病因/方法类
咳必清的功效与作用	病因/方法类
黄金果是什么水果	病因/方法类
胎儿腹围偏小说明什么	病因/方法类
""".strip()
    cases = [line.split("\t") for line in cases.split("\n")]
    n_correct = 0
    for query,type in tqdm(cases):
        references = reference_pipe(query)
        answer = pipe(query, references)
        pred_is_judge = "c" in answer
        label_is_judge = type == "判断类"
        correct = pred_is_judge==label_is_judge
        n_correct += int(correct)
        if not correct:
            print(f"query:\n{query}\ntype:{type}\npred:{answer}")
    print(f"acc: {n_correct/len(cases)*100:.2f}%")

if __name__ == "__main__":
    test()