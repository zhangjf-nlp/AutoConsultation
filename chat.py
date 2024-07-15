import time
import openai
import tiktoken
import concurrent
import numpy as np

openai.api_key = ""
openai.num_tokens = 0

def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def num_tokens_of_string(string):
    messages = [{"role":"user", "content":string}]
    return num_tokens_from_messages(messages)


def chat(prompt, temperature=0.1):
    input_message = {"role": "user", "content": prompt}
    openai.num_tokens += num_tokens_from_messages([input_message])
    output_message = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[input_message],
        temperature=temperature,
    )["choices"][0]["message"]
    openai.num_tokens += num_tokens_from_messages([output_message])
    answer = output_message["content"]
    return answer


global spend_time_history
spend_time_history = [30] * 10
def chat_with_auto_retry(prompt, temperature=0.1, max_retry_times=3, min_interval_seconds=3, max_interval_seconds=30):
    global spend_time_history
    def chat_(prompt, timeout, temperature):
        executor = concurrent.futures.ThreadPoolExecutor()
        future = executor.submit(chat, prompt, temperature)
        try:
            a = future.result(timeout=timeout)
            return a
        except concurrent.futures.TimeoutError:
            print(f"chatgpt-api timed out after {timeout:.2f} seconds, retrying...")
            executor.shutdown(wait=False)
            timeout = timeout * 2
        except Exception as e:
            print(f"chatgpt-api raise exception, retrying... : {e}")
            time.sleep(1)
        return chat_(prompt, timeout, temperature)

    start_time = time.time()
    timeout = max(max_interval_seconds, np.mean(sorted(spend_time_history)[:8]))
    for retry_times in range(max_retry_times):
        answer = chat_(prompt, timeout*(retry_times+1), temperature)
        if answer is not None: break
    end_time = time.time()
    elapsed_time = end_time - start_time
    spend_time_history = spend_time_history[1:] + [elapsed_time]
    delayTime = min_interval_seconds - elapsed_time
    if delayTime > 0:
        time.sleep(delayTime)
    return answer

if __name__ == "__main__":
    import re

    prompt = """
    请根据参考资料，改写话术，为其中的药物/食物类型补充具体示例，示例如下：
    
    示例1.
    话术：颈椎病可能会让颈部血管受到压迫，导致大脑的血量供应不足，因此缺氧并感到头疼。可以采取物理治疗，如针灸和按摩等，或使用镇痛药进行缓解，如布洛芬和扑热息痛等。
    参考资料：
    镇痛药：布洛芬、阿司匹林、扑热息痛、曲马多等
    改写：颈椎病可能会让颈部血管受到压迫，导致大脑的血量供应不足，因此缺氧并感到头疼。可以采取物理治疗，如针灸和按摩等，或在医生的指导下，使用镇痛药进行缓解，如布洛芬、阿司匹林和扑热息痛等。
    
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
    话术：纳尔逊综合征是一种罕见的疾病，可能由于肾上腺切除术后垂体肿瘤引起。治疗方案包括使用糖皮质激素和抗肿瘤药物等，以及通过手术切除肿瘤来治疗。
    参考资料：
    抗肿瘤药物：环磷酰胺、顺铂、多柔比星等
    改写：纳尔逊综合征是一种罕见的疾病，可能由于肾上腺切除术后垂体肿瘤引起。治疗方案包括在医生的指导下使用糖皮质激素和抗肿瘤等药物，例如环磷酰胺、顺铂、多柔比星等，以及通过手术切除肿瘤来治疗。
    
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
    
    现在请改写
    话术：胃痛可能是由不良饮食习惯引起的，如暴饮暴食、吃太多辛辣食物等，对于这种情况的患者，建议调整饮食，少吃辛辣、油腻、刺激性食物，多吃易消化的食物，如米粥、面包、水果等。
    参考资料：
    辛辣食物：辣椒、花椒、咖喱、葱姜蒜等
    刺激性食物：咖啡、浓茶、白酒、冷饮、冰淇淋、生鱼片、生蚝等
    改写：
    """
    prompt = "\n".join([line.strip() for line in prompt.strip().split("\n")])
    prompt_for_query = prompt
    print(prompt_for_query)
    print("-" * 32)
    answer = chat_with_auto_retry(prompt_for_query, temperature=0.1)
    print(answer)