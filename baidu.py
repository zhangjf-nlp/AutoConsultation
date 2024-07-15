import time
import requests
from bs4 import BeautifulSoup

def add_baidu_tasks(queries):
    for query in queries:
        baidu(query, max_retry_times=1)
        print(f"add task to baidu-api: {query}")
    return

from old_code_0831.utils import baidu_in_loop as baidu_old

def baidu(query, task_id=None, secret="cndzys_dazhong", max_retry_times=3, interval=5):
    """
    def try_to_get_result(secret, query, task_id):
        url = "http://press.cndzys.com/baidu-rank-api/query"
        if task_id is None:
            payload = {
                "query": query,
                "secret": secret
            }
        else:
            payload = {
                "task_id": task_id,
                "secret": secret
            }
        response = requests.post(url, data=payload)
        data = response.json()["data"]
        if type(data) is list:
            content = [BeautifulSoup(_["content"], 'html.parser').get_text() for _ in data]
        else:
            content = None
            task_id = data["task_id"]
        return content, task_id

    content, task_id = try_to_get_result(secret=secret, query=query, task_id=task_id)
    retry_times = 1
    while content is None and retry_times < max_retry_times:
        print(f"retry ({retry_times}/{max_retry_times}): sleep {interval} to wait result of {query} from baidu-api ...")
        time.sleep(interval)
        content, task_id = try_to_get_result(secret=secret, query=query, task_id=task_id)
        retry_times += 1
    """
    #if content is None and query is not None:
    if True:
        print(f"baidu-api failed, try old version ...")
        answers_ = []
        answers = baidu_old(query)
        for answer in answers:
            if not any([_ for _ in answers_ if _['abstract'] == answer['abstract']]):
                answers_.append(answer)
        content = [answer['abstract'] for answer in answers]

    #content = [] if content is None else content
    return content, task_id


if __name__ == "__main__":
    for task_id in range(1, 40):
        print(f"task_id: {task_id}")
        try:
            print(f"task: {baidu(query=None, task_id=str(task_id), max_retry_times=10, interval=60)}")
        except Exception as e:
            print(e)