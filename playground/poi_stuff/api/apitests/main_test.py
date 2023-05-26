"""
RUN API Test

NOTE:
1.　進stating apitest OK
2. local可顯示api結果

"""
import sys
import requests
import argparse
import os
import pprint
from inputs import input_data1
os.environ["PYTHONIOENCODING"] = "utf-8"

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--project", help="project_name")
parser.add_argument("-d", "--domain", help="domain")
parser.add_argument("-t", "--token", help="api_token")
parser.add_argument("-c", "--client", help="client")

args = parser.parse_args()
print(f"[Info] Project name: {args.project}")
print(f"[Info] Project domain: {args.domain}")
print(f"[Info] Project token: {args.token}")
print(f"[Info] Project client: {args.client}")

headers = {'Authorization': f'Bearer {args.token}',
           'X-Client-Id': f'{args.client}'}

REPO_NAME = 'poi-service'

try:
    if args.project is None and args.domain is None:
        url = 'http://localhost:8000/predict/v1'
        local = True
    else:
        url = f'https://{args.project}.{args.domain}/{REPO_NAME}/predict/v1'
        local = False
    res = requests.post(
        url=url,
        headers=headers,
        json=input_data1,
        verify=False,
        timeout=100
    )
    print(f"[Info] {res.status_code}")
    print(f"[Info] {res.json}")
except BaseException as e:
    print(f"[Error] Exception occurs. {str(e)}")
    sys.exit(1)
if res.status_code != 200:
    print(f"[Info] status_code is not equal to 200! it's {res.status_code}")
    print(f"[Info] json: {res.json}")
    print(f"[Info] text: {res.text}")
    sys.exit(1)
else:
    if local:
        pprint.pprint(res.json())
    print("[Info] You have successfully run the api test. END.")
