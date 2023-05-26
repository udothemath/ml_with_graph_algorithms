# %%
from datetime import date
import datetime
import json
import numpy as np
from api_service_predict import Operation
import pprint

VERBOSE = False


class NpEncoder(json.JSONEncoder):
    """ change return object type for json format requirement """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, date):
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)


input_json = {
    "business_unit": "C170",
    "request_id": "ca0a1a91-37e6-41d2-a03a-4ee82445dc40",
    "inputs": {
        "register_ids": {
            "nearby": [
                "good_1000"
            ],
            "self": []
        },
        "address": "地址",
        "lon": 123.23,
        "lat": 23.23
    }
}

print("Start of api run test")
a0 = datetime.datetime.now()
api = Operation()
ans = api.run(input_json)
if VERBOSE:
    pprint.pprint(ans)
ans_to_json = json.dumps(ans, indent=4, cls=NpEncoder, ensure_ascii=False)
a1 = datetime.datetime.now()
print(f'total time: {a1-a0}')

print(ans_to_json)

print("End of api run test")


# %%
