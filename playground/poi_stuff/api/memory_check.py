"""
IF Polaris
"""
from mlaas_tools2.api_tool import APIBase
from mlaas_tools2.api_exceptions import DependencyError
import os


class Operation(APIBase):
    """IF Polaris"""

    def __init__(self):
        super().__init__()
        self.response = {
            "status_code": "",
            "status_msg": "",
            "memory": ""
        }

        threshold = os.getenv('MEMORY_THRESHOLD')
        if threshold:
            self.mem_threshold = int(threshold)
        else:
            self.mem_threshold = 80

    def get_mem_info(self):
        """
        Use /sys/fs/cgroup/memory to get pod memory info
        """

        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            mem_total = f.read()
        with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
            mem_used = f.read()

        self.mem_usage = float("%.2f" % (100 * int(mem_used) / int(mem_total)))

    def run(self, inputs):
        """
        執行流程
        """
        try:
            if not inputs:
                self.logger.info("input is null")
            # initialize api response
            self.get_mem_info()
            self.response['memory'] = f"{self.mem_usage}"

            if self.mem_threshold < self.mem_usage:
                self.response['status_code'] = "0001"
                self.response['status_msg'] = "usage more than threshold"
                raise DependencyError(self.response)
            else:
                self.response['status_code'] = "0000"
                self.response['status_msg'] = "OK"

            # make predictions from inputs and return result
            return self.response
        except DependencyError:
            raise DependencyError(self.response)
        except Exception:
            self.logger.error("unexpected error occurred!", exc_info=True)
            self.response['status_code'] = '0001'
            self.response['status_msg'] = 'code error'
            return self.response
