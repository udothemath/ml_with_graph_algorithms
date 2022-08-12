1. 拉base image 下來
可參考工合提供的base image
或是從既有的docker file 取得 base image
工合images list:https://docs.primehub.io/docs/guide_manual/images-list
工合registry:https://hub.docker.com/r/infuseai/docker-stacks/tags?page=1

```
docker pull infuseai/docker-stacks:tensorflow-notebook-v2-5-0-63fdf50a-gpu-cuda-11
```
2. 找到你拉下來的image
docker images
REPOSITORY                           TAG                                               IMAGE ID       CREATED         SIZE
infuseai/docker-stacks               tensorflow-notebook-v2-5-0-63fdf50a-gpu-cuda-11   2c2c71e89f86   12 months ago   19.5GB

3. run image 並在裡面做安裝測試
```
docker run -it --user root -p7687:7687 -p7474:7474 -p6379:6379 -p8080:8080 -p8888:8888 526090483388 bash
```
NOTE: add -pxxxx:xxxx to forward the port if you want to test an installed service. 

4. 進入container後可以做安裝套件測試，像是pip install XXX，apt-get XXX

5. 確認可以安裝後修改Dockerfile，把這些安裝指令加進去
```
RUN pip install --upgrade \
        matplotlib==3.0.2 \
        urllib3
```
6. dockerfile 修改完後將他建立成imags
```
docker build -f cpu.Dockerfile -t primehub.airgap:5000/[name]:Tagv1 .  
```
Tag為當日日期，後面記得要有一個.要記得！
E.g., 
```
docker build -f cpu.Dockerfile -t primehub.airgap:5000/esun-notebook:20220609
```
7. build完image後可以run進去看看套件有沒有順利裝成功
```
docker run -ti --rm primehub.airgap:5000/esun-notebook:20220609 bash
```
8. OK後把image save下來，放進USB
```
docker save primehub.airgap:5000/[name]:Tagv1  | gzip > update-images-Tagv1.tar.gz   
```
E.g., 
```
docker save primehub.airgap:5000/rapids-notebook:20220720 | gzip > update-images-20220720.tar.gz
```

NOTE:
- Comment these lines of dockerfile in your own computer when building the image: 
```
#----powerline
# COPY .bash_profile /etc/profile.d/bashrc.sh
##############################
# For PrimeHub Job Submission#
##############################
# USER $NB_UID
# ENV PATH $PATH:/opt/conda/bin
# ENV PIP_CONFIG_FILE /etc/python/pip.config
# COPY ai-cloud-pip.config /etc/python/pip.config
```
- When port forwarding fail, check if the reference solve the problem: https://github.com/microsoft/vscode-remote-release/issues/764#issuecomment-506983759 (set serving host from localhost to 0.0.0.0 in container)

# Nvidia Driver Installation: 

## 問題描述: 

下`nvidia-smi`以後會出現`Failed to initialize NVML: Driver/library version mismatch`
下`python` / `import cudf`則會出現: `system has unsupported display driver / cuda driver combination`

## 分析: 

### 相關解法:

https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch
https://forums.developer.nvidia.com/t/failed-to-initialize-nvml-driver-library-version-mismatch/190421/4

cat /proc/driver/nvidia/version
>> 470.42.01

dkms status
>> nvidia, 515.48.07: added

Try: 
apt purge nvidia* libnvidia*
apt install nvidia-driver-470
