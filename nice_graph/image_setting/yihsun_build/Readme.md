1. 拉base image 下來
可參考工合提供的base image
或是從既有的docker file 取得 base image
工合images list:https://docs.primehub.io/docs/guide_manual/images-list
工合registry:https://hub.docker.com/r/infuseai/docker-stacks/tags?page=1

    bicc@BICCdeMBP build_image % docker pull infuseai/docker-stacks:tensorflow-notebook-v2-5-0-63fdf50a-gpu-cuda-11
                tensorflow-notebook-v2-5-0-63fdf50a-gpu-cuda-11: Pulling from infuseai/docker-stacks

2. 找到你拉下來的image
docker images
REPOSITORY                           TAG                                               IMAGE ID       CREATED         SIZE
infuseai/docker-stacks               tensorflow-notebook-v2-5-0-63fdf50a-gpu-cuda-11   2c2c71e89f86   12 months ago   19.5GB

3. run image 並在裡面做安裝測試
docker run -ti --rm infuseai/docker-stacks:tensorflow-notebook-v2-5-0-63fdf50a-gpu-cuda-11 bash
jovyan@92f13f4b13b8:~$

4. 進入container後可以做安裝套件測試，像是pip install XXX，apt-get XXX

5. 確認可以安裝後修改Dockerfile，把這些安裝指令加進去
RUN pip install --upgrade \
        matplotlib==3.0.2 \
        urllib3

6. dockerfile 修改完後將他建立成imags
docker buile -f cpu.Dockerfile -t primehub.airgap:5000/[name]:Tagv1 .  #Tag為當日日期，後面記得要有一個.要記得！
docker buile -f cpu.Dockerfile -t primehub.airgap:5000/esun-notebook:20220609

7. build完image後可以run進去看看套件有沒有順利裝成功
docker run -ti --rm primehub.airgap:5000/esun-notebook:20220609 bash

8. OK後把image save下來，放進USB
docker save primehub.airgap:5000/[name]:Tagv1  | gzip > update-images-Tagv1.tar.gz   #Tag為當日日期
docker save primehub.airgap:5000/esun-notebook:20220609 | gzip > update-images-20220609.tar.gz   

