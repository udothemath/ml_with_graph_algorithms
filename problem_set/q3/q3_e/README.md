# 說明

使用Recsys Challenge 2015資料(歐洲電商網站)，以一次user visit中所造訪的item建一張graph(item為node, item間的順序為edge)，預測是否購買該item(node classification)。

reference: [Hands-on Graph Neural Networks with PyTorch & PyTorch Geometric] (https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8)

主要程式檔YooChooseClick.ipynb
主要檔案存放於data/
根據指定的sample session_id個數處理後, 檔案會存放至data/yoochoose_click_binary_1M_sess.dataset

# Usage
1. 將kaggle上下載的資料放置於data/中
2. 執行YooChooseClick.ipynb
3. 指定想要的sample個數(預設為1000) 
sampled_session_id = np.random.choice(df.session_id.unique(),1000, replace=False)