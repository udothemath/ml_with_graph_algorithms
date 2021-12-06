# Readme

- 這是從PyG的github上複製下來的檔案，目標是在做node classification（某篇paper屬於哪種領域的論文）。

- Model架構為先使用unsupervised方式train出node embeddings之後，再用LogisticRegression產出最後的預測結果。

- 圖的建法：node都是paper，edge代表兩篇文章有引用的關係。

- 只要有安裝完相關套件（torch-scatter, torch-sparse, torch-cluster, torch-geometric）即可執行。