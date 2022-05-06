# ml_with_graph_algorithms
## 🚧Work in Progress!🚧

## 前言
從零到有(使用圖演算法)建模，當中有很多必經開發過程，請協助將開發元件模組化，列出遇到的問題，並找出解法(附上參考資料)。

## 目標
團隊透過此repo協作開發，交流ML/Graph核心技術及建模方法，增強實作技能。

## repo更新方式
1. 每週從develop創建新的repo，依據該週的題號命名(feature/<題號>)，如：feature/q1
2. 個人/團隊可創建新的分支，建議命名方式為(feature/<題號>_<個人ID>)，如：feature/q1_20033
3. 該週討論完後請將個人/小組的分支推送到該週題號的對應分支，並刪除個人/小組分支
4. 該週結束前會將題號分支推送到develop分支，並保留題號分支

<img src="https://user-images.githubusercontent.com/10674490/142558203-0f6e4e36-9fbd-4d90-beb9-6a65eeca58fc.png" height="400">

## 作法
每週四(下午1:30-2:30)會議討論作法及相關參考資料。兩人一組，共同負責程式開發及說明。    
每週五(下午1:30-2:30)討論CS224W@Stanford課程([連結](http://web.stanford.edu/class/cs224w/))，repo[資料夾連結](https://github.com/udothemath/ml_with_graph_algorithms/tree/features/add_cs224/study_group)

|成員/日期|1/20(q8)|1/27(q9)|2/3(q10)|
|-|-|-|-|
|奕勳(a)|&#x1F34E;ab|ad|&#x1F34E;af|
|品瑜(b)|&#x1F34E;ab|&#x1F34E;be|bc|
|崇爾(c)|&#x1F34E;cd|&#x1F34E;cf|bc|
|彥銘(d)|&#x1F34E;cd|ad|&#x1F34E;de|
|甫璋(e)|ef|&#x1F34E;be|&#x1F34E;de|
|昕靜(f)|ef|&#x1F34E;cf|&#x1F34E;af|

|成員/日期|12/16(q5)|12/30(q6)|2022/1/13(q7)|
|-|-|-|-|
|崇爾(a)|&#x1F34E;ab|ad|&#x1F34E;af|
|品瑜(b)|&#x1F34E;ab|&#x1F34E;be|bc|
|甫璋(c)|&#x1F34E;cd|&#x1F34E;cf|bc|
|彥銘(d)|&#x1F34E;cd|ad|&#x1F34E;de|
|奕勳(e)|ef|&#x1F34E;be|&#x1F34E;de|
|昕靜(f)|ef|&#x1F34E;cf|&#x1F34E;af|

附註：標示&#x1F34E;的為該週分享組別

|成員/日期|11/25(q2)|12/2(q3)|12/9(q4)|
|-|-|-|-|
|甫璋(a)|&#x1F34E;ab|ad|&#x1F34E;af|
|品瑜(b)|&#x1F34E;ab|&#x1F34E;be|bc|
|奕勳(c)|&#x1F34E;cd|&#x1F34E;cf|bc|
|彥銘(d)|&#x1F34E;cd|ad|&#x1F34E;de|
|崇爾(e)|ef|&#x1F34E;be|&#x1F34E;de|
|昕靜(f)|ef|&#x1F34E;cf|&#x1F34E;af|

## 議題
|日期|編號|題目|分享組別|附註|
|-|-|-|-|-|
|2021/11/18|q1|將資料轉成圖演算法可以使用的格式|甫璋|-|
|2021/11/25|q2|產生節點(Node)以及關係(relation)的屬性|甫璋/品瑜, 奕勳/彥銘|-|
|2021/12/2|q3|建構完整圖神經網絡開發流程(暫時不需要關注成效)|品瑜/崇爾, 奕勳/昕靜|-|
|2021/12/9|q4|說明圖演算法中的GCN，GraphSage，以及GAT的程式碼|甫璋/昕靜, 彥銘/崇爾|-|
|2021/12/16|q5|實作GraphSage演算法於Link Prediction問題<br />實作GAT演算法於inductive問題|崇爾/品瑜<br />甫璋/彥銘|-|
|2021/12/30|q6|實作GraphSage並測試計算加速前後的效能比較|品瑜/奕勳, 甫璋/昕靜|-|
|2022/1/13|q7|PyTorch Lightning簡介<br />[PyG AutoScale Framework](https://arxiv.org/pdf/2106.05609.pdf)實作|崇爾/昕靜<br /> 彥銘/奕勳|-|
|2022/1/20|q8|介紹平行化處理工具 billiard<br />介紹玉山人工智慧公開挑戰賽建模嘗試|-|-|

## 預計新增議題
- 平行化處理
- GPU設定
- 開發流程模板
- 異質圖的建模方法
- 解釋模型訓練流程
- 解釋完整建模流程

## 待辦
新增commit共用格式
