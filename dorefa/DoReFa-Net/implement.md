## データセット（モデル）一覧

- mnist
  - convnetあり
  - dorefaあり
- cifar
  - convnetあり
  - dorefaあり
- svhn
  - convnetあり
  - dorefaあり
- imagenet
  - convnetなし
  - dorefaあり

## タスク

上野くんがmnist、cifarデータセットに取り組んでいたので、続くのは以下のタスクかなと思います。

- svhnデータセットで、svhn-digit-convnet.pyとsvhn-digit-dorefa.pyがあるので、それぞれ実装して動かし、精度比較やモデル圧縮具合、学習時間などを見る
- alexnet-dorefa.py（dorefaで組む場合）とload-alexnet.py（普通に組む場合）にモデルがあるので、上と同様に、精度比較やモデル圧縮具合、学習時間などを見る
- load-vgg16.pyがあって、ネットワーク定義が書いてあるので、これを元に学習させた場合と、dorefaでバイナリ化して実装してみた場合とで、精度比較やモデル圧縮具合、学習時間などを見る

## 問題

- svhnデータセットってmnistくらい超軽いはずなのに計算が非常に遅い…3it/sくらいしか出ないのでCPUで回しているかのようなスピードなのがなぜなのかわからない…
