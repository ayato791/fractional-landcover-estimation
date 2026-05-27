#

# Thesis repository

## 名前/ NAME

XXXX

## 研究タイトル/ Research title

決まったら書く

## このレポジトリについて / About this repository

このレポジトリは卒論・修論・博論のために使用します。
研究室内限定公開です。

This repository is used for bachelor's and master's theses, and doctoral dissertations.
It is restricted to within the research laboratory.


## Github レポジトリの整理/Organize your github repository

- Readmeをきちんと書いて、どのような分析をどのような手順でやったのかまとめてください。 コードは他人が見てわかりやすいように工夫してください。再現可能な環境を作るようにしてください。
- Colab上で実行したものついては、colabのファイル「名前.ipynb」をレポジトリに保存するのと同時に、[こちら](https://anvil.works/learn/tutorials/google-colab-github)のようにrequirementsファイルを作成してください。
- Earth Engine playground上のコードは、Githubのレポジトリ内に「名前.js」という形で保存・管理してください。
- PythonやR環境についてはDockerでまとめ、Githubレポジトリに保存・管理するようにしてください。再現性を保つために必要です。


- Write a proper Readme and summarize the analysis and procedures performed. Make the code as easy to understand as possible for others. Make sure to create an environment that can be reproduced using the code.
- For executions performed on Colab, in addition to saving the Colab file "name.ipynb" in the repository, please create the requirements file following [this guide](https://anvil.works/learn/tutorials/google-colab-github).
- For code on Earth Engine javascript playground, please save and manage it in the repository as "name.js".
- For the R or Python environment, consolidate it with Docker, and save and manage it in the Github repository. This is necessary to reproduce the environment.

## データ/Data

Githubでは大きなデータを置くことが難しいため、研究関連データは整理して（できる限りわかりやすく配置し、無駄な中間生成データなどは削除して）、自分のGoogle driveやGoogleclass roomの自分のフォルダにアップロードして、フォルダに「readme.md」ファイルを添付すること。

Since it is difficult to store data on Github, please organize research-related data (as clearly as possible and delete unnecessary intermediate data, etc.) and upload it to your own folder on Google Drive or Google Classroom.
Attach a readme.md file.

## 進捗報告について/ Progress Reports

weekly todosを毎週月曜9AMまでにissuesとして提出すること。
毎週、取り組むタスクを checkbox listとしてリストアップすること。
作成したsub-issueに対して、一つずつブランチを作成して、完了したら作業ブランチ（例えばdev）にpull requestして修正を反映させる。

Submit weekly todos as a Github issue by 9AM every Monday.
Develop a checkbox list for your weekly tasks.
Create a new branch when you work for an issue and pull request to merge your updates to your target branch (i.e., dev).

## 研究計画書の作成/ Writing a research proposal

研究計画書は`ResearchPlan.md`のテンプレートを使用すること。

The research plan should use the template `ResearchPlan.md`.

# 論文執筆フロー

## 論文執筆フロー/ Paper Writing Flow

```
1. 研究の目的をとりあえず書く。
2. アブストをとりあえず書く（[参考](https://github.com/su-giscience/Thesis_workplace/blob/master/Ref/Nature_abstract_example.pdf)）。
3. アブストに記載した内容を実現するために何をすべきかリストアップする。
4. リストアップしたものを参考に論文の骨子を固めていく。各章（はじめに（背景）、手法 、データ、結果、考察、おわりに（結論）のどこに当てはまるのかを整理する。
5. 論文を�条書きで書く。
6. 1-5を繰り返す。
7. 全体から個へ、できるところから作業（レビューや分析など）をすすめる。
8. 1-7を繰り返す。
```

```
1. Write down the purpose of your tentative research .
2. Write down a tentative abstract ([reference](https://github.com/su-giscience/Thesis_workplace/blob/master/Ref/Nature_abstract_example.pdf)).
3. List up what should be done to achieve the content described in the abstract.
4. Based on the listed items, solidify the outline of the paper. Rearrange them according to each chapter (Introduction (Background), Methodology, Data, Results, Discussion, Conclusion).
5. Write the paper in bullet points.
6. Repeat steps 1-5.
7. Proceed with the work (such as reviews and analysis) from the whole to the individual, starting with what can be done.
8. Repeat steps 1-7.
```
## Github flow

__注意:__ mainブランチは完成版を表すため、開発を進めないこと。
```
1. 新たなブランチを切る（例：Dev）。
2. 切ったブランチで適宜commitしながら書いていく。
3. レビューしてほしい段階まで作れたらpull requestsをmainブランチに向かって投げる。
4. Reviewerをアサインする（自分自身でも良い）。
5. Reviewerはpull requestが出されたものに対してレビューしコメントを残す。
6. Reviewerがコメントした内容一つ一つに対応し修正する
7. 次のレビューしてほしい段階まで来たら、レビューを依頼する。
8. すべて承認されたらmainブランチにmergeする。
```

__Note:__  Do not work on the main branch.　The main branch is always the completed version.
```
1. Create a new branch (e.g., Dev).
2. Update your codes or any other files in the created branch, committing as appropriate.
3. When you have reached a stage where you want it reviewed, submit a pull request to the main branch.
4. Assign a reviewer.
5. The reviewer will review the pull request and leave comments.
6. Address the comments and update the content, then resolve the comments.
7. Ask for a review when all revisions have been done.
8. When you reach the next stage for review... repeat the cycle.
```

## 実行環境構築 / Setting up the environment

再現性、後輩への引き継ぎのため、自分のPCが壊れたときのため、可能な限りdockerで研究分析環境を整備する。
**一つのレポジトリで研究環境を構築し完結すること。**

To ensure reproducibility and facilitate knowledge transfer to your juniors, you should establish a research and analysis environment using Docker as much as possible, in case your own PC breaks down.

The goal is to create a self-contained environment for your reseach within a single repository.

### R

R関連の環境はすでに本レポジトリで大体整っている。
適宜修正すること。

The R-related environment is already largely set up in this repository.
Please make the necessary modifications.

### Python

実行環境の雛形は[ここ](https://github.com/naru-T/MyPySpatial_build)を参照のこと。古いかも。

The template for the execution environment can be found [here](https://github.com/naru-T/MyPySpatial_build). Please refer to it.

### Tex

[latexmk](https://texwiki.texjp.org/?Latexmk)などを活用する。
[dockerとlatexmkなどを組み合わせることができるらしい](https://korosuke613.hatenablog.com/entry/2019/06/24/171246)。ただし[このあたり](https://acetaminophen.hatenablog.com/entry/2018/09/23/195200)も参照のこと。
もしくは[overleaf](https://ja.overleaf.com/)を使う。Compilerを「LaTex」に変更することに注意！
[テンプレート](https://ja.overleaf.com/read/jjpktvztqnqs)
編集可能なファイルを共有しますので堤田までリクエストしてください。

Use [Overleaf](https://ja.overleaf.com/).
Please note to change the compiler to "LaTex".
You can find a [template](https://ja.overleaf.com/read/jjpktvztqnqs) as well.
Request an editable file on Overleaf.


## 執筆ツール/ Writing tools

Githubで読めるツールを推奨
Please recommend tools that can be read on Github.

```
- LaTex: 推奨
- Markdown: 推奨
- Word: Tex、Markdownがどうしても苦手な場合のみ。ただしGithubで修正を管理できない。
```

```
- Tex: Recommended.
- Markdown: Recommended.
- Word: Only if you have a strong preference for Word. However, please be aware that you won't be able to manage edits on GitHub.
```

## 参考/References

ここにあげたものは何度も見返すと良い。

The items listed here should be reviewed multiple times for best results.

JP

- [研究法](https://youtu.be/vn0cL7fxYh8)
- [理科系の作文技術](https://www.amazon.co.jp/dp/4121006240/ref=cm_sw_em_r_mt_dp_RW1X306WP34STJ1ATP83)
- [日本語の作文技術](https://www.amazon.co.jp/dp/4121006240/ref=cm_sw_r_tw_dp_RW1X306WP34STJ1ATP83)
- [イシューよりはじめよ](https://www.amazon.co.jp/dp/B00MTL340G/ref=cm_sw_em_r_mt_dp_YNHWKDQBJKVX4GRM6AAW)

EN

- [How to write a great research paper](https://www.cis.upenn.edu/~sweirich/icfp-plmw15/slides/peyton-jones.pdf) [[video](https://youtu.be/WP-FkUaOcOM?si=fp2zYCOZ0Q0hA4kB)]
- https://www.uaar.edu.pk/fs/books/6.pdf

### tips

#### Dockerを使いこなせるようになろう/Let's become proficient in using Docker.

深くまで知る必要はない。使えるようになればよい。

There is no need to delve too deep. It is sufficient to become proficient.

- [Docker official](https://docs.docker.com/)
- [docker](https://docs.docker.com/install/)
- https://qiita.com/gold-kou/items/44860fbda1a34a001fc1
- https://qiita.com/zembutsu/items/24558f9d0d254e33088f
- https://qiita.com/Michinosuke/items/5778e0d9e9c04038903c

#### git/githubを使いこなせるようになろう/Let's become proficient in using git/github.

深くまで知る必要はない。使えるようになればよい。

There is no need to delve too deep. It is sufficient to become proficient.

- https://github.co.jp/features
- https://qiita.com/gold-kou/items/7f6a3b46e2781b0dd4a0
- https://qiita.com/ren0826jam/items/cf766a6fd43049b3cff4
- https://qiita.com/renesisu727/items/248cb9468a402c622003

#### MarkdownのためのVScode設定
[こちら](https://www.notion.so/sugiscience/Markdown-VScode-2082ad05e77d4ac6b7fae50e5854c841?pvs=4)を参照のこと


#### bibtexの使い方/How to use BibTeX

引用情報を管理し、文章に埋め込むのにbibtexを活用する。
bibtexはR Markdown, Markdown共に使い方は同じ。

Google scholarの場合
```
1. Google scholarで、「マイライブラリ」→「ラベルの管理」で原稿用のラベルを作成する（例：卒論）
2. 引用する論文をGoogle scholarで検索し、スターマークをつける。その際に作成したラベルを指定する
3. 「全てをエクスポート」からbibtexを選択すると、引用論文情報のbibtexが生成されるので、BBB.bibとして、AAA.mdと同じディレクトリに置く。
```
Markdown, Rmarkdownの場合、
引用は各bibtex内のエイリアス（./md/ref.bibを例にとると、「darwin1859」や「PhysRev.47.777」）に「@」をつけて、[]で囲む（例：`Darwin[@darwin1859]は`.... ）。
AAA.md内のYAMLに指定したBBB.bibに基づいてpandocが変換する際に「# References Bibliography」に引用文献リストを作成してくれる。
Texの場合`\cite{dawrin1859}`とする。

BibTeX is used to manage citation information and embed it in a document.
The usage of BibTeX is the same in R Markdown and Markdown.


For Google Scholar:
```
1. In Google Scholar, go to "My library" and create a label for your manuscript (e.g., "Thesis").
2. Search for the paper you want to cite in Google Scholar and mark it with a star. Specify the label you created.
3. Choose "Export all" and select BibTeX. This will generate a BibTeX file with the citation information. Save it as BBB.bib in the same directory as [AAA.md](http://aaa.md/).
```
To cite a reference by Markdown or Rmarkdown, add an "@" symbol to the alias in each BibTeX entry (e.g., "@darwin1859" or "@PhysRev.47.777") and enclose it in square brackets (e.g., `Darwin[@darwin1859]`).
When pandoc converts [AAA.md](http://aaa.md/) based on the BBB.bib specified in the YAML of [AAA.md](http://aaa.md/), it will create a bibliography list under "# References Bibliography".
In Tex, `\cite{darwin1859}`.

####  VScodeでコンテナに入り込見たい場合/If you want to enter a container in VScode

VScode extensionの[Remote-Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)と、[Docker](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)をインストールしてください。
起動しているDockerコンテナを右クリックし、「Attach Visual studio code」をクリックする。
新たなVScodeが立ち上がり、コンテナに入り込むことができる。

After installing the [Remote-Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension and [Docker](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code.
Right-click on a running Docker container and click "Attach Visual Studio Code".
A new VS Code window will open and you will be able to access the container.
