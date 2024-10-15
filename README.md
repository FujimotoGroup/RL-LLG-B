### 高速磁化反転のための外部磁場印加方法推定 ー 強化学習利用
### Code
- [ ] コメントを増やす
- ./x.py
  - 強化学習プログラム．磁場方向 x
  -   `python3 x.py`で実行

- ./xy.py
  - 強化学習プログラム．磁場方向 xy
  - `python3 xy.py`で実行

---

### Results
- [ ] ディレクトリの中身を説明する  
例えば  
./Results/H=x_dH=10_da=0.01_ani=(0,-10000,100)/  
というようなディレクトリがある  
意味としては  
H=[外部磁場の方向]  
dH=[１回の行動で変化させる磁場量　[Oe]]  
da=[行動の間隔　[ns]]  
ani=[磁気異方性定数　[Oe];　　+の場合は容易軸異方性　　-の場合は困難軸異方性]

---

### Prior_Research
- [ ] フォルダ名を変える
- [ ] それに伴って説明文を加える
先行研究との比較（結果のみ）
codeは./x.pyを使用

Bauer　：　"Switching behavior of a Stoner particle beyond the relaxation time limit", Phys. Rev. B **61**, 3410 (2000).

Schumacher　：　"Quasiballistic Magnetization Reversal", Phys. Rev. Lett. **90**, 017204 (2003).

