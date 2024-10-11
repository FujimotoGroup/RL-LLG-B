### 高速磁化反転のための外部磁場印加方法推定 ー 強化学習利用
### Code
./x.py　：　強化学習プログラム．磁場方向 x

./xy.py　：　強化学習プログラム．磁場方向 xy

./Fig.py　：　Result 内の text データからグラフを表示

./m_3Dplot.py　：　text データから磁化を３次元表示

./LLG_Hconst.py　：　一定磁場を加えた場合の磁化ダイナミクスをグラフ表示

./LLG_Hpulse.py　：　パルス磁場を加えた場合の磁化ダイナミクスをグラフ表示

./LLG_Hchange.py　：　磁場を線形に変化させた場合の磁化ダイナミクスをグラフ表示

ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
### Result
H=  ー　外部磁場の方向

dH=　ー　１回の行動で変化させる磁場量　[Oe]

da=　ー　行動の間隔　[ns]

ani= ー　磁気異方性定数　[Oe]　　+ 容易軸　　- 困難軸

ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
### Prior_Research
先行研究との比較

Bauer　：　"Switching behavior of a Stoner particle beyond the relaxation time limit", Phys. Rev. B **61**, 3410 (2000).

Schumacher　：　"Quasiballistic Magnetization Reversal", Phys. Rev. Lett. **90**, 017204 (2003).

