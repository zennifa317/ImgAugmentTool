# ImgAugmentTool
データ拡張のための画像処理と座標変換

### yolo_fomat.py
- yolo形式のアノテーションファイルを扱うクラス
- 仕様
    - インスタンス変数
        - self.imgs　画像idをキーに画像情報を格納する辞書型
            - img_name ファイル名
            - path パス
            - height 高さ
            - width 幅
        - self.anns　画像idをキーにアノテーション情報を格納する辞書型
            - cat_id　カテゴリーid
            - bbox　バウンディングボックス座標
        - self.cats　
   - メソッド